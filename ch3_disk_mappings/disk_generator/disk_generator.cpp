#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <complex>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "random.h"
#include "args.hxx"

typedef Xoshiro256PlusPlus RNG;

enum class Direction { E, N, W, S };

const Direction oppositeDirection[4] = {Direction::W, Direction::S, Direction::E,Direction::N};
const Direction ccwDirection[4] = {Direction::N, Direction::W, Direction::S,Direction::E};
const Direction cwDirection[4] = {Direction::S, Direction::E, Direction::N,Direction::W};

struct Edge {
    int id;
    Direction dir;
};

Edge rotateCCW(Edge e) {
    return {e.id,ccwDirection[(int)e.dir]};
}
Edge rotateCW(Edge e) {
    return {e.id,cwDirection[(int)e.dir]};
}

struct IntVec2D {
    int x,y;
    
    friend bool operator<(const IntVec2D& l, const IntVec2D& r)
    {
        return l.y < r.y || (l.y == r.y && l.x < r.x);
    }
    friend bool operator==(const IntVec2D& l, const IntVec2D& r)
    {
        return l.y == r.y && l.x == r.x;
    }
    IntVec2D operator+(const IntVec2D& rVec) const
    {
        return {x + rVec.x, y + rVec.y};
    }
};

const IntVec2D directionVectors[4] = {{1,0},{0,1},{-1,0},{0,-1}};

struct Square {
    int id;
    int nbr[4]; // East-North-West-South
    IntVec2D pos;
};

IntVec2D walktotal(std::vector<IntVec2D>::iterator begin, std::vector<IntVec2D>::iterator end) {
    return std::accumulate(begin,end,IntVec2D{0,0});
}

// produce a uniform random simple walk of length 2*n + |end| ending at end
std::vector<int> randomBridge(int n, int end, RNG &rng) {
    int up = n + (end > 0 ? end : 0) ;
    int down = n + (end < 0 ? -end : 0 );
	std::vector<int> steps;
	steps.reserve(up+down);
	while( up + down > 0 ) {
		if (uniform_int(rng,up+down) < up) {
			steps.push_back(1);
			--up;
		} else {
			steps.push_back(-1);
			--down;
		}
	}
	return steps;
}

std::vector<IntVec2D> halfplaneExcursion(int n,RNG &rng) {
    std::vector<int> x = randomBridge(n-1,-1,rng);
    std::vector<int> y = randomBridge(n-1,-1,rng);
    std::vector<IntVec2D> walk;
    walk.reserve(2*n-1);
    IntVec2D minp = {0,0}, curp = {0,0};
    int minindex = 0;
    for(int i=0;i<2*n-1;++i) {
        walk.push_back({(x[i]+y[i])/2,(x[i]-y[i])/2});
        curp = curp + walk.back();
        if( curp < minp )
        {
            minp = curp;
            minindex = i+1;
        }
    }
    std::rotate(walk.begin(),walk.begin()+minindex,walk.end());
    return walk;
}

void purgeSubexcursions(std::vector<IntVec2D> & walk) {
    std::stack<std::pair<IntVec2D,std::vector<IntVec2D>::iterator>> ladder;
    IntVec2D curp = {0,0};
    ladder.push({curp,walk.begin()});
    auto it = walk.begin();
    while( it +1 != walk.end() ) {
        curp = curp + *it;
        if( ladder.empty() || ladder.top().first < curp ) {
            ladder.push({curp,it+1});
            it++;
        } else {
            while( curp < ladder.top().first ) {
                 ladder.pop();
            } 
            if( !ladder.empty() && curp == ladder.top().first ) {
                walk.erase( ladder.top().second, it+1);
                it = ladder.top().second;
            } else
            {
                ladder.push({curp,it+1});
                it++;
            }
        }
    }
}

struct Subdisk {
    std::vector<IntVec2D>::iterator start, end; // iterators indicating the corresponding subexcursion in the walk
    IntVec2D total; // the total increment of the subexcursion
    Edge e; // indicates the west-most edge to which to glue the subdisk
};

// Consider the walk from (0,0) determined by the steps following start. Find the first time it leaves H and return
// the iterator to the step that leaves H and the final position.
std::pair<std::vector<IntVec2D>::iterator,IntVec2D> findNextDescendingLadderPoint(std::vector<IntVec2D>::iterator start) {
    IntVec2D curp{0,0};
    while( !(curp < IntVec2D{0,0}) )
    {
        curp = curp + *start;
        start++;
    }
    return {start,curp};
} 
std::pair<std::vector<IntVec2D>::iterator,IntVec2D> findReverseDescendingLadderPoint(std::vector<IntVec2D>::iterator start) {
    IntVec2D curp{0,0};
    while( !(IntVec2D{0,0} < curp ) )
    {
        curp = curp + *start;
        start = start - 1;
    }
    return {start,curp};
} 

void printMma(std::vector<IntVec2D>::iterator begin, std::vector<IntVec2D>::iterator end) {
    std::cout << "{";
    bool first = true;
    for( auto it = begin;it!= end;it++) {
        std::cout << (first?"":",") << "{" << it->x << ", " << it->y << "}";
        first = false;
    }
    std::cout << "}\n";
}

class Polygon {
public:
    Polygon() {}
    void printwalk() {
        IntVec2D curp{0,0};
        for( auto x : walk_) {
            curp = curp + x;
            std::cout << "(" << x.x << ", " << x.y << ") -> " << curp.x << "," << curp.y << "\n";
        }
    }
    Square & newSquare() {
        sq_.push_back(Square{(int)sq_.size(),{-1,-1,-1,-1},{0,0}});
        return sq_.back();
    }
    void setAdjacent(Edge e, int sqid) {
        sq_[e.id].nbr[(int)e.dir] = sqid;
        sq_[sqid].nbr[(int)oppositeDirection[(int)e.dir]] = e.id;
    }
    Edge moveForward(Edge e) {
        return Edge{sq_[e.id].nbr[(int)e.dir],e.dir};
    }
    void printwalkMma() {
        printMma(walk_.begin(),walk_.end());
    }
    void printSquares(bool mma) {
        bool first = true;
        if( !mma )
            std::cout << sq_.size() << "\n";
        for( auto s : sq_) {
            if( mma )
                std::cout << (first ? "{":",") << "{{" << s.pos.x << "," << s.pos.y << "},{" << s.nbr[0] << "," << s.nbr[1] << "," << s.nbr[2] << "," << s.nbr[3] << "}}";
            else    
                std::cout << s.pos.x << " " << s.pos.y << " " << s.nbr[0] << " " << s.nbr[1] << " " << s.nbr[2] << " " << s.nbr[3] << "\n";
            first = false;
        }
        if( mma )
            std::cout << "}\n";
    }
    int generate(int n, RNG &rng) {
        walk_ = halfplaneExcursion(n,rng);
        purgeSubexcursions(walk_);
        return (walk_.size()+1)/2;
    }
    int generateSize(int nmin, int nmax, int len, RNG &rng) {
        int size = 0;
        while( size < nmin || size > nmax ) {
            size = generate(len,rng);
        }
        return size;
    }
    void generatePolygon() {
        std::stack<Subdisk> subs;
        subs.push(Subdisk{walk_.begin(),walk_.end(),walktotal(walk_.begin(),walk_.end()),{-1,Direction::N}});
        while( !subs.empty() )
        {
            Subdisk s = subs.top();
            subs.pop();
            if( s.start +1 == s.end )
            {
                // this is a boundary edge
                continue;
            }
            assert(s.total.y==0);
            assert(s.total.x!=0);
            if( s.total.x > 0 ) {
                // walk in H from (-total.x,0) to (0,0) starting with a North step
                assert(s.start->x==0 && s.start->y==1);
                auto it = s.start+1;
                std::vector<Subdisk> leftdisks;
                int leftwidth = 0;
                while( true) {
                    auto excursion = findNextDescendingLadderPoint(it);
                    leftdisks.push_back(Subdisk{it,excursion.first,excursion.second,{-1,Direction::N}});
                    it = excursion.first;
                    if( excursion.second.y == -1 ) {
                        break;
                    }
                    leftwidth += -excursion.second.x;
                }
                // now we have stepped back to height 0
                assert((it-1)->x == 0 && (it-1)->y == -1);
                std::vector<Subdisk> rightdisks;
                int rightwidth = 0;
                while( it != s.end ) {
                    auto excursion = findNextDescendingLadderPoint(it);
                    rightdisks.push_back(Subdisk{it,excursion.first,excursion.second,{-1,Direction::N}});
                    it = excursion.first;
                    rightwidth += -excursion.second.x;
                }
                // it remains to disect the last of the leftdisks
                std::vector<Subdisk> oppositedisks;
                it = leftdisks.back().end - 2;
                int width = 0;
                while( true ) {
                    auto excursion = findReverseDescendingLadderPoint(it);
                    oppositedisks.push_back(Subdisk{excursion.first+1,it+1,excursion.second,{-1,Direction::S}});
                    width += excursion.second.x;
                    it = excursion.first;
                    if( it + 1 == leftdisks.back().start ) {
                        break;
                    }
                }
                leftdisks.pop_back();

                // add a row of squares of size width
                assert( leftwidth >=0 );
                assert( rightwidth >= 0);
                assert( width >= 1 );
                assert( leftwidth + rightwidth + std::abs(s.total.x) == width ); 
                int prev = -1;
                int firstid = sq_.size();
                IntVec2D pos{0,0};
                if( s.e.id != -1 ) {
                    pos = sq_[s.e.id].pos + directionVectors[(int)s.e.dir];
                }
                pos.x -= leftwidth;
                for(int i=0;i<width;i++) {
                    Square & thissquare = newSquare();
                    thissquare.nbr[(int)Direction::W] = prev;
                    thissquare.pos = pos;
                    if( prev != -1 )
                    {
                        sq_[prev].nbr[(int)Direction::E] = thissquare.id;
                    }
                    prev = thissquare.id;
                    pos.x += 1;
                }
                int curid = firstid;
                for(auto it = leftdisks.rbegin();it!=leftdisks.rend();it++) {
                    it->e = Edge{curid,Direction::N};
                    curid += -it->total.x;
                }
                // glue strip appropriately
                Edge curedge = s.e;
                for(int i = 0;i<s.total.x;i++) {
                    if( curedge.id != -1 ) {
                        setAdjacent(curedge,curid);
                        curedge = Edge{sq_[curedge.id].nbr[(int)Direction::E],curedge.dir};
                    } else {
                        root_ = Edge{curid,Direction::N};
                    }
                    curid++;
                }
                for(auto it = rightdisks.rbegin();it!=rightdisks.rend();it++) {
                    it->e = Edge{curid,Direction::N};
                    curid += -it->total.x;
                }
                assert(curid == (int)sq_.size());
                curid = firstid;
                for(auto it = oppositedisks.rbegin();it!=oppositedisks.rend();it++) {
                    it->e = Edge{curid,Direction::S};
                    curid += it->total.x;
                }


                // push subdisks onto the queue in counterclockwise order
                for(auto d = rightdisks.rbegin(); d != rightdisks.rend(); d++ )
                    subs.push(*d);
                for(auto d : oppositedisks)
                    subs.push(d);
                for(auto d = leftdisks.rbegin(); d != leftdisks.rend(); d++ )
                    subs.push(*d);
            } else
            {
                // walk in H from (0,0) to (s.total.x<0,0) ending with a South step
                assert((s.end-1)->x==0 && (s.end-1)->y==-1);
                auto it = s.end-2;
                std::vector<Subdisk> leftdisks;
                int leftwidth = 0;
                while( true) {
                    auto excursion = findReverseDescendingLadderPoint(it);
                    leftdisks.push_back(Subdisk{excursion.first+1,it+1,excursion.second,{0,Direction::N}});
                    it = excursion.first;
                    if( excursion.second.y == 1 ) {
                        break;
                    }
                    leftwidth += excursion.second.x;
                }
                // now we have stepped back to height 0
                assert((it+1)->x == 0 && (it+1)->y == 1);
                std::vector<Subdisk> rightdisks;
                int rightwidth = 0;
                while( it+1 != s.start ) {
                    auto excursion = findReverseDescendingLadderPoint(it);
                    rightdisks.push_back(Subdisk{excursion.first+1,it+1,excursion.second,{0,Direction::N}});
                    it = excursion.first;
                    rightwidth += excursion.second.x;
                }
                // it remains to disect the last of the leftdisks
                std::vector<Subdisk> oppositedisks;
                it = leftdisks.back().start + 1;
                int width = 0;
                while( true ) {
                    auto excursion = findNextDescendingLadderPoint(it);
                    oppositedisks.push_back(Subdisk{it,excursion.first,excursion.second,{0,Direction::S}});
                    width += -excursion.second.x;
                    it = excursion.first;
                    if( it == leftdisks.back().end ) {
                        break;
                    }
                }
                leftdisks.pop_back();


                // add a row of squares of size width
                assert( leftwidth >=0 );
                assert( rightwidth >= 0);
                assert( width >= 1 );
                assert( leftwidth + rightwidth + std::abs(s.total.x) == width ); 
                int prev = -1;
                int firstid = sq_.size();
                IntVec2D pos{0,0};
                if( s.e.id != -1 ) {
                    pos = sq_[s.e.id].pos + directionVectors[(int)s.e.dir];
                }
                pos.x -= leftwidth;
                for(int i=0;i<width;i++) {
                    Square & thissquare = newSquare();
                    thissquare.nbr[(int)Direction::W] = prev;
                    thissquare.pos = pos;
                    if( prev != -1 )
                    {
                        sq_[prev].nbr[(int)Direction::E] = thissquare.id;
                    }
                    prev = thissquare.id;
                    pos.x += 1;
                }
                int curid = firstid;
                for(auto it = leftdisks.rbegin();it!=leftdisks.rend();it++) {
                    it->e = Edge{curid,Direction::S};
                    curid += it->total.x;
                }
                // glue strip appropriately
                Edge curedge = s.e;
                for(int i = 0;i<-s.total.x;i++) {
                    if( curedge.id != -1 ) {
                        setAdjacent(curedge,curid);
                        curedge = Edge{sq_[curedge.id].nbr[(int)Direction::E],curedge.dir};
                    } else {
                        root_ = Edge{curid,Direction::S};
                    }
                    curid++;

                }
                for(auto it = rightdisks.rbegin();it!=rightdisks.rend();it++) {
                    it->e = Edge{curid,Direction::S};
                    curid += it->total.x;
                }
                assert(curid == (int)sq_.size());
                curid = firstid;
                for(auto it = oppositedisks.rbegin();it!=oppositedisks.rend();it++) {
                    it->e = Edge{curid,Direction::N};
                    curid += -it->total.x;
                }

                // push subdisks onto the stack in counterclockwise order
                for(auto d : leftdisks )
                    subs.push(d);
                for(auto d = oppositedisks.rbegin(); d != oppositedisks.rend(); d++ )
                    subs.push(*d);
                for(auto d : rightdisks)
                    subs.push(d);
            }
        }
    }
    IntVec2D total() {
        return std::accumulate(walk_.begin(),walk_.end(),IntVec2D{0,0});
    }
    void checkSquares() {
        int bdry = 0;
        for(int i=0,endi=sq_.size();i<endi;i++) {
            assert( sq_[i].id == i);
            for( int j=0;j<4;j++) {
                assert( sq_[i].nbr[j] == -1 || sq_[sq_[i].nbr[j]].nbr[(int)oppositeDirection[j]] == i);
                if( sq_[i].nbr[j] == -1 ) 
                    bdry++;
            }
        }
        assert(bdry == (int)walk_.size()+1 );
    }
    void printBoundary(bool mma) {
        Edge curedge = root_;
        bool first = true;
        if( !mma )
            std::cout << walk_.size()+1 << "\n";
        do {
            if( mma )
                std::cout << (first?"{":",") <<"{" << curedge.id << "," << (int)curedge.dir << "}";
            else
                std::cout << curedge.id << " " << (int)curedge.dir << "\n";
            first = false;
            curedge = rotateCCW(curedge);
            while( sq_[curedge.id].nbr[(int)curedge.dir] != -1 ) {
                curedge = rotateCW(moveForward(curedge));
            }
        } while(!(curedge.id == root_.id && curedge.dir == root_.dir));
        if( mma )
            std::cout << "}\n";
    }
private:
    std::vector<IntVec2D> walk_;
    std::vector<Square> sq_;
    Edge root_;
};

int main(int argc, char **argv)
{
    args::ArgumentParser parser("MCMC simulation of Euclidean or hyperbolic disks");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> size_arg(parser,"size", "The minimal (half) number of sides of the disk", {'n',"size"});
    args::ValueFlag<int> maxsize_arg(parser,"size", "The maximal (half) number of sides of the disk", {'m',"maxsize"});
    args::ValueFlag<int> walk_arg(parser,"walk", "The (half) number of steps of the random walk used", {'l',"walk"});
    args::ValueFlag<unsigned long> seed_arg(parser,"seed", "Seed for random number generator", {"seed"});
    args::Flag mma_arg(parser, "mma", "Use mathematica syntax", {"mma"});
    try {
        parser.ParseCLI(argc,argv);
    }
    catch( args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    int size = size_arg ? args::get(size_arg) : 10;
    int maxsize = maxsize_arg ? args::get(maxsize_arg) : (int)(2*size);
    int walklength = walk_arg ? args::get(walk_arg) : (int)(1.5*maxsize);
    unsigned long seed = seed_arg ? args::get(seed_arg) : getseed();
    
    Xoshiro256PlusPlus rng(seed);

    Polygon pol;
    pol.generateSize(size,maxsize,walklength,rng); 
    pol.generatePolygon();
    if(mma_arg )
        std::cout << "{";
    pol.printSquares(mma_arg);
    if(mma_arg)
        std::cout << ",";
    pol.printBoundary(mma_arg);
    if(mma_arg)
        std::cout << "}\n";
    
    return 0;
}