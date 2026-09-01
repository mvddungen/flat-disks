#ifndef RANDOM_H
#define RANDOM_H

#include <time.h>
#include <stdint.h>
#include <unistd.h> 
#include <cmath>

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

static inline unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

static inline unsigned long getseed()
{
	return mix(clock(), time(NULL), getpid());
}


class SplitMix {
private:
	uint64_t x;
public: 
	SplitMix(uint64_t seed) : x(seed) {}
	uint64_t operator()() {
		uint64_t z = (x += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}
};

class Xoshiro256PlusPlus {
private:
	uint64_t s[4];
	uint64_t rotl(const uint64_t x, int k) 
	{
		return (x << k) | (x >> (64 - k));
	}
public:
	Xoshiro256PlusPlus(uint64_t seed) 
	{
		SplitMix rng(seed);
		s[0] = rng();
		s[1] = rng();
		s[2] = rng();
		s[3] = rng();
	}
	uint64_t operator()() 
	{
		const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

		const uint64_t t = s[1] << 17;

		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];

		s[2] ^= t;

		s[3] = rotl(s[3], 45);

		return result;
	}
};

// generate a uniform random integer in 0, ...., range-1
template<typename RNG>
static inline uint32_t uniform_int(RNG& rng, uint32_t range) {
    uint32_t x = rng();
    uint64_t m = uint64_t(x) * uint64_t(range);
    uint32_t l = uint32_t(m);
    if (l < range) {
        uint32_t t = -range;
        if (t >= range) {
            t -= range;
            if (t >= range) 
                t %= range;
        }
        while (l < t) {
            x = rng();
            m = uint64_t(x) * uint64_t(range);
            l = uint32_t(m);
        }
    }
    return m >> 32;
}

// See: http://prng.di.unimi.it/
static inline double uint64_to_double(uint64_t x) {
	return (x >> 11) * (1. / (UINT64_C(1) << 53));
}

// generate real number in [0,1)
template<typename RNG>
static inline double uniform_real(RNG& rng) {
    return uint64_to_double(rng());
}

// generate real number in [min,max)
template<typename RNG>
static inline double uniform_real(RNG& rng, double min, double max) {
	return min + (max-min) * uniform_real(rng);
}

//Box muller method, see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
template<typename RNG>
double random_normal(RNG& rng, double mean, double sigma)
{
	const double two_pi = 2.0*3.14159265358979323846;

	double u1, u2;
	u1 = uniform_real(rng);
	u2 = uniform_real(rng);

	double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
	return z0 * sigma + mean;
}

template<typename RNG>
double random_bernoulli(RNG& rng, double probTrue)
{
    return uniform_real(rng) <= probTrue;
}

template<typename RNG>
int random_choice(RNG& rng, std::vector<double> & weights) 
{
    double total = 0.0;
    for( auto w : weights )
        total += w;
    
    double p = uniform_real(rng,0.0,total);
    for(int i=0,endi=weights.size();i<endi;++i)
    {
        p -= weights[i];
        if( p <= 0.0 )
            return i;
    }
    return 0;
}


#endif
