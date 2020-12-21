#include <iostream>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>

using namespace std;

__global__ void binomial_kernel(double *S, double *kvpq, double *prices, int size, int nsteps, bool am_b, bool put_b);
double *binomial_model_gpu(double *S0, double K, double T, double t, double qd, double r, double sigma, int nsteps, bool am_b, bool put_b, int size);
double binomial_model_cpu(double S0, double K, double T, double t, double qd, double r, double sigma, int nsteps, bool am_b, bool put_b);
void fill_array(double *a, int len, double low, double high);

int ss = 1000;

int main(int argc, char **argv)
{
	double *S = new double[ss];
	double K = 60;
	double T = 200.0 / 365.0;
	double t = 0;
	double qd = 0.02;
	double r = 0.03;
	double sigma = 0.2;
	int nsteps = ss;
	bool am_b = true;
	bool put_b = false;

	fill_array(S, ss, 0, 150);

	clock_t start = clock();
	for (int i = 0; i < ss; i++)
		binomial_model_cpu(S[i], K, T, t, qd, r, sigma, nsteps, am_b, put_b);
	clock_t end = clock();
	cout << "The CPU takes " << (end - start) << " milliseconds to run " << ss << " binomial models with " << ss << " steps each" << endl;
	
	start = clock();
	binomial_model_gpu(S, K, T, t, qd, r, sigma, nsteps, am_b, put_b, 1000);
	end = clock();
	cout << "The GPU takes " << (end - start) << " milliseconds to run " << ss << " binomial models with " << ss << " steps each" << endl;

	return 0;
}

__global__ void binomial_kernel(double *S, double *kvpq, double *prices, int size, int nsteps, bool am_b, bool put_b)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size) return;

	double *SF = NULL;
	
	while (!SF)
		SF = new double[nsteps + 1];

	// initialize option value array with values at expiry  
	for (int i = 0; i <= nsteps; i++)
	{
		double Si = S[id] * exp(kvpq[1] * (2.0f * i - nsteps));
		if (!put_b)
			SF[i] = (Si - kvpq[0]) > 0 ? (Si - kvpq[0]) : 0;
		else
			SF[i] = (kvpq[0] - Si) > 0 ? (kvpq[0] - Si) : 0;
	}

	// for each time step    
	for (int j = nsteps; j > 0; j--)
	{
		// for each node at the current time step    
		for (int i = 0; i < j; i++)
		{
			if (am_b)
			{
				double cv = kvpq[3] * SF[i] + kvpq[2] * SF[i + 1], sv = 0.0;
				if (!put_b)
					sv = (SF[i] * exp(kvpq[1]) - kvpq[0]) > 0 ? (SF[i] * exp(kvpq[1]) - kvpq[0]) : 0;
				else
					sv = (kvpq[0] - SF[i] * exp(kvpq[1])) > 0 ? (kvpq[0] - SF[i] * exp(kvpq[1])) : 0;
				SF[i] = max(cv, sv);
			}
			else
				// compute the discounted expected value of the option      
				SF[i] = kvpq[3] * SF[i] + kvpq[2] * SF[i + 1];
		}
	}

	delete SF;
	prices[id] = SF[0];
}

void fill_array(double *a, int len, double low, double high)
{
	double step = (high - low) / (len - 1);
	for (int i = 0; i < len; i++)
		a[i] = low + step * i;
}

double *binomial_model_gpu(double *S0, double K, double T, double t, double qd, double r, double sigma, int nsteps, bool am_b, bool put_b, int size)
{
	double *s_values, *kvpq, *prices, kvpq_cpu[4], *prices_cpu = new double[size];

	double dt = T / nsteps;
	double vdt = sigma * sqrt(dt);
	double pu = (exp((r - qd) * dt) - exp(-vdt)) / (exp(vdt) - exp(-vdt));
	double p = pu * exp(-r * dt);
	double q = (1.0 - pu) *  exp(-r * dt);

	kvpq_cpu[0] = K;
	kvpq_cpu[1] = vdt;
	kvpq_cpu[2] = p;
	kvpq_cpu[3] = q;

	// allocate and copy memory to the gpu
	if (cudaMalloc(&s_values, sizeof(double) * size) != cudaSuccess)
	{
		cout << "[+] Unable to allocate GPU memory" << endl;
		return NULL;
	}
	if (cudaMalloc(&kvpq, sizeof(double) * 4) != cudaSuccess)
	{
		cout << "[+] Unable to allocate GPU memory" << endl;
		return NULL;
	}
	if (cudaMalloc(&prices, sizeof(double) * size) != cudaSuccess)
	{
		cout << "[+] Unable to allocate GPU memory" << endl;
		return NULL;
	}
	if (cudaMemcpy(s_values, S0, sizeof(double) * size, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "[+] Error in moving memory to the GPU" << endl;
		return NULL;
	}
	if (cudaMemcpy(kvpq, kvpq_cpu, sizeof(double) * 4, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "[+] Error in moving memory to the GPU" << endl;
		return NULL;
	}

	binomial_kernel<<< (size / 128) + 1, 128 >>>(s_values, kvpq, prices, size, nsteps, am_b, put_b);
	cudaMemcpy(prices_cpu, prices, sizeof(double)*size, cudaMemcpyDeviceToHost);
	
	return prices_cpu;
}

double binomial_model_cpu(double S0, double K, double T, double t, double qd, double r, double sigma, int nsteps, bool am_b, bool put_b)
{
	double *S = new double[nsteps + 1];
	double dt = T / nsteps;
	double u = exp(sigma * sqrt(dt));
	double d = 1 / u;
	double disc = exp(r * dt);
	double pu = (exp((r - qd) * dt) - d) / (u - d);

	// initialize option value array with values at expiry  
	for (int i = 0; i <= nsteps; i++)
	{
		double Si = S0 * pow(u, 2 * i - nsteps);
		if (!put_b)
			S[i] = max(0.0, Si - K);
		else
			S[i] = max(0.0, K - Si);
	}
	
	// for each time step    
	for (int j = nsteps; j > 0; j--)
	{
		// for each node at the current time step    
		for (int i = 0; i < j; i++)
		{
			if (am_b)
			{
				double cv = ((1 - pu) * S[i] + pu * S[i + 1]) / disc, sv = 0.0;
				if (!put_b)
					sv = max(0.0, S[i] * u - K);
				else
					sv = max(0.0, K - S[i] * u);
				S[i] = max(cv, sv);
			}
			else
				// compute the discounted expected value of the option      
				S[i] = ((1 - pu) * S[i] + pu * S[i + 1]) / disc;
		}
	}

	return S[0];
}