#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Run in a single block
// Calculate bilayer COM
extern "C" __global__ void computeOrigin( 
const real4* __restrict__ posq,
double* origin,
double* origin_buffer,
const int* __restrict__ particles_for_origin, 
const double* __restrict__ mass_for_origin
) {
    int threadIndex = threadIdx.x;

    origin_buffer[threadIndex] = 0.0;

    // Calculate COM
    for (int index=threadIndex; index<ORIGIN_NUM_ATOMS; index+=blockDim.x) {
        origin_buffer[threadIndex] += mass_for_origin[index] * posq[particles_for_origin[index]].z;
    }
    __syncthreads();

    // Do parallel reduction for origin_buffer
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIndex < stride) {
            origin_buffer[threadIndex] += origin_buffer[threadIndex + stride];
        }
        __syncthreads();
    }
    __syncthreads();
    
    if (threadIndex == 0) {
        origin[0] = origin_buffer[0] / total_mass_origin;
    }
}

// Run in a single block
extern "C" __global__ void computePreFFtotal(
double* A_real_xray_current, 
double* A_real_neutron_current, 
double* A_complex_xray_current, 
double* A_complex_neutron_current
) {
    int threadIndex = threadIdx.x;

    // Init A_current
    // Set the A to 0.0 to avoid porblems on reduction step if blockDim.x > Nq_xray or Nq_neutron
    for (int i=0; i<Nq_xray; ++i){
        A_real_xray_current[i*blockDim.x + threadIndex] = 0.0;
        A_complex_xray_current[i*blockDim.x + threadIndex] = 0.0;
    }
    for (int i=0; i<Nq_neutron; ++i){
        A_real_neutron_current[i*blockDim.x + threadIndex] = 0.0;
        A_complex_neutron_current[i*blockDim.x + threadIndex] = 0.0;
    }
}

// Calculate form factor components for the current system
extern "C" __global__ void computeFFtotal( 
const real4* __restrict__ posq,
double* origin,
const float* __restrict__ box, 
const int* __restrict__ is_water, 
const float* __restrict__ xray_strength, 
const float* __restrict__ neutron_strength, 
const char* __restrict__ atom_names,
const float* __restrict__ xray_qs, 
const float* __restrict__ neutron_qs, 
const float* __restrict__ cutoff, 
const float* __restrict__ d_parts, 
double* A_real_xray_current, 
double* A_real_neutron_current, 
double* A_complex_xray_current, 
double* A_complex_neutron_current
) {
    int threadIndex = threadIdx.x;

    // Now the system itself. A_real_current, A_complex_current
    for (int index=blockIdx.x * blockDim.x + threadIndex; index<Natoms; index+=blockDim.x * gridDim.x) {

        // Wrap the coords of the atom
        float z = posq[index].z - origin[0];
	if (fabsf(z) > box[2]/2.0) {
            if (z<0.0) {
                z = z - box[2]*floorf((z - 0.5*(box[2]))/box[2]);
            }
            if (z>=0.0) {
                z = z - box[2]*floorf((z + 0.5*(box[2]))/box[2]);
            }
        }

	if (fabsf(z)<=cutoff[0]){
    	    // X-ray
	    for (int i=0; i<Nq_xray; ++i){
		atomicAdd(&A_real_xray_current[i*blockDim.x + threadIndex], xray_strength[index*Nq_xray + i] * cosf(xray_qs[i]*z));
		atomicAdd(&A_complex_xray_current[i*blockDim.x + threadIndex], xray_strength[index*Nq_xray + i] * sinf(xray_qs[i]*z));
    	    }
    	    // Neutron
    	    for (int i=0; i<Nq_neutron; ++i){
		atomicAdd(&A_real_neutron_current[i*blockDim.x + threadIndex], neutron_strength[index*Nq_neutron + i] * cosf(neutron_qs[i]*z));
    	    	atomicAdd(&A_complex_neutron_current[i*blockDim.x + threadIndex], neutron_strength[index*Nq_neutron + i] * sinf(neutron_qs[i]*z));
    	    }
	}
    }
}

// Run in a single block
extern "C" __global__ void computePostFFtotal( 
double* A_real_xray_current, 
double* A_real_neutron_current, 
double* A_complex_xray_current, 
double* A_complex_neutron_current, 
double* A_real_xray_out, 
double* A_real_neutron_out, 
double* A_complex_xray_out, 
double* A_complex_neutron_out, 
double* A_sqr_xray_out,
double* A_sqr_neutron_out
) {
    int threadIndex = threadIdx.x;

    // Reduce to have one value for each q and calculate forces

    // Reduce X-ray
    for (int i=0; i<Nq_xray; ++i){
        for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
            if (threadIndex < stride) {
                A_real_xray_current[threadIndex + i*blockDim.x] += A_real_xray_current[threadIndex + i*blockDim.x + stride];
                A_complex_xray_current[threadIndex + i*blockDim.x] += A_complex_xray_current[threadIndex + i*blockDim.x + stride];
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // We have the reduced values in A_real_xray_current[i*blockDim.x] for each i-th q value
    // Now put them into the first Nq_xray elements of A_real_xray_current for a future use
    for (int index=threadIndex; index<Nq_xray; index+=blockDim.x) {
	A_real_xray_out[index] = A_real_xray_current[index*blockDim.x];
	A_complex_xray_out[index] = A_complex_xray_current[index*blockDim.x];
	A_sqr_xray_out[index] = A_real_xray_current[index*blockDim.x]*A_real_xray_current[index*blockDim.x] + A_complex_xray_current[index*blockDim.x]*A_complex_xray_current[index*blockDim.x];
    }

    // Reduce neutron
    for (int i=0; i<Nq_neutron; ++i){
	for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
            if (threadIndex < stride) {
                A_real_neutron_current[threadIndex + i*blockDim.x] += A_real_neutron_current[threadIndex + i*blockDim.x + stride];
                A_complex_neutron_current[threadIndex + i*blockDim.x] += A_complex_neutron_current[threadIndex + i*blockDim.x + stride];
            }
            __syncthreads();
        }
    }
    __syncthreads();
    
    // We have the reduced values in A_real_neutron_current[i*blockDim.x] for each i-th q value
    // Now put them into the first Nq_neutrin elements of A_real_neutron_current for a future use
    for (int index=threadIndex; index<Nq_neutron; index+=blockDim.x) {
	A_real_neutron_out[index] = A_real_neutron_current[index*blockDim.x];
	A_complex_neutron_out[index] = A_complex_neutron_current[index*blockDim.x];
	A_sqr_neutron_out[index] = A_real_neutron_current[index*blockDim.x]*A_real_neutron_current[index*blockDim.x] + A_complex_neutron_current[index*blockDim.x]*A_complex_neutron_current[index*blockDim.x];
    }
}

// Here we precalculate B_real/B_sqr
extern "C" __global__ void computePreForce( 
const real4* __restrict__ posq,
const float* __restrict__ box, 
const float* __restrict__ xray_h_strength, 
const float* __restrict__ xray_o_strength, 
const float* __restrict__ xray_qs, 
const float* __restrict__ neutron_qs, 
const float* __restrict__ neutron_o_strength, 
const float* __restrict__ neutron_h_strength, 
const float* __restrict__ neutron_d_strength, 
const double* __restrict__ w_dens, 
const double* __restrict__ w_dens_sqr, 
const float* __restrict__ cutoff, 
const float* __restrict__ d_parts, 
double* B_real_xray_current, 
double* B_real_neutron_current,
double* B_sqr_xray_current, 
double* B_sqr_neutron_current 
) {
    int threadIndex = threadIdx.x;

    // Get B_real
    // Xray
    for (int index=threadIndex; index<Nq_xray; index+=blockDim.x) {
        const float wXrayStrength = 2.0 * xray_h_strength[index] * (1.0 + (-0.48)*exp(-xray_qs[index]*xray_qs[index]/(2*0.22*0.22))) +
                                          xray_o_strength[index] * (1.0 + 0.12*exp(-xray_qs[index]*xray_qs[index]/(2*0.22*0.22)));
        B_real_xray_current[index] = 2.0 * w_dens[0] * box[0] * box[1] * wXrayStrength * sinf(xray_qs[index]*cutoff[0]) / xray_qs[index];
	B_sqr_xray_current[index] = w_dens_sqr[0] * (2.0*box[0] * box[1] * wXrayStrength * sinf(xray_qs[index]*cutoff[0]) / xray_qs[index]) 
					  * (2.0*box[0] * box[1] * wXrayStrength * sinf(xray_qs[index]*cutoff[0]) / xray_qs[index]);
    }
    
    // Neutron
    for (int index=threadIndex; index<Nq_neutron; index+=blockDim.x) {
        const double w_neutr_scatt_streng = neutron_o_strength[0] + 2.0*(d_parts[index] * neutron_d_strength[0] + (1.0 - d_parts[index]) * neutron_h_strength[0]);
        B_real_neutron_current[index] = 2.0 * w_dens[0] * box[0] * box[1] * w_neutr_scatt_streng * sinf(neutron_qs[index]*cutoff[0]) / neutron_qs[index];
	B_sqr_neutron_current[index] = w_dens_sqr[0] * (2.0*box[0] * box[1] * w_neutr_scatt_streng * sinf(neutron_qs[index]*cutoff[0]) / neutron_qs[index]) 
					     * (2.0*box[0] * box[1] * w_neutr_scatt_streng * sinf(neutron_qs[index]*cutoff[0]) / neutron_qs[index]);
    }
}

// Calculate force
extern "C" __global__ void computeForce( 
const real4* __restrict__ posq,
const float* __restrict__ alpha, 
double* origin,
const double* __restrict__ k_xray, 
const double* __restrict__ k_neutron, 
const double* __restrict__ T, 
const float* __restrict__ box, 
const int* __restrict__ is_water, 
const float* __restrict__ xray_strength, 
const float* __restrict__ neutron_strength, 
const char* __restrict__ atom_names,
const float* __restrict__ xray_qs, 
const float* __restrict__ neutron_qs, 
const float* __restrict__ cutoff, 
const float* __restrict__ d_parts, 
double* B_real_xray_current, 
double* B_real_neutron_current,
double* A_real_xray_out, 
double* A_real_neutron_out, 
double* A_complex_xray_out, 
double* A_complex_neutron_out, 
double* F_total_xray, 
double* F_total_neutron, 
const int* __restrict__ particles, 
const double* __restrict__ F_exp_xray,
const double* __restrict__ F_exp_neutron,
const double* __restrict__ delta_F_exp_xray,
const double* __restrict__ delta_F_exp_neutron,
unsigned long long* __restrict__ forceBuffer
) {
    int threadIndex = threadIdx.x;

    const double const_xray = (-2.0) * alpha[0] * k_xray[0]* T[0] * (1.380658e-23 * 6.0221367e23 /1e3) * (1.0/Nq_xray); // k_Boltzmann (J*K^-1) * N_avagadro (mol^-1) * 10e-3 (J->kJ) = 0.0083144621 (kJ/(mol*K))
    const double const_neutron = (-2.0) * alpha[0] * k_neutron[0]* T[0] * (1.380658e-23 * 6.0221367e23 /1e3) * (1.0/Nq_neutron); // k_Boltzmann (J*K^-1) * N_avagadro (mol^-1) * 10e-3 (J->kJ) = 0.0083144621 (kJ/(mol*K))
    for (int index=blockIdx.x * blockDim.x + threadIndex; index<particles_size; index+=blockDim.x * gridDim.x) {
        // Wrap the coords of the atom
        float z = posq[particles[index]].z - origin[0];
        if (fabsf(z) > box[2]/2.0) {
            if (z<0.0) {
                z = z - box[2]*floorf((z - 0.5*(box[2]))/box[2]);
            }
            if (z>=0.0) {
                z = z - box[2]*floorf((z + 0.5*(box[2]))/box[2]);
            }
        }
	if (fabsf(z)<=cutoff[0]){ 
    	    double scatt_streng_for_force;
	    // Collect force for the atom number particles[index]
	    double force = 0.0;

    	    // X-ray
	    for (int i=0; i<Nq_xray; ++i){
        	scatt_streng_for_force = xray_strength[particles[index]*Nq_xray + i];
        	force += const_xray * ((F_total_xray[i] - F_exp_xray[i]) / (delta_F_exp_xray[i]*delta_F_exp_xray[i])) *
                            ((1.0/F_total_xray[i]) * scatt_streng_for_force * xray_qs[i] * 
			     (-sinf(xray_qs[i]*z)*(A_real_xray_out[i] - B_real_xray_current[i]) + cosf(xray_qs[i]*z)*A_complex_xray_out[i]));
	    }
    	    // Neutron
    	    for (int i=0; i<Nq_neutron; ++i){
        	scatt_streng_for_force = neutron_strength[particles[index]*Nq_neutron + i];
		force += const_neutron * ((F_total_neutron[i] - F_exp_neutron[(i)]) / (delta_F_exp_neutron[i]*delta_F_exp_neutron[i])) *
                            ((1.0/F_total_neutron[i]) * scatt_streng_for_force * neutron_qs[i] * 
			     (-sinf(neutron_qs[i]*z)*(A_real_neutron_out[i] - B_real_neutron_current[i]) + cosf(neutron_qs[i]*z)*A_complex_neutron_out[i]));
    	    }
	
	    // Add the result
	    atomicAdd(&forceBuffer[particles[index]+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long)(force*0x100000000)));
	}
    }
}

// Calculate energy
extern "C" __global__ void computeEnergy( 
const float* __restrict__ alpha, 
float* energy_buffer,
const double* __restrict__ k_xray, 
const double* __restrict__ k_neutron, 
const double* __restrict__ T, 
double* F_total_xray, 
double* F_total_neutron, 
const double* __restrict__ F_exp_xray,
const double* __restrict__ F_exp_neutron,
const double* __restrict__ delta_F_exp_xray,
const double* __restrict__ delta_F_exp_neutron,
real* __restrict__ energyBuffer
) {
    int threadIndex = threadIdx.x;
    
    // Compute energy

    // Zero out
    energy_buffer[threadIndex] = 0.0;

    for (int index=threadIndex; index<Nq_xray; index+=blockDim.x) {// k_B (J*K^-1) * N_avagadro (mol^-1) * 10e-3 (J->kJ) = 0.0083144621 (kJ/(mol*K))
        energy_buffer[threadIndex] += alpha[0] * k_xray[0]* T[0] * (1.380658e-23 * 6.0221367e23 /1e3) * (1.0/Nq_xray) * ((F_total_xray[index] - F_exp_xray[index])*(F_total_xray[index] - F_exp_xray[index]) / (delta_F_exp_xray[index]*delta_F_exp_xray[index]));
    }
    __syncthreads();

    for (int index=threadIndex; index<Nq_neutron; index+=blockDim.x) {// k_B (J*K^-1) * N_avagadro (mol^-1) * 10e-3 (J->kJ) = 0.0083144621 (kJ/(mol*K))
        energy_buffer[threadIndex] += alpha[0] * k_neutron[0]* T[0] * (1.380658e-23 * 6.0221367e23 /1e3) * (1.0/Nq_neutron) * ((F_total_neutron[index] - F_exp_neutron[index])*(F_total_neutron[index] - F_exp_neutron[index]) / (delta_F_exp_neutron[index]*delta_F_exp_neutron[index]));
    }
    __syncthreads();

    // Reduce energy
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIndex < stride) {
            energy_buffer[threadIndex] += energy_buffer[threadIndex + stride];
        }
        __syncthreads();
    }

    if (threadIndex == 0) {
        energyBuffer[0] += energy_buffer[0];
    }
}

// This kernel is used if we set on_gpu flag to true for updateParametersInContext(context, on_gpu)
// If on_gpu is set to false we do the form factor calculation on the host
extern "C" __global__ void computeGlobalFtotal( 
const double* __restrict__ A_real_xray_out,
const double* __restrict__ A_complex_xray_out,
const double* __restrict__ A_sqr_xray_out,
const double* __restrict__ A_real_neutron_out,
const double* __restrict__ A_complex_neutron_out,
const double* __restrict__ A_sqr_neutron_out,
const double* __restrict__ B_real_xray_global,
const double* __restrict__ B_sqr_xray_global,
const double* __restrict__ B_real_neutron_global,
const double* __restrict__ B_sqr_neutron_global,
double* F_total_xray,
double* F_total_neutron
) {
    int threadIndex = threadIdx.x;

    for (int index=blockIdx.x * blockDim.x + threadIndex; index<Nq_xray; index+=blockDim.x * gridDim.x) {
	double intens = A_real_xray_out[index]*A_real_xray_out[index] + A_complex_xray_out[index]*A_complex_xray_out[index]
                                      + B_real_xray_global[index]*B_real_xray_global[index] - 2.0*A_real_xray_out[index]*B_real_xray_global[index]
                                      + A_sqr_xray_out[index] - A_real_xray_out[index]*A_real_xray_out[index] - A_complex_xray_out[index]*A_complex_xray_out[index]
                                      - B_sqr_xray_global[index] + B_real_xray_global[index]*B_real_xray_global[index];
	F_total_xray[index] = copysignf(1.0, intens)*sqrt(fabs(intens));
    }

    for (int index=blockIdx.x * blockDim.x + threadIndex; index<Nq_neutron; index+=blockDim.x * gridDim.x) {
	double intens = A_real_neutron_out[index]*A_real_neutron_out[index] + A_complex_neutron_out[index]*A_complex_neutron_out[index]
                                      + B_real_neutron_global[index]*B_real_neutron_global[index] - 2.0*A_real_neutron_out[index]*B_real_neutron_global[index]
                                      + A_sqr_neutron_out[index] - A_real_neutron_out[index]*A_real_neutron_out[index] - A_complex_neutron_out[index]*A_complex_neutron_out[index]
                                      - B_sqr_neutron_global[index] + B_real_neutron_global[index]*B_real_neutron_global[index];
	F_total_neutron[index] = copysignf(1.0, intens)*sqrt(fabs(intens));
    }
    __syncthreads();
}
