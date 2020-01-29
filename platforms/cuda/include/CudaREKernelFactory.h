#ifndef OPENMM_CUDAREKERNELFACTORY_H_
#define OPENMM_CUDAREKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the RE plugin.
 */

class CudaREKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAREKERNELFACTORY_H_*/
