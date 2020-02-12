/*
 * Core bits
 */
// This one selects platform specific stuff. probably should come first
#include "uTensor/core/uTensor_util.hpp"
#include "uTensor/core/modelBase.hpp"
#include "uTensor/core/operatorBase.hpp"

/*
 * Allocators
 */
#include "uTensor/allocators/arenaAllocator.hpp"

/*
 * Operators
 */
#include "uTensor/ops/Arithmetic.hpp"
#include "uTensor/ops/Convolution.hpp"
#include "uTensor/ops/Matrix.hpp"

/*
 * Tensors
 */

#include "uTensor/tensors/BufferTensor.hpp"
#include "uTensor/tensors/RamTensor.hpp"
#include "uTensor/tensors/RomTensor.hpp"
