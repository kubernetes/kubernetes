#include "version.h"

#ifndef BUILD_VERSION
#error Must specify -DBUILD_VERSION=... when building this file.
#endif

namespace cert_trans {


const char kBuildVersion[] = BUILD_VERSION;


}  // namespace cert_trans
