//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//

#define CK_PTR *
#ifndef NULL_PTR
#define NULL_PTR 0
#endif
#define CK_DEFINE_FUNCTION(returnType, name) returnType name
#define CK_DECLARE_FUNCTION(returnType, name) returnType name
#define CK_DECLARE_FUNCTION_POINTER(returnType, name) returnType (* name)
#define CK_CALLBACK_FUNCTION(returnType, name) returnType (* name)

#include <unistd.h>
#ifdef PACKED_STRUCTURES
# pragma pack(push, 1)
# include "pkcs11.h"
# pragma pack(pop)
#else
# include "pkcs11.h"
#endif

// Copy of CK_INFO but with default alignment (not packed). Go hides unaligned
// struct fields so copying to an aligned struct is necessary to read CK_INFO
// from Go on Windows where packing is required.
typedef struct ckInfo {
	CK_VERSION cryptokiVersion;
	CK_UTF8CHAR manufacturerID[32];
	CK_FLAGS flags;
	CK_UTF8CHAR libraryDescription[32];
	CK_VERSION libraryVersion;
} ckInfo, *ckInfoPtr;
