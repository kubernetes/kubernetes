// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build ppc64 ppc64le
// +build !gccgo

#include "textflag.h"

//
// System calls for ppc64, Linux
//

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT	·Syscall(SB),NOSPLIT,$0-56
	BR	syscall·Syscall(SB)

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	BR	syscall·Syscall6(SB)

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	BR	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	BR	syscall·RawSyscall6(SB)
