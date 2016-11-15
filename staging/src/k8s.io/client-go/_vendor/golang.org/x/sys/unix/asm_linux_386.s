// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

#include "textflag.h"

//
// System calls for 386, Linux
//

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT	·Syscall(SB),NOSPLIT,$0-28
	JMP	syscall·Syscall(SB)

TEXT	·Syscall6(SB),NOSPLIT,$0-40
	JMP	syscall·Syscall6(SB)

TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	JMP	syscall·RawSyscall(SB)

TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	JMP	syscall·RawSyscall6(SB)

TEXT ·socketcall(SB),NOSPLIT,$0-36
	JMP	syscall·socketcall(SB)

TEXT ·rawsocketcall(SB),NOSPLIT,$0-36
	JMP	syscall·rawsocketcall(SB)

TEXT ·seek(SB),NOSPLIT,$0-28
	JMP	syscall·seek(SB)
