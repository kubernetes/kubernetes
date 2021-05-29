// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·syscall_syscall(SB),NOSPLIT,$0
        JMP     syscall·_syscall(SB)

TEXT ·syscall_syscall6(SB),NOSPLIT,$0
        JMP     syscall·_syscall6(SB)
