// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
	SQRTSS x+0(FP), X0
	MOVSS X0, ret+8(FP)
	RET
