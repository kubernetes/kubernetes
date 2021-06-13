// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
	FMOVS	x+0(FP), F0
	FSQRTS	F0, F0
	FMOVS	F0, ret+8(FP)
	RET
