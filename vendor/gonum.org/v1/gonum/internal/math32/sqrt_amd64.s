// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

// TODO(kortschak): use textflag.h after we drop Go 1.3 support
//#include "textflag.h"
// Don't insert stack check preamble.
#define NOSPLIT	4

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
	SQRTSS x+0(FP), X0
	MOVSS X0, ret+8(FP)
	RET
