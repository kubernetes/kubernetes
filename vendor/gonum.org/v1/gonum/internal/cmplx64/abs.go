// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx64

import math "gonum.org/v1/gonum/internal/math32"

// Abs returns the absolute value (also called the modulus) of x.
func Abs(x complex64) float32 { return math.Hypot(real(x), imag(x)) }
