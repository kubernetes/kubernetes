// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx64

import math "gonum.org/v1/gonum/internal/math32"

// IsInf returns true if either real(x) or imag(x) is an infinity.
func IsInf(x complex64) bool {
	if math.IsInf(real(x), 0) || math.IsInf(imag(x), 0) {
		return true
	}
	return false
}

// Inf returns a complex infinity, complex(+Inf, +Inf).
func Inf() complex64 {
	inf := math.Inf(1)
	return complex(inf, inf)
}
