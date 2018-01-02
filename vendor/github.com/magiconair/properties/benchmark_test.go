// Copyright 2013-2014 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"fmt"
	"testing"
)

// Benchmarks the decoder by creating a property file with 1000 key/value pairs.
func BenchmarkLoad(b *testing.B) {
	input := ""
	for i := 0; i < 1000; i++ {
		input += fmt.Sprintf("key%d=value%d\n", i, i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Load([]byte(input), ISO_8859_1)
	}
}
