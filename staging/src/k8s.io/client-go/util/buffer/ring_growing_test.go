/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package buffer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGrowth(t *testing.T) {
	t.Parallel()
	x := 10
	g := NewRingGrowing(1)
	for i := 0; i < x; i++ {
		assert.Equal(t, i, g.readable)
		g.WriteOne(i)
	}
	read := 0
	for g.readable > 0 {
		v, ok := g.ReadOne()
		assert.True(t, ok)
		assert.Equal(t, read, v)
		read++
	}
	assert.Equalf(t, x, read, "expected to have read %d items: %d", x, read)
	assert.Zerof(t, g.readable, "expected readable to be zero: %d", g.readable)
	assert.Equalf(t, 16, g.n, "expected N to be 16: %d", g.n)
}

func TestEmpty(t *testing.T) {
	t.Parallel()
	g := NewRingGrowing(1)
	_, ok := g.ReadOne()
	assert.False(t, ok)
}
