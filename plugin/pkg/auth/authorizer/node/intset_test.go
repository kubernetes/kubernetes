/*
Copyright 2018 The Kubernetes Authors.

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

package node

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntSet(t *testing.T) {
	i := newIntSet()

	assert.False(t, i.has(1))
	assert.False(t, i.has(2))
	assert.False(t, i.has(3))
	assert.False(t, i.has(4))

	i.reset()
	i.increment(1) // to 1
	i.increment(2) // to 1

	assert.True(t, i.has(1))
	assert.True(t, i.has(2))
	assert.False(t, i.has(3))
	assert.False(t, i.has(4))

	i.decrement(1) // to 0
	i.increment(3) // to 1

	assert.False(t, i.has(1)) // removed
	assert.True(t, i.has(2))  // still present
	assert.True(t, i.has(3))  // added
	assert.False(t, i.has(4)) // not yet present

	i.decrement(2) // to 0
	i.increment(3) // to 2
	i.decrement(3) // to 1
	i.increment(4) // to 1

	assert.False(t, i.has(1))
	assert.False(t, i.has(2))
	assert.True(t, i.has(3))
	assert.True(t, i.has(4))

	i.reset()
	assert.False(t, i.has(1))
	assert.False(t, i.has(2))
	assert.False(t, i.has(3))
	assert.False(t, i.has(4))
}
