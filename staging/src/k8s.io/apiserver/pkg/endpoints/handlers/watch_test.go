/*
Copyright 2021 The Kubernetes Authors.

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

package handlers

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCyclicBuffer(t *testing.T) {
	c := newCyclicBuffer(3)
	assert.Empty(t, c.list(), "expected result to be empty")

	e1 := entry{start: time.Unix(1, 0)}
	c.add(e1)
	assert.Equal(t, []entry{e1}, c.list(), "expected result to be [e1]")

	e2 := entry{start: time.Unix(2, 0)}
	e3 := entry{start: time.Unix(3, 0)}
	c.add(e2)
	c.add(e3)
	assert.Equal(t, []entry{e1, e2, e3}, c.list(), "expected result to be [e1, e2, e3]")

	// First overflow
	e4 := entry{start: time.Unix(4, 0)}
	c.add(e4)
	assert.Equal(t, []entry{e2, e3, e4}, c.list(), "expected result to be [e2, e3, e4]")

	e5 := entry{start: time.Unix(5, 0)}
	c.add(e5)
	e6 := entry{start: time.Unix(6, 0)}
	c.add(e6)
	assert.Equal(t, []entry{e4, e5, e6}, c.list(), "expected result to be [e4, e5, e6]")
}
