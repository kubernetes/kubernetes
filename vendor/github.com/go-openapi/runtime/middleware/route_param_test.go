// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package middleware

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRouteParams(t *testing.T) {
	coll1 := RouteParams([]RouteParam{
		{"blah", "foo"},
		{"abc", "bar"},
		{"ccc", "efg"},
	})

	v := coll1.Get("blah")
	assert.Equal(t, v, "foo")
	v2 := coll1.Get("abc")
	assert.Equal(t, v2, "bar")
	v3 := coll1.Get("ccc")
	assert.Equal(t, v3, "efg")
	v4 := coll1.Get("ydkdk")
	assert.Empty(t, v4)
}
