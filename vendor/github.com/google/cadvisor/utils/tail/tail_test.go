// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tail

import (
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReadNewTail(t *testing.T) {
	// Read should return (0, io.EOF) before first
	// attemptOpen.
	tail, err := newTail("test/nonexist/file")
	assert.NoError(t, err)
	buf := make([]byte, 0, 100)
	n, err := tail.Read(buf)
	assert.Equal(t, n, 0)
	assert.Equal(t, len(buf), 0)
	assert.EqualError(t, err, io.EOF.Error())
}
