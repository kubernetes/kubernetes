// Copyright 2015 CoreOS, Inc.
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

package ioutil

import (
	"bytes"
	"testing"
)

func TestLimitedBufferReaderRead(t *testing.T) {
	buf := bytes.NewBuffer(make([]byte, 10))
	ln := 1
	lr := NewLimitedBufferReader(buf, ln)
	n, err := lr.Read(make([]byte, 10))
	if err != nil {
		t.Fatalf("unexpected read error: %v", err)
	}
	if n != ln {
		t.Errorf("len(data read) = %d, want %d", n, ln)
	}
}
