/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package v1

import (
	"bytes"
	"testing"
)

func TestWriteJsonString(t *testing.T) {
	var buf bytes.Buffer
	WriteJsonString(&buf, "foo")
	if string(buf.Bytes()) != `"foo"` {
		t.Fatalf("Expected: %v\nGot: %v", `"foo"`, string(buf.Bytes()))
	}

	buf.Reset()
	WriteJsonString(&buf, `f"oo`)
	if string(buf.Bytes()) != `"f\"oo"` {
		t.Fatalf("Expected: %v\nGot: %v", `"f\"oo"`, string(buf.Bytes()))
	}
	// TODO(pquerna): all them important tests.
}
