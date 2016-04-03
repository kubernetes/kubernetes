// Copyright 2015 The appc Authors
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

package types

import "testing"

func TestEmptyHash(t *testing.T) {
	dj := `{"imageName": "example.com/reduce-worker-base"}`

	var d Dependency

	err := d.UnmarshalJSON([]byte(dj))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Marshal to verify that marshalling works without validation errors
	buf, err := d.MarshalJSON()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Unmarshal to verify that the generated json will not create wrong empty hash
	err = d.UnmarshalJSON(buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
