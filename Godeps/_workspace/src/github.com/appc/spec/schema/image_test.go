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

package schema

import "testing"

func TestEmptyApp(t *testing.T) {
	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.6.1",
		    "name": "example.com/test"
		}
		`

	var im ImageManifest

	err := im.UnmarshalJSON([]byte(imj))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Marshal and Unmarshal to verify that no "app": {} is generated on
	// Marshal and converted to empty struct on Unmarshal
	buf, err := im.MarshalJSON()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = im.UnmarshalJSON(buf)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestImageManifestMerge(t *testing.T) {
	imj := `{"name": "example.com/test"}`
	im := &ImageManifest{}

	if im.UnmarshalJSON([]byte(imj)) == nil {
		t.Fatal("Manifest JSON without acKind and acVersion unmarshalled successfully")
	}

	im = BlankImageManifest()

	err := im.UnmarshalJSON([]byte(imj))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
