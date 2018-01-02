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

package lastditch

import (
	"fmt"
	"reflect"
	"testing"
)

func TestInvalidImageManifest(t *testing.T) {
	tests := []struct {
		desc     string
		json     string
		expected ImageManifest
	}{
		{
			desc:     "Check an empty image manifest",
			json:     imgJ("", labsJ(), ""),
			expected: imgI("", labsI()),
		},
		{
			desc:     "Check an image manifest with an empty label",
			json:     imgJ("example.com/test!", labsJ(labJ("", "", "")), ""),
			expected: imgI("example.com/test!", labsI(labI("", ""))),
		},
		{
			desc:     "Check an image manifest with an invalid name",
			json:     imgJ("example.com/test!", labsJ(), ""),
			expected: imgI("example.com/test!", labsI()),
		},
		{
			desc:     "Check an image manifest with labels with invalid names",
			json:     imgJ("im", labsJ(labJ("!n1", "v1", ""), labJ("N2~", "v2", "")), ""),
			expected: imgI("im", labsI(labI("!n1", "v1"), labI("N2~", "v2"))),
		},
		{
			desc:     "Check an image manifest with duplicated labels",
			json:     imgJ("im", labsJ(labJ("n1", "v1", ""), labJ("n1", "v2", "")), ""),
			expected: imgI("im", labsI(labI("n1", "v1"), labI("n1", "v2"))),
		},
		{
			desc:     "Check an image manifest with some extra fields",
			json:     imgJ("im", labsJ(), extJ("stuff")),
			expected: imgI("im", labsI()),
		},
		{
			desc:     "Check an image manifest with a label containing some extra fields",
			json:     imgJ("im", labsJ(labJ("n1", "v1", extJ("clutter"))), extJ("stuff")),
			expected: imgI("im", labsI(labI("n1", "v1"))),
		},
	}
	for _, tt := range tests {
		got := ImageManifest{}
		if err := got.UnmarshalJSON([]byte(tt.json)); err != nil {
			t.Errorf("%s: unexpected error during unmarshalling image manifest: %v", tt.desc, err)
		}
		if !reflect.DeepEqual(tt.expected, got) {
			t.Errorf("%s: did not get expected image manifest, got:\n  %#v\nexpected:\n  %#v", tt.desc, got, tt.expected)
		}
	}
}

func TestBogusImageManifest(t *testing.T) {
	bogus := []string{`
		{
		    "acKind": "Bogus",
		    "acVersion": "0.8.9",
		}
		`, `
		<html>
		    <head>
		        <title>Certainly not a JSON</title>
		    </head>
		</html>`,
	}

	for _, str := range bogus {
		im := ImageManifest{}
		if im.UnmarshalJSON([]byte(str)) == nil {
			t.Errorf("bogus image manifest unmarshalled successfully: %s", str)
		}
	}
}

// imgJ returns an image manifest JSON with given name and labels
func imgJ(name, labels, extra string) string {
	return fmt.Sprintf(`
		{
		    %s
		    "acKind": "ImageManifest",
		    "acVersion": "0.8.9",
		    "name": "%s",
		    "labels": %s
		}`, extra, name, labels)
}

// imgI returns an image manifest instance with given name and labels
func imgI(name string, labels Labels) ImageManifest {
	return ImageManifest{
		ACVersion: "0.8.9",
		ACKind:    "ImageManifest",
		Name:      name,
		Labels:    labels,
	}
}
