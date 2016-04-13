// Copyright 2015 The rkt Authors
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

package image

import (
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/coreos/rkt/common/apps"
)

func TestGuessImageType(t *testing.T) {
	tests := []struct {
		image        string
		expectedType apps.AppImageType
	}{
		// guess obvious hash as a hash
		{
			image:        "sha512-a8d0943eb94eb9da4a6dddfa51e5e3de84375de77271d26c41ac1ce6f588b618",
			expectedType: apps.AppImageHash,
		},
		// guess obvious URL as a URL
		{
			image:        "http://example.com/image.aci",
			expectedType: apps.AppImageURL,
		},
		// guess obvious absolute path as a path
		{
			image:        "/usr/libexec/rkt/stage1.aci",
			expectedType: apps.AppImagePath,
		},
		// guess stuff with colon as a name
		{
			image:        "example.com/stage1:1.2.3",
			expectedType: apps.AppImageName,
		},
		// guess stuff with ./ as a path
		{
			image:        "some/relative/../path/with/dots/file",
			expectedType: apps.AppImagePath,
		},
		// the same
		{
			image:        "./another/obviously/relative/path",
			expectedType: apps.AppImagePath,
		},
		// guess stuff ending with .aci as a path
		{
			image:        "some/relative/path/with/aci/extension.aci",
			expectedType: apps.AppImagePath,
		},
		// guess stuff without .aci, ./ and : as a name
		{
			image:        "example.com/stage1",
			expectedType: apps.AppImageName,
		},
		// another try
		{
			image:        "example.com/stage1,version=1.2.3,foo=bar",
			expectedType: apps.AppImageName,
		},
	}
	for _, tt := range tests {
		guessed := guessImageType(tt.image)
		if tt.expectedType != guessed {
			t.Errorf("expected %q to be guessed as %q, but got %q", tt.image, imageTypeToString(tt.expectedType), imageTypeToString(guessed))
		}
	}
}

func imageTypeToString(imType apps.AppImageType) string {
	switch imType {
	case apps.AppImageGuess:
		return "to-be-guessed"
	case apps.AppImageHash:
		return "hash"
	case apps.AppImageURL:
		return "URL"
	case apps.AppImagePath:
		return "path"
	case apps.AppImageName:
		return "name"
	default:
		return "unknown"
	}
}

func TestSignatureURLFromImageURL(t *testing.T) {
	tests := []struct {
		i string
		s string
	}{
		{
			i: "http://example.com/image",
			s: "http://example.com/image.aci.asc",
		},
		{
			i: "http://example.com/image.aci",
			s: "http://example.com/image.aci.asc",
		},
		{
			i: "http://example.com/image.aci?foo=bar&baz=quux#blah",
			s: "http://example.com/image.aci.asc?foo=bar&baz=quux#blah",
		},
	}
	for _, tt := range tests {
		iu, err := url.Parse(tt.i)
		if err != nil {
			t.Errorf("failed to parse %q as an image URL: %v", tt.i, err)
			continue
		}
		su, err := url.Parse(tt.s)
		if err != nil {
			t.Errorf("failed to parse %q as a signature URL: %v", tt.s, err)
			continue
		}
		got := ascURLFromImgURL(iu)
		if su.String() != got.String() {
			t.Errorf("expected signature URL for image URL %q to be %q, but got %q", iu.String(), su.String(), got.String())
		}
	}
}

func TestSignaturePathFromImagePath(t *testing.T) {
	tests := []struct {
		i string
		s string
	}{
		{
			i: "/some/path/to/image",
			s: "/some/path/to/image.aci.asc",
		},
		{
			i: "/some/path/to/image.aci",
			s: "/some/path/to/image.aci.asc",
		},
	}
	for _, tt := range tests {
		got := ascPathFromImgPath(tt.i)
		if tt.s != got {
			t.Errorf("expected signature path for image path %q to be %q, but got %q", tt.i, tt.s, got)
		}
	}
}

func TestUseCached(t *testing.T) {
	tests := []struct {
		age int
		use bool
	}{
		{
			age: -11,
			use: false,
		},
		{
			age: -1,
			use: true,
		},
	}
	maxAge := 10
	for _, tt := range tests {
		age := time.Now().Add(time.Duration(tt.age) * time.Second)
		got := useCached(age, maxAge)
		if got != tt.use {
			t.Errorf("expected useCached(%v, %v) to return %v, but it returned %v", age, maxAge, tt.use, got)
		}
	}
}

func TestIsReallyNil(t *testing.T) {
	tests := []struct {
		name  string
		iface interface{}
		isNil bool
	}{
		// plain nil
		{
			name:  "plain nil",
			iface: nil,
			isNil: true,
		},
		// some pointer
		{
			name:  "some pointer",
			iface: &struct{}{},
			isNil: false,
		},
		// a nil interface
		{
			name:  "a nil interface",
			iface: func() io.Closer { return nil }(),
			isNil: true,
		},
		// a non-nil interface with nil value
		{
			name:  "a non-nil interface with nil value",
			iface: func() io.Closer { var v *os.File; return v }(),
			isNil: true,
		},
		// a non-nil interface with non-nil value
		{
			name:  "a non-nil interface with non-nil value",
			iface: func() io.Closer { return ioutil.NopCloser(nil) }(),
			isNil: false,
		},
	}
	for _, tt := range tests {
		t.Log(tt.name)
		got := isReallyNil(tt.iface)
		if tt.isNil != got {
			t.Errorf("expected isReallyNil(%#v) to return %v, but got %v", tt.iface, tt.isNil, got)
		}
	}
}
