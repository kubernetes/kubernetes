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
	"fmt"
	"net/url"
	"path/filepath"
	"testing"
	"time"

	dist "github.com/coreos/rkt/pkg/distribution"
)

func TestGuessAppcOrPath(t *testing.T) {
	tests := []struct {
		image        string
		expectedType imageStringType
	}{
		// guess obvious absolute path as a path
		{
			image:        "/usr/libexec/rkt/stage1.aci",
			expectedType: imageStringPath,
		},
		// guess stuff with colon as a name
		{
			image:        "example.com/stage1:1.2.3",
			expectedType: imageStringName,
		},
		// guess stuff with ./ as a path
		{
			image:        "some/relative/../path/with/dots/file",
			expectedType: imageStringPath,
		},
		// the same
		{
			image:        "./another/obviously/relative/path",
			expectedType: imageStringPath,
		},
		// guess stuff ending with .aci as a path
		{
			image:        "some/relative/path/with/aci/extension.aci",
			expectedType: imageStringPath,
		},
		// guess stuff without .aci, ./ and : as a name
		{
			image:        "example.com/stage1",
			expectedType: imageStringName,
		},
		// another try
		{
			image:        "example.com/stage1,version=1.2.3,foo=bar",
			expectedType: imageStringName,
		},
	}
	for _, tt := range tests {
		guessed := guessAppcOrPath(tt.image, []string{".aci"})
		if tt.expectedType != guessed {
			t.Errorf("expected %q to be guessed as %q, but got %q", tt.image, imageTypeToString(tt.expectedType), imageTypeToString(guessed))
		}
	}
}

func imageTypeToString(imType imageStringType) string {
	switch imType {
	case imageStringName:
		return "name"
	case imageStringPath:
		return "path"
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

func TestDistFromImageString(t *testing.T) {
	relPath1 := "some/relative/../path/with/dots/file.aci"
	absPath1, err := filepath.Abs(relPath1)
	if err != nil {

		t.Fatalf("unexpected error: %v", err)
	}
	relPath2 := "some/relative/../path/with/dots/file"
	absPath2, err := filepath.Abs(relPath2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		in         string
		distString string
		err        error
	}{
		// Appc
		{
			"example.com/app01",
			"cimd:appc:v=0:example.com/app01",
			nil,
		},
		{
			"example.com/app01:v1.0.0",
			"cimd:appc:v=0:example.com/app01?version=v1.0.0",
			nil,
		},
		{
			"example.com/app01:v1.0.0,label01=?&*/",
			"cimd:appc:v=0:example.com/app01?label01=%3F%26%2A%2F&version=v1.0.0",
			nil,
		},
		{
			"some-image-name",
			"cimd:appc:v=0:some-image-name",
			nil,
		},
		{
			"some-image-name:v1.0.0",
			"cimd:appc:v=0:some-image-name?version=v1.0.0",
			nil,
		},
		{
			"some-image-name:f6432b725a9a5f27eaecfa47a0cbab3c0ea00f22",
			"cimd:appc:v=0:some-image-name?version=f6432b725a9a5f27eaecfa47a0cbab3c0ea00f22",
			nil,
		},
		// ACIArchive
		{
			"file:///absolute/path/to/file.aci",
			"cimd:aci-archive:v=0:file%3A%2F%2F%2Fabsolute%2Fpath%2Fto%2Ffile.aci",
			nil,
		},
		{
			"/absolute/path/to/file.aci",
			"cimd:aci-archive:v=0:file%3A%2F%2F%2Fabsolute%2Fpath%2Fto%2Ffile.aci",
			nil,
		},
		{
			relPath1,
			"cimd:aci-archive:v=0:" + url.QueryEscape("file://"+absPath1),
			nil,
		},
		{
			"https://example.com/app.aci",
			"cimd:aci-archive:v=0:https%3A%2F%2Fexample.com%2Fapp.aci",
			nil,
		},
		// Path with no .aci extension
		{
			"/absolute/path/to/file",
			"",
			fmt.Errorf("invalid image string %q", "file:///absolute/path/to/file"),
		},
		{
			"/absolute/path/to/file.tar",
			"",
			fmt.Errorf("invalid image string %q", "file:///absolute/path/to/file.tar"),
		},
		{
			relPath2,
			"",
			fmt.Errorf("invalid image string %q", "file://"+absPath2),
		},
		// Docker
		{
			"docker:busybox",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			nil,
		},
		{
			"docker://busybox",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			nil,
		},
		{
			"docker:busybox:latest",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			nil,
		},
		{
			"docker://busybox:latest",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:latest",
			nil,
		},
		{
			"docker:busybox:1.0",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox:1.0",
			nil,
		},
		{
			"docker:busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6",
			"cimd:docker:v=0:registry-1.docker.io/library/busybox@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6",
			nil,
		},
		{
			"docker:myregistry.example.com:4000/busybox",
			"cimd:docker:v=0:myregistry.example.com:4000/busybox:latest",
			nil,
		},
		{
			"docker:myregistry.example.com:4000/busybox",
			"cimd:docker:v=0:myregistry.example.com:4000/busybox:latest",
			nil,
		},
		{
			"docker:myregistry.example.com:4000/busybox:1.0",
			"cimd:docker:v=0:myregistry.example.com:4000/busybox:1.0",
			nil,
		},
	}

	for _, tt := range tests {
		d, err := DistFromImageString(tt.in)
		if err != nil {
			if tt.err == nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.err.Error() != err.Error() {
				t.Fatalf("expected error %v, but got error %v", tt.err, err)
			}
			continue
		} else {
			if tt.err != nil {
				t.Fatalf("expected error %v, but got nil error", tt.err)
			}
		}
		td, err := dist.Parse(tt.distString)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !d.Equals(td) {
			t.Fatalf("expected identical distribution but got %q != %q", tt.distString, d.CIMD().String())
		}
	}

}
