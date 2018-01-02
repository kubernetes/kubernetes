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

package aci

import (
	"archive/tar"
	"compress/gzip"
	"io/ioutil"
	"os"
	"testing"
)

func newTestACI(usedotslash bool) (*os.File, error) {
	tf, err := ioutil.TempFile("", "")
	if err != nil {
		return nil, err
	}

	manifestBody := `{"acKind":"ImageManifest","acVersion":"0.8.9","name":"example.com/app"}`

	gw := gzip.NewWriter(tf)
	tw := tar.NewWriter(gw)

	manifestPath := "manifest"
	if usedotslash {
		manifestPath = "./" + manifestPath
	}
	hdr := &tar.Header{
		Name: manifestPath,
		Size: int64(len(manifestBody)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return nil, err
	}
	if _, err := tw.Write([]byte(manifestBody)); err != nil {
		return nil, err
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gw.Close(); err != nil {
		return nil, err
	}
	return tf, nil
}

func newEmptyTestACI() (*os.File, error) {
	tf, err := ioutil.TempFile("", "")
	if err != nil {
		return nil, err
	}
	gw := gzip.NewWriter(tf)
	tw := tar.NewWriter(gw)
	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gw.Close(); err != nil {
		return nil, err
	}
	return tf, nil
}

func TestManifestFromImage(t *testing.T) {
	for _, usedotslash := range []bool{false, true} {
		img, err := newTestACI(usedotslash)
		if err != nil {
			t.Fatalf("newTestACI: unexpected error: %v", err)
		}
		defer img.Close()
		defer os.Remove(img.Name())

		im, err := ManifestFromImage(img)
		if err != nil {
			t.Fatalf("ManifestFromImage: unexpected error: %v", err)
		}
		if im.Name.String() != "example.com/app" {
			t.Errorf("expected %s, got %s", "example.com/app", im.Name.String())
		}

		emptyImg, err := newEmptyTestACI()
		if err != nil {
			t.Fatalf("newEmptyTestACI: unexpected error: %v", err)
		}
		defer emptyImg.Close()
		defer os.Remove(emptyImg.Name())

		im, err = ManifestFromImage(emptyImg)
		if err == nil {
			t.Fatalf("ManifestFromImage: expected error")
		}
	}
}

func TestNewCompressedTarReader(t *testing.T) {
	img, err := newTestACI(false)
	if err != nil {
		t.Fatalf("newTestACI: unexpected error: %v", err)
	}
	defer img.Close()
	defer os.Remove(img.Name())

	cr, err := NewCompressedTarReader(img)
	if err != nil {
		t.Fatalf("NewCompressedTarReader: unexpected error: %v", err)
	}

	ftype, err := DetectFileType(cr)
	if err != nil {
		t.Fatalf("DetectFileType: unexpected error: %v", err)
	}

	if ftype != TypeText {
		t.Errorf("expected %v, got %v", TypeText, ftype)
	}
}

func TestNewCompressedReader(t *testing.T) {
	img, err := newTestACI(false)
	if err != nil {
		t.Fatalf("newTestACI: unexpected error: %v", err)
	}
	defer img.Close()
	defer os.Remove(img.Name())

	cr, err := NewCompressedReader(img)
	if err != nil {
		t.Fatalf("NewCompressedReader: unexpected error: %v", err)
	}

	ftype, err := DetectFileType(cr)
	if err != nil {
		t.Fatalf("DetectFileType: unexpected error: %v", err)
	}

	if ftype != TypeTar {
		t.Errorf("expected %v, got %v", TypeTar, ftype)
	}
}
