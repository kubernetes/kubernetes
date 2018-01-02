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

package acirenderer

import (
	"archive/tar"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

const tstprefix = "acirenderer-test"

type testTarEntry struct {
	header   *tar.Header
	contents string
}

func newTestTar(entries []*testTarEntry, dir string) (string, error) {
	t, err := ioutil.TempFile(dir, "tar")
	if err != nil {
		return "", err
	}
	defer t.Close()
	tw := tar.NewWriter(t)
	for _, entry := range entries {
		// Add default mode
		if entry.header.Mode == 0 {
			if entry.header.Typeflag == tar.TypeDir {
				entry.header.Mode = 0755
			} else {
				entry.header.Mode = 0644
			}
		}
		// Add calling user uid and gid or tests will fail
		entry.header.Uid = os.Getuid()
		entry.header.Gid = os.Getgid()
		if err := tw.WriteHeader(entry.header); err != nil {
			return "", err
		}
		if _, err := io.WriteString(tw, entry.contents); err != nil {
			return "", err
		}
	}
	if err := tw.Close(); err != nil {
		return "", err
	}
	return t.Name(), nil
}

type fileInfo struct {
	path     string
	typeflag byte
	size     int64
	mode     os.FileMode
}

func newTestACI(entries []*testTarEntry, dir string, ds *TestStore) (string, error) {
	testTarPath, err := newTestTar(entries, dir)
	if err != nil {
		return "", err
	}

	key, err := ds.WriteACI(testTarPath)
	if err != nil {
		return "", err
	}

	return key, nil
}

func createImageManifest(imj string) (*schema.ImageManifest, error) {
	var im schema.ImageManifest
	err := im.UnmarshalJSON([]byte(imj))
	if err != nil {
		return nil, err
	}
	return &im, nil
}

func addDependencies(imj string, deps ...types.Dependency) (string, error) {
	im, err := createImageManifest(imj)
	if err != nil {
		return "", err
	}

	for _, dep := range deps {
		im.Dependencies = append(im.Dependencies, dep)
	}
	imjb, err := im.MarshalJSON()
	return string(imjb), err
}

func genSimpleImage(imj string, pwl []string, level uint16, dir string, ds *TestStore) (*Image, error) {
	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
	}
	key, err := newTestACI(entries, dir, ds)
	if err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}
	im.PathWhitelist = pwl
	image1 := &Image{Im: im, Key: key, Level: level}
	return image1, nil

}

func TestGetUpperPWLM(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	l0pwl1 := []string{"/a/path/white/list/level0/1"}
	l1pwl1 := []string{"/a/path/white/list/level1/1"}

	l0pwl1m := pwlToMap(l0pwl1)

	// An image at level 0 with l0pwl1
	iml0pwl1, err := genSimpleImage(imj, l0pwl1, 0, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// An image at level 0 without pwl
	iml0nopwl, err := genSimpleImage(imj, []string{}, 0, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// An image at level 1 with l1pwl1
	iml1pwl1, err := genSimpleImage(imj, l1pwl1, 1, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// An image at level 1 without pwl
	iml1nopwl, err := genSimpleImage(imj, []string{}, 0, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// An image at level 2 without pwl
	iml2nopwl, err := genSimpleImage(imj, []string{}, 2, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var emptypwlm map[string]struct{}

	// A (pwl)
	// Searching for the upper pwlm of A should return nil
	A := *iml0pwl1
	images := Images{A}
	pwlm := getUpperPWLM(images, 0)
	if !reflect.DeepEqual(pwlm, emptypwlm) {
		t.Errorf("wrong PathWhitelist, got %#v, want: %#v", pwlm, emptypwlm)

	}

	// A (pwl) ---- B (pwl) --- C
	// Searching for the upper pwlm of C should return l0pwl1m
	A = *iml0pwl1
	B := *iml1pwl1
	C := *iml2nopwl
	images = Images{A, B, C}
	pwlm = getUpperPWLM(images, 2)
	if !reflect.DeepEqual(pwlm, l0pwl1m) {
		t.Errorf("wrong PathWhitelist, got %#v, want: %#v", pwlm, l0pwl1m)

	}

	// A ---- B --- D
	//    \-- C (pwl)
	// Searching for the upper pwlm of C should return nil
	A = *iml0nopwl
	B = *iml1nopwl
	C = *iml1pwl1
	D := *iml2nopwl
	images = Images{A, C, B, D}
	pwlm = getUpperPWLM(images, 3)
	if !reflect.DeepEqual(pwlm, emptypwlm) {
		t.Errorf("wrong PathWhitelist, got %#v, want: %#v", pwlm, emptypwlm)
	}
}

// Test an image with 1 dep. The parent provides a dir not provided by the image.
func TestDirFromParent(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// An empty dir
		{
			header: &tar.Header{
				Name:     "rootfs/a",
				Typeflag: tar.TypeDir,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a", typeflag: tar.TypeDir},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 0}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The image provides a dir not provided by the parent.
func TestNewDir(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// An empty dir
		{
			header: &tar.Header{
				Name:     "rootfs/a",
				Typeflag: tar.TypeDir,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 0}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a", typeflag: tar.TypeDir},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 1}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The image overrides dirs modes from the parent dep. Verifies the right permissions.
func TestDirOverride(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/a",
				Typeflag: tar.TypeDir,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// An empty dir
		{
			header: &tar.Header{
				Name:     "rootfs/a",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a", typeflag: tar.TypeDir, mode: 0700},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 0}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The parent provides a file not provided by the image.
func TestFileFromParent(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 0}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 5},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 1}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The image provides a file not provided by the parent.
func TestNewFile(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 0}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 10,
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 10},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 1}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The image overrides a file already provided by the parent dep.
func TestFileOverride(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 10,
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 10},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 0}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The upper image overrides a dir provided by a
// parent with a non-dir file.
func TestFileOvverideDir(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/a/b",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/a/b/c",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/b/c/file01",
				Size: 5,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/b",
				Size: 10,
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/b", typeflag: tar.TypeReg, size: 10},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 0}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The parent image has a pathWhiteList.
func TestPWLOnlyParent(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/b/link01.txt", "/c/", "/d/" ]
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 5,
			},
		},
		// This should not appear in rendered aci
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file03.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/b/link01.txt",
				Linkname: "file01.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
		// The file "rootfs/c/file01.txt" should not appear but a new file "rootfs/c/file02.txt" provided by the upper image should appear.
		// The directory should be left with its permissions
		{
			header: &tar.Header{
				Name:     "rootfs/c",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
		// The file "rootfs/d/file01.txt" should not appear but the directory should be left and also its permissions
		{
			header: &tar.Header{
				Name:     "rootfs/d",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/d/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
		// The file and the directory should not appear
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/e/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/b/file01.txt",
				Size: 10,
			},
		},
		// New file
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file02.txt",
				Size: 5,
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/a/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/link01.txt", typeflag: tar.TypeSymlink},
		&fileInfo{path: "rootfs/b/file01.txt", typeflag: tar.TypeReg, size: 10},
		&fileInfo{path: "rootfs/c", typeflag: tar.TypeDir, mode: 0700},
		&fileInfo{path: "rootfs/c/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/d", typeflag: tar.TypeDir, mode: 0700},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 0}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with 1 dep. The upper image has a pathWhiteList.
func TestPWLOnlyImage(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image in rendered aci
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 5,
			},
		},
		// It should not appear in rendered aci
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file03.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/b/link01.txt",
				Linkname: "file01.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
		// The file "rootfs/c/file01.txt" should not appear but a new file "rootfs/c/file02.txt" provided by the upper image should appear.
		{
			header: &tar.Header{
				Name:     "rootfs/c",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
		// The file "rootfs/d/file01.txt" should not appear but the directory should be left and also its permissions
		{
			header: &tar.Header{
				Name:     "rootfs/d",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/d/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
		// The file and the directory should not appear
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/e/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/b/link01.txt", "/b/file01.txt", "/c/file02.txt", "/d/" ]
		}
	`

	k1, _ := types.NewHash(key1)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 10,
			},
		},
		// New file
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file02.txt",
				Size: 5,
			},
		},
	}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 10},
		&fileInfo{path: "rootfs/a/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/link01.txt", typeflag: tar.TypeSymlink},
		&fileInfo{path: "rootfs/c/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/d", typeflag: tar.TypeDir, mode: 0700},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 0}

	images := Images{image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test02", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with a pathwhitelist and 2 deps (first with pathWhiteList and the second without pathWhiteList)
// A (pwl) ---- B (pwl)
//          \-- C
func Test2Deps1(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/a/file03.txt", "/a/file04.txt", "/b/link01.txt", "/b/file01.txt" ]
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
		// It should be overridden by the one provided by the next dep
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 5,
			},
		},
		// It should remain like this
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file03.txt",
				Size: 5,
			},
		},
		// It should not appear in rendered aci
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file04.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/b/link01.txt",
				Linkname: "file01.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 10,
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 10,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file01.txt",
				Size: 5,
			},
		},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test03",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/a/file03.txt", "/b/link01.txt", "/b/file01.txt", "/b/file02.txt", "/c/file01.txt" ]
		}
	`

	k1, _ := types.NewHash(key1)
	k2, _ := types.NewHash(key2)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
		types.Dependency{
			ImageName: "example.com/test02",
			ImageID:   k2},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// Overridden
		{
			contents: "hellohellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 15,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file02.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file01.txt",
				Size: 5,
			},
		},
	}

	key3, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image3 := Image{Im: im, Key: key3, Level: 0}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 15},
		&fileInfo{path: "rootfs/a/file02.txt", typeflag: tar.TypeReg, size: 10},
		&fileInfo{path: "rootfs/a/file03.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/link01.txt", typeflag: tar.TypeSymlink},
		&fileInfo{path: "rootfs/b/file01.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/c/file01.txt", typeflag: tar.TypeReg, size: 5},
	}

	images := Images{image3, image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test03", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test an image with a pathwhitelist and 2 deps (first without pathWhiteList and the second with pathWhiteList)
// A (pwl) ---- B
//          \-- C (pwl)
func Test2Deps2(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
		// It should be overridden by the one provided by the next dep
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 5,
			},
		},
		// It should remain like this
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file03.txt",
				Size: 5,
			},
		},
		// It should not appear in rendered aci
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file04.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/b/link01.txt",
				Linkname: "file01.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/b/file01.txt" ]
		}
	`

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 10,
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 10,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file01.txt",
				Size: 5,
			},
		},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 1}

	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test03",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/a/file03.txt", "/b/link01.txt", "/b/file01.txt", "/b/file02.txt", "/c/file01.txt" ]
		}
	`

	k1, _ := types.NewHash(key1)
	k2, _ := types.NewHash(key2)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
		types.Dependency{
			ImageName: "example.com/test02",
			ImageID:   k2},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// Overridden
		{
			contents: "hellohellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 15,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file02.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file01.txt",
				Size: 5,
			},
		},
	}

	key3, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image3 := Image{Im: im, Key: key3, Level: 0}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 15},
		&fileInfo{path: "rootfs/a/file02.txt", typeflag: tar.TypeReg, size: 10},
		&fileInfo{path: "rootfs/a/file03.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/link01.txt", typeflag: tar.TypeSymlink},
		&fileInfo{path: "rootfs/b/file01.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/c/file01.txt", typeflag: tar.TypeReg, size: 5},
	}

	images := Images{image3, image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test03", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test A (pwl) ---- B
//               \-- C -- D
func Test3Deps(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	ds := NewTestStore()

	// B
	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test01"
		}
	`

	entries := []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 5,
			},
		},
		// It should be overridden by the one provided by the next dep
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 5,
			},
		},
		// It should remain like this
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file03.txt",
				Size: 5,
			},
		},
		// It should not appear in rendered aci
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file04.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "rootfs/b/link01.txt",
				Linkname: "file01.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
	}

	key1, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err := createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image1 := Image{Im: im, Key: key1, Level: 1}

	// D
	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test03"
		}
	`

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file01.txt",
				Size: 5,
			},
		},
		// It should not appear in rendered aci
		{
			header: &tar.Header{
				Name:     "rootfs/d",
				Typeflag: tar.TypeDir,
				Mode:     0700,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/d/file01.txt",
				Size: 5,
				Mode: 0700,
			},
		},
	}

	key2, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image2 := Image{Im: im, Key: key2, Level: 2}

	// C
	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test02"
		}
	`
	k2, _ := types.NewHash(key2)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test03",
			ImageID:   k2},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// It should be overridden by the one provided by the upper image
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 10,
			},
		},
		{
			contents: "hellohello",
			header: &tar.Header{
				Name: "rootfs/a/file02.txt",
				Size: 10,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file01.txt",
				Size: 5,
			},
		},
	}

	key3, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image3 := Image{Im: im, Key: key3, Level: 1}

	// A
	imj = `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.1.1",
		    "name": "example.com/test04",
		    "pathWhitelist" : [ "/a/file01.txt", "/a/file02.txt", "/a/file03.txt", "/b/link01.txt", "/b/file01.txt", "/b/file02.txt", "/c/file01.txt" ]
		}
	`

	k1, _ := types.NewHash(key1)
	k3, _ := types.NewHash(key3)
	imj, err = addDependencies(imj,
		types.Dependency{
			ImageName: "example.com/test01",
			ImageID:   k1},
		types.Dependency{
			ImageName: "example.com/test02",
			ImageID:   k3},
	)

	entries = []*testTarEntry{
		{
			contents: imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(imj)),
			},
		},
		// Overridden
		{
			contents: "hellohellohello",
			header: &tar.Header{
				Name: "rootfs/a/file01.txt",
				Size: 15,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/b/file02.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "rootfs/c/file01.txt",
				Size: 5,
			},
		},
	}

	key4, err := newTestACI(entries, dir, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	im, err = createImageManifest(imj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	image4 := Image{Im: im, Key: key4, Level: 0}

	expectedFiles := []*fileInfo{
		&fileInfo{path: "manifest", typeflag: tar.TypeReg},
		&fileInfo{path: "rootfs/a/file01.txt", typeflag: tar.TypeReg, size: 15},
		&fileInfo{path: "rootfs/a/file02.txt", typeflag: tar.TypeReg, size: 10},
		&fileInfo{path: "rootfs/a/file03.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/link01.txt", typeflag: tar.TypeSymlink},
		&fileInfo{path: "rootfs/b/file01.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/b/file02.txt", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "rootfs/c/file01.txt", typeflag: tar.TypeReg, size: 5},
	}

	images := Images{image4, image3, image2, image1}
	err = checkRenderACIFromList(images, expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = checkRenderACI("example.com/test04", expectedFiles, ds)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Given an image app name and optional labels, get the best matching image
// available in the store, build its dependency list and render it inside dir
func RenderACI(name types.ACIdentifier, labels types.Labels, ap ACIRegistry) (map[string]*fileInfo, error) {
	renderedACI, err := GetRenderedACI(name, labels, ap)
	if err != nil {
		return nil, err
	}
	return renderImage(renderedACI, ap)
}

// Given an already populated dependency list, it will extract, under the provided
// directory, the rendered ACI
func RenderACIFromList(imgs Images, ap ACIProvider) (map[string]*fileInfo, error) {
	renderedACI, err := GetRenderedACIFromList(imgs, ap)
	if err != nil {
		return nil, err
	}
	return renderImage(renderedACI, ap)
}

// Given a RenderedACI, it will extract, under the provided directory, the
// needed files from the right source ACI.
// The manifest will be extracted from the upper ACI.
// No file overwriting is done as it should usually be called
// providing an empty directory.
func renderImage(renderedACI RenderedACI, ap ACIProvider) (map[string]*fileInfo, error) {
	files := make(map[string]*fileInfo)
	for _, ra := range renderedACI {
		rs, err := ap.ReadStream(ra.Key)
		if err != nil {
			return nil, err
		}
		defer rs.Close()
		tr := tar.NewReader(rs)
		for {
			hdr, err := tr.Next()
			if err == io.EOF {
				// end of tar archive
				break
			}
			if err != nil {
				return nil, fmt.Errorf("Error reading tar entry: %v", err)
			}
			typ := hdr.Typeflag
			cleanName := filepath.Clean(hdr.Name)
			if _, ok := ra.FileMap[cleanName]; ok {
				switch {
				case typ == tar.TypeReg || typ == tar.TypeRegA:
					files[cleanName] = &fileInfo{path: cleanName, typeflag: tar.TypeReg, size: hdr.Size, mode: hdr.FileInfo().Mode().Perm()}
				case typ == tar.TypeDir:
					files[cleanName] = &fileInfo{path: cleanName, typeflag: tar.TypeDir, mode: hdr.FileInfo().Mode().Perm()}
				case typ == tar.TypeSymlink:
					files[cleanName] = &fileInfo{path: cleanName, typeflag: tar.TypeSymlink, mode: hdr.FileInfo().Mode()}
				default:
					return nil, fmt.Errorf("wrong type flag: %v\n", typ)
				}
			}

		}
	}
	return files, nil
}

func checkRenderACI(app types.ACIdentifier, expectedFiles []*fileInfo, ds *TestStore) error {
	files, err := RenderACI(app, nil, ds)
	if err != nil {
		return err
	}
	err = checkExpectedFiles(files, FISliceToMap(expectedFiles))
	if err != nil {
		return err
	}

	return nil
}

func checkRenderACIFromList(images Images, expectedFiles []*fileInfo, ds *TestStore) error {
	files, err := RenderACIFromList(images, ds)
	if err != nil {
		return err
	}
	err = checkExpectedFiles(files, FISliceToMap(expectedFiles))
	if err != nil {
		return err
	}
	return nil
}

func checkExpectedFiles(files map[string]*fileInfo, expectedFiles map[string]*fileInfo) error {
	// Set defaults for not specified expected file mode
	for _, ef := range expectedFiles {
		if ef.mode == 0 {
			if ef.typeflag == tar.TypeDir {
				ef.mode = 0755
			} else {
				ef.mode = 0644
			}
		}
	}

	for _, ef := range expectedFiles {
		_, ok := files[ef.path]
		if !ok {
			return fmt.Errorf("Expected file \"%s\" not in files", ef.path)
		}

	}

	for _, file := range files {
		ef, ok := expectedFiles[file.path]
		if !ok {
			return fmt.Errorf("file \"%s\" not in expectedFiles", file.path)
		}
		if ef.typeflag != file.typeflag {
			return fmt.Errorf("file \"%s\": file type differs: found %d, wanted: %d", file.path, file.typeflag, ef.typeflag)
		}
		if ef.typeflag == tar.TypeReg && file.path != "manifest" {
			if ef.size != file.size {
				return fmt.Errorf("file \"%s\": size differs: found %d, wanted: %d", file.path, file.size, ef.size)
			}
		}
		// Check modes but ignore symlinks
		if ef.mode != file.mode && ef.typeflag != tar.TypeSymlink {
			return fmt.Errorf("file \"%s\": mode differs: found %#o, wanted: %#o", file.path, file.mode, ef.mode)
		}

	}
	return nil
}

func FISliceToMap(slice []*fileInfo) map[string]*fileInfo {
	fim := make(map[string]*fileInfo, len(slice))
	for _, fi := range slice {
		fim[fi.path] = fi
	}
	return fim
}

func TestEmptyRootFsDir(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)

	ds := NewTestStore()

	tests := []struct {
		name          types.ACIdentifier
		imj           string
		entries       []*testTarEntry
		expectedFiles []*fileInfo
	}{
		// Image with an empty rootfs dir.
		{
			"example.com/test_empty_rootfs",
			`
		            {
		                "acKind": "ImageManifest",
		                "acVersion": "0.8.9",
		                "name": "example.com/test_empty_rootfs"
		            }
                        `,
			[]*testTarEntry{
				// Empty rootfs directory.
				{
					header: &tar.Header{
						Name:     "rootfs",
						Typeflag: tar.TypeDir,
						Mode:     0700,
					},
				},
			},
			[]*fileInfo{
				{path: "manifest", typeflag: tar.TypeReg},
				{path: "rootfs", typeflag: tar.TypeDir, mode: 0700},
			},
		},

		// Image with an empty rootfs dir and pathWhitelist.
		{
			"example.com/test_empty_rootfs_pwl",
			`
		            {
		                "acKind": "ImageManifest",
		                "acVersion": "0.8.9",
		                "name": "example.com/test_empty_rootfs_pwl",
                                "pathWhitelist": ["foo"]
		            }
                        `,
			[]*testTarEntry{
				// Empty rootfs directory.
				{
					header: &tar.Header{
						Name:     "rootfs",
						Typeflag: tar.TypeDir,
						Mode:     0700,
					},
				},
			},
			[]*fileInfo{
				{path: "manifest", typeflag: tar.TypeReg},
				{path: "rootfs", typeflag: tar.TypeDir, mode: 0700},
			},
		},
	}

	for i, tt := range tests {
		tt.entries = append(tt.entries, &testTarEntry{
			contents: tt.imj,
			header: &tar.Header{
				Name: "manifest",
				Size: int64(len(tt.imj)),
			},
		})

		if _, err := newTestACI(tt.entries, dir, ds); err != nil {
			t.Fatalf("%d: unexpected error: %v", i, err)
		}

		if err := checkRenderACI(tt.name, tt.expectedFiles, ds); err != nil {
			t.Fatalf("%d: unexpected error: %v", i, err)
		}
	}
}
