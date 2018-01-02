package archive

import (
	"archive/tar"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/docker/docker/pkg/ioutils"
)

func TestApplyLayerInvalidFilenames(t *testing.T) {
	// TODO Windows: Figure out how to fix this test.
	if runtime.GOOS == "windows" {
		t.Skip("Passes but hits breakoutError: platform and architecture is not supported")
	}
	for i, headers := range [][]*tar.Header{
		{
			{
				Name:     "../victim/dotdot",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{
			{
				// Note the leading slash
				Name:     "/../victim/slash-dotdot",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("applylayer", "docker-TestApplyLayerInvalidFilenames", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestApplyLayerInvalidHardlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("TypeLink support on Windows")
	}
	for i, headers := range [][]*tar.Header{
		{ // try reading victim/hello (../)
			{
				Name:     "dotdot",
				Typeflag: tar.TypeLink,
				Linkname: "../victim/hello",
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (/../)
			{
				Name:     "slash-dotdot",
				Typeflag: tar.TypeLink,
				// Note the leading slash
				Linkname: "/../victim/hello",
				Mode:     0644,
			},
		},
		{ // try writing victim/file
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim/file",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (hardlink, symlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "symlink",
				Typeflag: tar.TypeSymlink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // Try reading victim/hello (hardlink, hardlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "hardlink",
				Typeflag: tar.TypeLink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // Try removing victim directory (hardlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("applylayer", "docker-TestApplyLayerInvalidHardlink", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestApplyLayerInvalidSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("TypeSymLink support on Windows")
	}
	for i, headers := range [][]*tar.Header{
		{ // try reading victim/hello (../)
			{
				Name:     "dotdot",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim/hello",
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (/../)
			{
				Name:     "slash-dotdot",
				Typeflag: tar.TypeSymlink,
				// Note the leading slash
				Linkname: "/../victim/hello",
				Mode:     0644,
			},
		},
		{ // try writing victim/file
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim/file",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (symlink, symlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "symlink",
				Typeflag: tar.TypeSymlink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (symlink, hardlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "hardlink",
				Typeflag: tar.TypeLink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // try removing victim directory (symlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("applylayer", "docker-TestApplyLayerInvalidSymlink", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestApplyLayerWhiteouts(t *testing.T) {
	// TODO Windows: Figure out why this test fails
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}

	wd, err := ioutil.TempDir("", "graphdriver-test-whiteouts")
	if err != nil {
		return
	}
	defer os.RemoveAll(wd)

	base := []string{
		".baz",
		"bar/",
		"bar/bax",
		"bar/bay/",
		"baz",
		"foo/",
		"foo/.abc",
		"foo/.bcd/",
		"foo/.bcd/a",
		"foo/cde/",
		"foo/cde/def",
		"foo/cde/efg",
		"foo/fgh",
		"foobar",
	}

	type tcase struct {
		change, expected []string
	}

	tcases := []tcase{
		{
			base,
			base,
		},
		{
			[]string{
				".bay",
				".wh.baz",
				"foo/",
				"foo/.bce",
				"foo/.wh..wh..opq",
				"foo/cde/",
				"foo/cde/efg",
			},
			[]string{
				".bay",
				".baz",
				"bar/",
				"bar/bax",
				"bar/bay/",
				"foo/",
				"foo/.bce",
				"foo/cde/",
				"foo/cde/efg",
				"foobar",
			},
		},
		{
			[]string{
				".bay",
				".wh..baz",
				".wh.foobar",
				"foo/",
				"foo/.abc",
				"foo/.wh.cde",
				"bar/",
			},
			[]string{
				".bay",
				"bar/",
				"bar/bax",
				"bar/bay/",
				"foo/",
				"foo/.abc",
				"foo/.bce",
			},
		},
		{
			[]string{
				".abc",
				".wh..wh..opq",
				"foobar",
			},
			[]string{
				".abc",
				"foobar",
			},
		},
	}

	for i, tc := range tcases {
		l, err := makeTestLayer(tc.change)
		if err != nil {
			t.Fatal(err)
		}

		_, err = UnpackLayer(wd, l, nil)
		if err != nil {
			t.Fatal(err)
		}
		err = l.Close()
		if err != nil {
			t.Fatal(err)
		}

		paths, err := readDirContents(wd)
		if err != nil {
			t.Fatal(err)
		}

		if !reflect.DeepEqual(tc.expected, paths) {
			t.Fatalf("invalid files for layer %d: expected %q, got %q", i, tc.expected, paths)
		}
	}

}

func makeTestLayer(paths []string) (rc io.ReadCloser, err error) {
	tmpDir, err := ioutil.TempDir("", "graphdriver-test-mklayer")
	if err != nil {
		return
	}
	defer func() {
		if err != nil {
			os.RemoveAll(tmpDir)
		}
	}()
	for _, p := range paths {
		if p[len(p)-1] == filepath.Separator {
			if err = os.MkdirAll(filepath.Join(tmpDir, p), 0700); err != nil {
				return
			}
		} else {
			if err = ioutil.WriteFile(filepath.Join(tmpDir, p), nil, 0600); err != nil {
				return
			}
		}
	}
	archive, err := Tar(tmpDir, Uncompressed)
	if err != nil {
		return
	}
	return ioutils.NewReadCloserWrapper(archive, func() error {
		err := archive.Close()
		os.RemoveAll(tmpDir)
		return err
	}), nil
}

func readDirContents(root string) ([]string, error) {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if info.IsDir() {
			rel = rel + "/"
		}
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return files, nil
}
