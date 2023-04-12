/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package files

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	prefix = "test-util-files"
)

type file struct {
	name string
	// mode distinguishes file type,
	// we only check for regular vs. directory in these tests,
	// specify regular as 0, directory as os.ModeDir
	mode os.FileMode
	data string // ignored if mode == os.ModeDir
}

func (f *file) write(fs utilfs.Filesystem, dir string) error {
	path := filepath.Join(dir, f.name)
	if f.mode.IsDir() {
		if err := fs.MkdirAll(path, defaultPerm); err != nil {
			return err
		}
	} else if f.mode.IsRegular() {
		// create parent directories, if necessary
		parents := filepath.Dir(path)
		if err := fs.MkdirAll(parents, defaultPerm); err != nil {
			return err
		}
		// create the file
		handle, err := fs.Create(path)
		if err != nil {
			return err
		}
		_, err = handle.Write([]byte(f.data))
		// The file should always be closed, not just in error cases.
		if cerr := handle.Close(); cerr != nil {
			return fmt.Errorf("error closing file: %v", cerr)
		}
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("mode not implemented for testing %s", f.mode.String())
	}
	return nil
}

func (f *file) expect(fs utilfs.Filesystem, dir string) error {
	path := filepath.Join(dir, f.name)
	if f.mode.IsDir() {
		info, err := fs.Stat(path)
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return fmt.Errorf("expected directory, got mode %s", info.Mode().String())
		}
	} else if f.mode.IsRegular() {
		info, err := fs.Stat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("expected regular file, got mode %s", info.Mode().String())
		}
		data, err := fs.ReadFile(path)
		if err != nil {
			return err
		}
		if f.data != string(data) {
			return fmt.Errorf("expected file data %q, got %q", f.data, string(data))
		}
	} else {
		return fmt.Errorf("mode not implemented for testing %s", f.mode.String())
	}
	return nil
}

// write files, perform some function, then attempt to read files back
// if err is non-empty, expects an error from the function performed in the test
// and skips reading back the expected files
type test struct {
	desc    string
	writes  []file
	expects []file
	fn      func(fs utilfs.Filesystem, dir string, c *test) []error
	err     string
}

func (c *test) write(t *testing.T, fs utilfs.Filesystem, dir string) {
	for _, f := range c.writes {
		if err := f.write(fs, dir); err != nil {
			t.Fatalf("error pre-writing file: %v", err)
		}
	}
}

// you can optionally skip calling t.Errorf by passing a nil t, and process the
// returned errors instead
func (c *test) expect(t *testing.T, fs utilfs.Filesystem, dir string) []error {
	errs := []error{}
	for _, f := range c.expects {
		if err := f.expect(fs, dir); err != nil {
			msg := fmt.Errorf("expect %#v, got error: %v", f, err)
			errs = append(errs, msg)
			if t != nil {
				t.Errorf("%s", msg)
			}
		}
	}
	return errs
}

// run a test case, with an arbitrary function to execute between write and expect
// if c.fn is nil, errors from c.expect are checked against c.err, instead of errors
// from fn being checked against c.err
func (c *test) run(t *testing.T, fs utilfs.Filesystem) {
	// isolate each test case in a new temporary directory
	dir, err := fs.TempDir("", prefix)
	if err != nil {
		t.Fatalf("error creating temporary directory for test: %v", err)
	}
	defer os.RemoveAll(dir)
	c.write(t, fs, dir)
	// if fn exists, check errors from fn, then check expected files
	if c.fn != nil {
		errs := c.fn(fs, dir, c)
		if len(errs) > 0 {
			for _, err := range errs {
				utiltest.ExpectError(t, err, c.err)
			}
			// skip checking expected files if we expected errors
			// (usually means we didn't create file)
			return
		}
		c.expect(t, fs, dir)
		return
	}
	// just check expected files, and compare errors from c.expect to c.err
	// (this lets us test the helper functions above)
	errs := c.expect(nil, fs, dir)
	for _, err := range errs {
		utiltest.ExpectError(t, err, c.err)
	}
}

// simple test of the above helper functions
func TestHelpers(t *testing.T) {
	// omitting the test.fn means test.err is compared to errors from test.expect
	cases := []test{
		{
			desc:    "regular file",
			writes:  []file{{name: "foo", data: "bar"}},
			expects: []file{{name: "foo", data: "bar"}},
		},
		{
			desc:    "directory",
			writes:  []file{{name: "foo", mode: os.ModeDir}},
			expects: []file{{name: "foo", mode: os.ModeDir}},
		},
		{
			desc:    "deep regular file",
			writes:  []file{{name: "foo/bar", data: "baz"}},
			expects: []file{{name: "foo/bar", data: "baz"}},
		},
		{
			desc:    "deep directory",
			writes:  []file{{name: "foo/bar", mode: os.ModeDir}},
			expects: []file{{name: "foo/bar", mode: os.ModeDir}},
		},
		{
			desc:    "missing file",
			expects: []file{{name: "foo", data: "bar"}},
			err:     missingFileError,
		},
		{
			desc:    "missing directory",
			expects: []file{{name: "foo/bar", mode: os.ModeDir}},
			err:     missingFolderError,
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}

func TestFileExists(t *testing.T) {
	fn := func(fs utilfs.Filesystem, dir string, c *test) []error {
		ok, err := FileExists(fs, filepath.Join(dir, "foo"))
		if err != nil {
			return []error{err}
		}
		if !ok {
			return []error{fmt.Errorf("does not exist (test)")}
		}
		return nil
	}
	cases := []test{
		{
			fn:     fn,
			desc:   "file exists",
			writes: []file{{name: "foo"}},
		},
		{
			fn:   fn,
			desc: "file does not exist",
			err:  "does not exist (test)",
		},
		{
			fn:     fn,
			desc:   "object has non-file mode",
			writes: []file{{name: "foo", mode: os.ModeDir}},
			err:    "expected regular file",
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}

func TestEnsureFile(t *testing.T) {
	fn := func(fs utilfs.Filesystem, dir string, c *test) []error {
		var errs []error
		for _, f := range c.expects {
			if err := EnsureFile(fs, filepath.Join(dir, f.name)); err != nil {
				errs = append(errs, err)
			}
		}
		return errs
	}
	cases := []test{
		{
			fn:      fn,
			desc:    "file exists",
			writes:  []file{{name: "foo"}},
			expects: []file{{name: "foo"}},
		},
		{
			fn:      fn,
			desc:    "file does not exist",
			expects: []file{{name: "bar"}},
		},
		{
			fn:      fn,
			desc:    "neither parent nor file exists",
			expects: []file{{name: "baz/quux"}},
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}

// Note: This transitively tests WriteTmpFile
func TestReplaceFile(t *testing.T) {
	fn := func(fs utilfs.Filesystem, dir string, c *test) []error {
		var errs []error
		for _, f := range c.expects {
			if err := ReplaceFile(fs, filepath.Join(dir, f.name), []byte(f.data)); err != nil {
				errs = append(errs, err)
			}
		}
		return errs
	}
	cases := []test{
		{
			fn:      fn,
			desc:    "file exists",
			writes:  []file{{name: "foo"}},
			expects: []file{{name: "foo", data: "bar"}},
		},
		{
			fn:      fn,
			desc:    "file does not exist",
			expects: []file{{name: "foo", data: "bar"}},
		},
		{
			fn: func(fs utilfs.Filesystem, dir string, c *test) []error {
				if err := ReplaceFile(fs, filepath.Join(dir, "foo/bar"), []byte("")); err != nil {
					return []error{err}
				}
				return nil
			},
			desc: "neither parent nor file exists",
			err:  missingFolderError,
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}

func TestDirExists(t *testing.T) {
	fn := func(fs utilfs.Filesystem, dir string, c *test) []error {
		ok, err := DirExists(fs, filepath.Join(dir, "foo"))
		if err != nil {
			return []error{err}
		}
		if !ok {
			return []error{fmt.Errorf("does not exist (test)")}
		}
		return nil
	}
	cases := []test{
		{
			fn:     fn,
			desc:   "dir exists",
			writes: []file{{name: "foo", mode: os.ModeDir}},
		},
		{
			fn:   fn,
			desc: "dir does not exist",
			err:  "does not exist (test)",
		},
		{
			fn:     fn,
			desc:   "object has non-dir mode",
			writes: []file{{name: "foo"}},
			err:    "expected dir",
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}

func TestEnsureDir(t *testing.T) {
	fn := func(fs utilfs.Filesystem, dir string, c *test) []error {
		var errs []error
		for _, f := range c.expects {
			if err := EnsureDir(fs, filepath.Join(dir, f.name)); err != nil {
				errs = append(errs, err)
			}
		}
		return errs
	}
	cases := []test{
		{
			fn:      fn,
			desc:    "dir exists",
			writes:  []file{{name: "foo", mode: os.ModeDir}},
			expects: []file{{name: "foo", mode: os.ModeDir}},
		},
		{
			fn:      fn,
			desc:    "dir does not exist",
			expects: []file{{name: "bar", mode: os.ModeDir}},
		},
		{
			fn:      fn,
			desc:    "neither parent nor dir exists",
			expects: []file{{name: "baz/quux", mode: os.ModeDir}},
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}

func TestWriteTempDir(t *testing.T) {
	// writing a tmp dir is covered by TestReplaceDir, but we additionally test filename validation here
	c := test{
		desc: "invalid file key",
		err:  "invalid file key",
		fn: func(fs utilfs.Filesystem, dir string, c *test) []error {
			if _, err := WriteTempDir(fs, filepath.Join(dir, "tmpdir"), map[string]string{"foo/bar": ""}); err != nil {
				return []error{err}
			}
			return nil
		},
	}
	c.run(t, &utilfs.DefaultFs{})
}

func TestReplaceDir(t *testing.T) {
	fn := func(fs utilfs.Filesystem, dir string, c *test) []error {
		errs := []error{}

		// compute filesets from expected files and call ReplaceDir for each
		// we don't nest dirs in test cases, order of ReplaceDir call is not guaranteed
		dirs := map[string]map[string]string{}

		// allocate dirs
		for _, f := range c.expects {
			if f.mode.IsDir() {
				path := filepath.Join(dir, f.name)
				if _, ok := dirs[path]; !ok {
					dirs[path] = map[string]string{}
				}
			} else if f.mode.IsRegular() {
				path := filepath.Join(dir, filepath.Dir(f.name))
				if _, ok := dirs[path]; !ok {
					// require an expectation for the parent directory if there is an expectation for the file
					errs = append(errs, fmt.Errorf("no prior parent directory in c.expects for file %s", f.name))
					continue
				}
				dirs[path][filepath.Base(f.name)] = f.data
			}
		}

		// short-circuit test case validation errors
		if len(errs) > 0 {
			return errs
		}

		// call ReplaceDir for each desired dir
		for path, files := range dirs {
			if err := ReplaceDir(fs, path, files); err != nil {
				errs = append(errs, err)
			}
		}
		return errs
	}
	cases := []test{
		{
			fn:      fn,
			desc:    "fn catches invalid test case",
			expects: []file{{name: "foo/bar"}},
			err:     "no prior parent directory",
		},
		{
			fn:      fn,
			desc:    "empty dir",
			expects: []file{{name: "foo", mode: os.ModeDir}},
		},
		{
			fn:   fn,
			desc: "dir with files",
			expects: []file{
				{name: "foo", mode: os.ModeDir},
				{name: "foo/bar", data: "baz"},
				{name: "foo/baz", data: "bar"},
			},
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			c.run(t, &utilfs.DefaultFs{})
		})
	}
}
