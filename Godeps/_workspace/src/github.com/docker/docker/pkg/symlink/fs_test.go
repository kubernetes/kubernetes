// Licensed under the Apache License, Version 2.0; See LICENSE.APACHE

package symlink

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

type dirOrLink struct {
	path   string
	target string
}

func makeFs(tmpdir string, fs []dirOrLink) error {
	for _, s := range fs {
		s.path = filepath.Join(tmpdir, s.path)
		if s.target == "" {
			os.MkdirAll(s.path, 0755)
			continue
		}
		if err := os.MkdirAll(filepath.Dir(s.path), 0755); err != nil {
			return err
		}
		if err := os.Symlink(s.target, s.path); err != nil && !os.IsExist(err) {
			return err
		}
	}
	return nil
}

func testSymlink(tmpdir, path, expected, scope string) error {
	rewrite, err := FollowSymlinkInScope(filepath.Join(tmpdir, path), filepath.Join(tmpdir, scope))
	if err != nil {
		return err
	}
	expected, err = filepath.Abs(filepath.Join(tmpdir, expected))
	if err != nil {
		return err
	}
	if expected != rewrite {
		return fmt.Errorf("Expected %q got %q", expected, rewrite)
	}
	return nil
}

func TestFollowSymlinkAbsolute(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkAbsolute")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	if err := makeFs(tmpdir, []dirOrLink{{path: "testdata/fs/a/d", target: "/b"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "testdata/fs/a/d/c/data", "testdata/b/c/data", "testdata"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkRelativePath(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkRelativePath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	if err := makeFs(tmpdir, []dirOrLink{{path: "testdata/fs/i", target: "a"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "testdata/fs/i", "testdata/fs/a", "testdata"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkSkipSymlinksOutsideScope(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkSkipSymlinksOutsideScope")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	if err := makeFs(tmpdir, []dirOrLink{
		{path: "linkdir", target: "realdir"},
		{path: "linkdir/foo/bar"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "linkdir/foo/bar", "linkdir/foo/bar", "linkdir/foo"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkInvalidScopePathPair(t *testing.T) {
	if _, err := FollowSymlinkInScope("toto", "testdata"); err == nil {
		t.Fatal("expected an error")
	}
}

func TestFollowSymlinkLastLink(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkLastLink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	if err := makeFs(tmpdir, []dirOrLink{{path: "testdata/fs/a/d", target: "/b"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "testdata/fs/a/d", "testdata/b", "testdata"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkRelativeLinkChangeScope(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkRelativeLinkChangeScope")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	if err := makeFs(tmpdir, []dirOrLink{{path: "testdata/fs/a/e", target: "../b"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "testdata/fs/a/e/c/data", "testdata/fs/b/c/data", "testdata"); err != nil {
		t.Fatal(err)
	}
	// avoid letting allowing symlink e lead us to ../b
	// normalize to the "testdata/fs/a"
	if err := testSymlink(tmpdir, "testdata/fs/a/e", "testdata/fs/a/b", "testdata/fs/a"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkDeepRelativeLinkChangeScope(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkDeepRelativeLinkChangeScope")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{{path: "testdata/fs/a/f", target: "../../../../test"}}); err != nil {
		t.Fatal(err)
	}
	// avoid letting symlink f lead us out of the "testdata" scope
	// we don't normalize because symlink f is in scope and there is no
	// information leak
	if err := testSymlink(tmpdir, "testdata/fs/a/f", "testdata/test", "testdata"); err != nil {
		t.Fatal(err)
	}
	// avoid letting symlink f lead us out of the "testdata/fs" scope
	// we don't normalize because symlink f is in scope and there is no
	// information leak
	if err := testSymlink(tmpdir, "testdata/fs/a/f", "testdata/fs/test", "testdata/fs"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkRelativeLinkChain(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkRelativeLinkChain")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	// avoid letting symlink g (pointed at by symlink h) take out of scope
	// TODO: we should probably normalize to scope here because ../[....]/root
	// is out of scope and we leak information
	if err := makeFs(tmpdir, []dirOrLink{
		{path: "testdata/fs/b/h", target: "../g"},
		{path: "testdata/fs/g", target: "../../../../../../../../../../../../root"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "testdata/fs/b/h", "testdata/root", "testdata"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkBreakoutPath(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkBreakoutPath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	// avoid letting symlink -> ../directory/file escape from scope
	// normalize to "testdata/fs/j"
	if err := makeFs(tmpdir, []dirOrLink{{path: "testdata/fs/j/k", target: "../i/a"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "testdata/fs/j/k", "testdata/fs/j/i/a", "testdata/fs/j"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkToRoot(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkToRoot")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	// make sure we don't allow escaping to /
	// normalize to dir
	if err := makeFs(tmpdir, []dirOrLink{{path: "foo", target: "/"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "foo", "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkSlashDotdot(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkSlashDotdot")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	tmpdir = filepath.Join(tmpdir, "dir", "subdir")

	// make sure we don't allow escaping to /
	// normalize to dir
	if err := makeFs(tmpdir, []dirOrLink{{path: "foo", target: "/../../"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "foo", "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkDotdot(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkDotdot")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	tmpdir = filepath.Join(tmpdir, "dir", "subdir")

	// make sure we stay in scope without leaking information
	// this also checks for escaping to /
	// normalize to dir
	if err := makeFs(tmpdir, []dirOrLink{{path: "foo", target: "../../"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "foo", "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkRelativePath2(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkRelativePath2")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{{path: "bar/foo", target: "baz/target"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "bar/foo", "bar/baz/target", ""); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkScopeLink(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkScopeLink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{
		{path: "root2"},
		{path: "root", target: "root2"},
		{path: "root2/foo", target: "../bar"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "root/foo", "root/bar", "root"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkRootScope(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkRootScope")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	expected, err := filepath.EvalSymlinks(tmpdir)
	if err != nil {
		t.Fatal(err)
	}
	rewrite, err := FollowSymlinkInScope(tmpdir, "/")
	if err != nil {
		t.Fatal(err)
	}
	if rewrite != expected {
		t.Fatalf("expected %q got %q", expected, rewrite)
	}
}

func TestFollowSymlinkEmpty(t *testing.T) {
	res, err := FollowSymlinkInScope("", "")
	if err != nil {
		t.Fatal(err)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if res != wd {
		t.Fatalf("expected %q got %q", wd, res)
	}
}

func TestFollowSymlinkCircular(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkCircular")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{{path: "root/foo", target: "foo"}}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "root/foo", "", "root"); err == nil {
		t.Fatal("expected an error for foo -> foo")
	}

	if err := makeFs(tmpdir, []dirOrLink{
		{path: "root/bar", target: "baz"},
		{path: "root/baz", target: "../bak"},
		{path: "root/bak", target: "/bar"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "root/foo", "", "root"); err == nil {
		t.Fatal("expected an error for bar -> baz -> bak -> bar")
	}
}

func TestFollowSymlinkComplexChainWithTargetPathsContainingLinks(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkComplexChainWithTargetPathsContainingLinks")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{
		{path: "root2"},
		{path: "root", target: "root2"},
		{path: "root/a", target: "r/s"},
		{path: "root/r", target: "../root/t"},
		{path: "root/root/t/s/b", target: "/../u"},
		{path: "root/u/c", target: "."},
		{path: "root/u/x/y", target: "../v"},
		{path: "root/u/v", target: "/../w"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "root/a/b/c/x/y/z", "root/w/z", "root"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkBreakoutNonExistent(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkBreakoutNonExistent")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{
		{path: "root/slash", target: "/"},
		{path: "root/sym", target: "/idontexist/../slash"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "root/sym/file", "root/file", "root"); err != nil {
		t.Fatal(err)
	}
}

func TestFollowSymlinkNoLexicalCleaning(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkNoLexicalCleaning")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	if err := makeFs(tmpdir, []dirOrLink{
		{path: "root/sym", target: "/foo/bar"},
		{path: "root/hello", target: "/sym/../baz"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := testSymlink(tmpdir, "root/hello", "root/foo/baz", "root"); err != nil {
		t.Fatal(err)
	}
}
