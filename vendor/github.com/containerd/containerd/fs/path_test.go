// +build !windows

package fs

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/containerd/containerd/fs/fstest"
	"github.com/pkg/errors"
)

type RootCheck struct {
	unresolved string
	expected   string
	scope      func(string) string
	cause      error
}

func TestRootPath(t *testing.T) {
	tests := []struct {
		name   string
		apply  fstest.Applier
		checks []RootCheck
		scope  func(string) (string, error)
	}{
		{
			name:   "SymlinkAbsolute",
			apply:  Symlink("/b", "fs/a/d"),
			checks: Check("fs/a/d/c/data", "b/c/data"),
		},
		{
			name:   "SymlinkRelativePath",
			apply:  Symlink("a", "fs/i"),
			checks: Check("fs/i", "fs/a"),
		},
		{
			name:   "SymlinkSkipSymlinksOutsideScope",
			apply:  Symlink("realdir", "linkdir"),
			checks: CheckWithScope("foo/bar", "foo/bar", "linkdir"),
		},
		{
			name:   "SymlinkLastLink",
			apply:  Symlink("/b", "fs/a/d"),
			checks: Check("fs/a/d", "b"),
		},
		{
			name:  "SymlinkRelativeLinkChangeScope",
			apply: Symlink("../b", "fs/a/e"),
			checks: CheckAll(
				Check("fs/a/e/c/data", "fs/b/c/data"),
				CheckWithScope("e", "b", "fs/a"), // Original return
			),
		},
		{
			name:  "SymlinkDeepRelativeLinkChangeScope",
			apply: Symlink("../../../../test", "fs/a/f"),
			checks: CheckAll(
				Check("fs/a/f", "test"),             // Original return
				CheckWithScope("a/f", "test", "fs"), // Original return
			),
		},
		{
			name: "SymlinkRelativeLinkChain",
			apply: fstest.Apply(
				Symlink("../g", "fs/b/h"),
				fstest.Symlink("../../../../../../../../../../../../root", "fs/g"),
			),
			checks: Check("fs/b/h", "root"),
		},
		{
			name:   "SymlinkBreakoutPath",
			apply:  Symlink("../i/a", "fs/j/k"),
			checks: CheckWithScope("k", "i/a", "fs/j"),
		},
		{
			name:   "SymlinkToRoot",
			apply:  Symlink("/", "foo"),
			checks: Check("foo", ""),
		},
		{
			name:   "SymlinkSlashDotdot",
			apply:  Symlink("/../../", "foo"),
			checks: Check("foo", ""),
		},
		{
			name:   "SymlinkDotdot",
			apply:  Symlink("../../", "foo"),
			checks: Check("foo", ""),
		},
		{
			name:   "SymlinkRelativePath2",
			apply:  Symlink("baz/target", "bar/foo"),
			checks: Check("bar/foo", "bar/baz/target"),
		},
		{
			name: "SymlinkScopeLink",
			apply: fstest.Apply(
				Symlink("root2", "root"),
				Symlink("../bar", "root2/foo"),
			),
			checks: CheckWithScope("foo", "bar", "root"),
		},
		{
			name: "SymlinkSelf",
			apply: fstest.Apply(
				Symlink("foo", "root/foo"),
			),
			checks: ErrorWithScope("foo", "root", errTooManyLinks),
		},
		{
			name: "SymlinkCircular",
			apply: fstest.Apply(
				Symlink("foo", "bar"),
				Symlink("bar", "foo"),
			),
			checks: ErrorWithScope("foo", "", errTooManyLinks), //TODO: Test for circular error
		},
		{
			name: "SymlinkCircularUnderRoot",
			apply: fstest.Apply(
				Symlink("baz", "root/bar"),
				Symlink("../bak", "root/baz"),
				Symlink("/bar", "root/bak"),
			),
			checks: ErrorWithScope("bar", "root", errTooManyLinks), // TODO: Test for circular error
		},
		{
			name: "SymlinkComplexChain",
			apply: fstest.Apply(
				fstest.CreateDir("root2", 0777),
				Symlink("root2", "root"),
				Symlink("r/s", "root/a"),
				Symlink("../root/t", "root/r"),
				Symlink("/../u", "root/root/t/s/b"),
				Symlink(".", "root/u/c"),
				Symlink("../v", "root/u/x/y"),
				Symlink("/../w", "root/u/v"),
			),
			checks: CheckWithScope("a/b/c/x/y/z", "w/z", "root"), // Original return
		},
		{
			name: "SymlinkBreakoutNonExistent",
			apply: fstest.Apply(
				Symlink("/", "root/slash"),
				Symlink("/idontexist/../slash", "root/sym"),
			),
			checks: CheckWithScope("sym/file", "file", "root"),
		},
		{
			name: "SymlinkNoLexicalCleaning",
			apply: fstest.Apply(
				Symlink("/foo/bar", "root/sym"),
				Symlink("/sym/../baz", "root/hello"),
			),
			checks: CheckWithScope("hello", "foo/baz", "root"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, makeRootPathTest(t, test.apply, test.checks))
	}

	// Add related tests which are unable to follow same pattern
	t.Run("SymlinkRootScope", testRootPathSymlinkRootScope)
	t.Run("SymlinkEmpty", testRootPathSymlinkEmpty)
}

func testRootPathSymlinkRootScope(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestFollowSymlinkRootScope")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	expected, err := filepath.EvalSymlinks(tmpdir)
	if err != nil {
		t.Fatal(err)
	}
	rewrite, err := RootPath("/", tmpdir)
	if err != nil {
		t.Fatal(err)
	}
	if rewrite != expected {
		t.Fatalf("expected %q got %q", expected, rewrite)
	}
}
func testRootPathSymlinkEmpty(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootPath(wd, "")
	if err != nil {
		t.Fatal(err)
	}
	if res != wd {
		t.Fatalf("expected %q got %q", wd, res)
	}
}

func makeRootPathTest(t *testing.T, apply fstest.Applier, checks []RootCheck) func(t *testing.T) {
	return func(t *testing.T) {
		applyDir, err := ioutil.TempDir("", "test-root-path-")
		if err != nil {
			t.Fatalf("Unable to make temp directory: %+v", err)
		}
		defer os.RemoveAll(applyDir)

		if apply != nil {
			if err := apply.Apply(applyDir); err != nil {
				t.Fatalf("Apply failed: %+v", err)
			}
		}

		for i, check := range checks {
			root := applyDir
			if check.scope != nil {
				root = check.scope(root)
			}

			actual, err := RootPath(root, check.unresolved)
			if check.cause != nil {
				if err == nil {
					t.Errorf("(Check %d) Expected error %q, %q evaluated as %q", i+1, check.cause.Error(), check.unresolved, actual)
				}
				if errors.Cause(err) != check.cause {
					t.Fatalf("(Check %d) Failed to evaluate root path: %+v", i+1, err)
				}
			} else {
				expected := filepath.Join(root, check.expected)
				if err != nil {
					t.Fatalf("(Check %d) Failed to evaluate root path: %+v", i+1, err)
				}
				if actual != expected {
					t.Errorf("(Check %d) Unexpected evaluated path %q, expected %q", i+1, actual, expected)
				}
			}
		}
	}
}

func Check(unresolved, expected string) []RootCheck {
	return []RootCheck{
		{
			unresolved: unresolved,
			expected:   expected,
		},
	}
}

func CheckWithScope(unresolved, expected, scope string) []RootCheck {
	return []RootCheck{
		{
			unresolved: unresolved,
			expected:   expected,
			scope: func(root string) string {
				return filepath.Join(root, scope)
			},
		},
	}
}

func ErrorWithScope(unresolved, scope string, cause error) []RootCheck {
	return []RootCheck{
		{
			unresolved: unresolved,
			cause:      cause,
			scope: func(root string) string {
				return filepath.Join(root, scope)
			},
		},
	}
}

func CheckAll(checks ...[]RootCheck) []RootCheck {
	all := make([]RootCheck, 0, len(checks))
	for _, c := range checks {
		all = append(all, c...)
	}
	return all
}

func Symlink(oldname, newname string) fstest.Applier {
	dir := filepath.Dir(newname)
	if dir != "" {
		return fstest.Apply(
			fstest.CreateDir(dir, 0755),
			fstest.Symlink(oldname, newname),
		)
	}
	return fstest.Symlink(oldname, newname)
}
