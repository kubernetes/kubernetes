//go:build windows

/*
Copyright The Kubernetes Authors.

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

package subpath

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/windows"
	"k8s.io/mount-utils"
)

func nativeTestSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
			t.Skipf("symbolic link creation requires SeCreateSymbolicLinkPrivilege: %v", err)
		}
		t.Fatal(err)
	}
}

func nativeTestJunction(t *testing.T, target, link string) {
	t.Helper()
	if output, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
		t.Fatalf("create junction: %v: %s", err, output)
	}
}

func TestNativeLinkType(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "file.cfg")
	if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "dir")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{file, dir} {
		kind, err := nativeLinkType(path)
		if err != nil || kind != windowsNotLink {
			t.Fatalf("ordinary path %q: kind=%v error=%v", path, kind, err)
		}
	}
	t.Run("file symlink", func(t *testing.T) {
		link := filepath.Join(root, "file-link")
		nativeTestSymlink(t, file, link)
		if kind, err := nativeLinkType(link); err != nil || kind != windowsSymbolicLink {
			t.Fatalf("kind=%v error=%v", kind, err)
		}
	})
	t.Run("directory symlink", func(t *testing.T) {
		link := filepath.Join(root, "dir-link")
		nativeTestSymlink(t, dir, link)
		if kind, err := nativeLinkType(link); err != nil || kind != windowsSymbolicLink {
			t.Fatalf("kind=%v error=%v", kind, err)
		}
	})
	t.Run("junction", func(t *testing.T) {
		link := filepath.Join(root, "junction")
		nativeTestJunction(t, dir, link)
		if kind, err := nativeLinkType(link); err != nil || kind != windowsJunction {
			t.Fatalf("kind=%v error=%v", kind, err)
		}
	})
	t.Run("hardlink", func(t *testing.T) {
		link := filepath.Join(root, "hard.cfg")
		if err := os.Link(file, link); err != nil {
			t.Fatal(err)
		}
		for _, path := range []string{file, link} {
			if kind, err := nativeLinkType(path); err != nil || kind != windowsHardLink {
				t.Fatalf("%q: kind=%v error=%v", path, kind, err)
			}
			if isLink, err := isLinkPath(path); err != nil || !isLink {
				t.Fatalf("%q: isLink=%v error=%v", path, isLink, err)
			}
		}
	})
	_, err := nativeLinkType(filepath.Join(root, "missing"))
	if !os.IsNotExist(err) {
		t.Fatalf("missing path should preserve not-exist error: %v", err)
	}
}

func TestEvalSymlinkNative(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "target")
	if err := os.MkdirAll(filepath.Join(dir, "leaf"), 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "file.cfg")
	if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name, target, suffix, expected string
		junction                       bool
	}{
		{name: "absolute file", target: file, expected: file},
		{name: "relative file", target: filepath.Join("target", "file.cfg"), expected: file},
		{name: "absolute directory parent", target: dir, suffix: "leaf", expected: filepath.Join(dir, "leaf")},
		{name: "relative directory parent", target: "target", suffix: "leaf", expected: filepath.Join(dir, "leaf")},
		{name: "junction parent", target: dir, suffix: "leaf", expected: filepath.Join(dir, "leaf"), junction: true},
	}
	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			link := filepath.Join(root, test.name)
			if test.junction {
				nativeTestJunction(t, test.target, link)
			} else {
				nativeTestSymlink(t, test.target, link)
			}
			// Resolution must work without a PowerShell executable in PATH.
			t.Setenv("PATH", filepath.Join(root, "no-executables"))
			actual, err := evalSymlink(filepath.Join(link, test.suffix))
			if err != nil || !strings.EqualFold(actual, test.expected) {
				t.Fatalf("case %d: got %q, %v; want %q", i, actual, err, test.expected)
			}
		})
	}
	actual, err := evalSymlink(file)
	if err != nil || actual != file {
		t.Fatalf("ordinary path: %q %v", actual, err)
	}
	if _, err := evalSymlink(filepath.Join(root, "missing")); !os.IsNotExist(err) {
		t.Fatalf("missing path: %v", err)
	}
}

func TestHardLinkTargetsMatchPowerShell(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "file.cfg")
	if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"alias1.cfg", "alias2.cfg"} {
		if err := os.Link(file, filepath.Join(root, name)); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"file.cfg", "alias1.cfg", "alias2.cfg"} {
		path := filepath.Join(root, name)
		targets, err := hardLinkTargets(path)
		if err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command("powershell", "/c", "$ErrorActionPreference = 'Stop'; (Get-Item -Force -LiteralPath $env:linkpath).Target")
		cmd.Env = append(os.Environ(), "linkpath="+path)
		output, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("PowerShell characterization: %v: %s", err, output)
		}
		expected := strings.FieldsFunc(strings.TrimSpace(string(output)), func(r rune) bool { return r == '\r' || r == '\n' })
		if len(targets) != len(expected) {
			t.Fatalf("%q: targets=%q PowerShell=%q", path, targets, expected)
		}
		for _, target := range targets {
			found := false
			for _, want := range expected {
				found = found || strings.EqualFold(target, want)
			}
			if !found {
				t.Fatalf("native target %q not in PowerShell targets %q", target, expected)
			}
		}
		resolved, err := evalSymlink(path)
		if err != nil || !strings.EqualFold(resolved, path) {
			t.Fatalf("hardlink names must not cause redirect recursion: %q %v", resolved, err)
		}
	}
}

func TestNativeSubpathContainmentAndLock(t *testing.T) {
	root := t.TempDir()
	volume := filepath.Join(root, "volume")
	if err := os.Mkdir(volume, 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(volume, "file.cfg")
	if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(volume, "alias.cfg")
	if err := os.Link(file, alias); err != nil {
		t.Fatal(err)
	}
	handles, err := lockAndCheckSubPath(volume, alias)
	if err != nil || len(handles) != 1 {
		unlockPath(handles)
		t.Fatalf("inside hardlink: handles=%d error=%v", len(handles), err)
	}
	if err := os.Rename(alias, alias+".renamed"); err == nil {
		unlockPath(handles)
		t.Fatal("hardlink name changed while locked")
	}
	unlockPath(handles)
	if err := os.Rename(alias, alias+".renamed"); err != nil {
		t.Fatalf("cleanup did not release handle: %v", err)
	}
	outside := filepath.Join(root, "outside")
	if err := os.Mkdir(outside, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(file, filepath.Join(outside, "alias.cfg")); err != nil {
		t.Fatal(err)
	}
	handles, err = lockAndCheckSubPath(volume, file)
	unlockPath(handles)
	if err == nil {
		t.Fatal("hardlink alias outside volume was accepted")
	}
	for _, junction := range []bool{false, true} {
		name := "escape-symlink"
		if junction {
			name = "escape-junction"
		}
		t.Run(name, func(t *testing.T) {
			link := filepath.Join(volume, name)
			if junction {
				nativeTestJunction(t, outside, link)
			} else {
				nativeTestSymlink(t, outside, link)
			}
			handles, err := lockAndCheckSubPath(volume, link)
			unlockPath(handles)
			if err == nil {
				t.Fatal("link outside volume was accepted")
			}
		})
	}
}

func TestEvalSymlinkSpecialPathsWithoutFollowing(t *testing.T) {
	for _, path := range []string{`\\127.0.0.1\not-a-share`, `//127.0.0.1/not-a-share`, `\\?\C:\not-a-path`, `\\.\PhysicalDrive999`, `UNC\127.0.0.1\not-a-share`, `Volume{00000000-0000-0000-0000-000000000000}\`, `C:`, `C:\`, ""} {
		actual, err := evalSymlink(path)
		if err != nil || actual != mount.NormalizeWindowsPath(path) {
			t.Fatalf("special path %q: got %q %v", path, actual, err)
		}
	}
	root := t.TempDir()
	link := filepath.Join(root, "unc-target")
	p, _ := windows.UTF16PtrFromString(link)
	target, _ := windows.UTF16PtrFromString(`\\127.0.0.1\not-a-share`)
	if err := windows.CreateSymbolicLink(p, target, windows.SYMBOLIC_LINK_FLAG_DIRECTORY); err != nil {
		if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
			t.Skipf("symbolic link creation privilege: %v", err)
		}
		t.Fatal(err)
	}
	actual, err := evalSymlink(link)
	if err != nil || actual != link {
		t.Fatalf("UNC target must not be followed: %q %v", actual, err)
	}
}

func TestNativeSubpathAccessDenied(t *testing.T) {
	volume := t.TempDir()
	path := filepath.Join(volume, "denied.cfg")
	if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	original, err := windows.GetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatal(err)
	}
	originalDACL, _, err := original.DACL()
	if err != nil {
		t.Fatal(err)
	}
	denied, err := windows.SecurityDescriptorFromString("D:(D;;GR;;;WD)(A;;GA;;;BA)(A;;GA;;;SY)")
	if err != nil {
		t.Fatal(err)
	}
	deniedDACL, _, err := denied.DACL()
	if err != nil {
		t.Fatal(err)
	}
	if err := windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, deniedDACL, nil); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.UNPROTECTED_DACL_SECURITY_INFORMATION, nil, nil, originalDACL, nil); err != nil {
			t.Errorf("restore fixture DACL: %v", err)
		}
	})
	handles, err := lockAndCheckSubPath(volume, path)
	unlockPath(handles)
	if err == nil {
		t.Skip("account bypasses the fixture's deny-read DACL")
	}
	if !strings.Contains(err.Error(), "Access is denied") {
		t.Fatalf("expected access-denied error, got: %v", err)
	}
}

func BenchmarkNativePrepareSafeSubpath(b *testing.B) {
	volume := b.TempDir()
	path := filepath.Join(volume, "a", "b", "c", "d")
	if err := os.MkdirAll(path, 0700); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handles, err := lockAndCheckSubPath(volume, path)
		unlockPath(handles)
		if err != nil {
			b.Fatal(err)
		}
	}
}
