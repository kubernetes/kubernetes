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
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows"
	"k8s.io/mount-utils"
)

// Keep the PowerShell oracle in one removable file. The native regression
// tests do not depend on it, and production never calls these helpers.
type powerShellLinkInfo struct {
	LinkType string
	Targets  []string
}

func powerShellLinkQuery(t *testing.T) func(string) (powerShellLinkInfo, error) {
	t.Helper()
	powershell, err := exec.LookPath("powershell")
	if err != nil {
		t.Skip("PowerShell is unavailable; native regression tests still run")
	}
	return func(path string) (powerShellLinkInfo, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		// Query the same properties as the previous implementation. JSON/UTF-8
		// preserves arrays and Unicode instead of depending on console encoding.
		cmd := exec.CommandContext(ctx, powershell, "-NoProfile", "-NonInteractive", "-Command", "$ErrorActionPreference = 'Stop'; [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; $item = Get-Item -Force -LiteralPath $env:linkpath; if ($null -eq $item) { throw 'Get-Item returned no item' }; @{LinkType = [string]$item.LinkType; Targets = @($item.Target | Where-Object { $null -ne $_ })} | ConvertTo-Json -Compress")
		cmd.Env = append(os.Environ(), "linkpath="+path)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return powerShellLinkInfo{}, fmt.Errorf("Get-Item %q: %w: %s", path, err, output)
		}
		var info powerShellLinkInfo
		if err := json.Unmarshal(output, &info); err != nil {
			return info, fmt.Errorf("decode Get-Item %q: %w: %s", path, err, output)
		}
		return info, nil
	}
}

// This test-only reference retains the previous resolver's traversal, using
// actual Get-Item results rather than nativeLinkType or os.Readlink.
func powerShellEvalSymlink(path string, query func(string) (powerShellLinkInfo, error)) (string, error) {
	path = mount.NormalizeWindowsPath(path)
	if isDeviceOrUncPath(path) || isDriveLetterorEmptyPath(path) {
		return path, nil
	}
	upper, base := path, ""
	var info powerShellLinkInfo
	for i := 0; i < MaxPathLength; i++ {
		var err error
		info, err = query(upper)
		if err != nil {
			return "", err
		}
		if info.LinkType != "" {
			break
		}
		base = filepath.Join(filepath.Base(upper), base)
		upper = getUpperPath(upper)
		if isDriveLetterorEmptyPath(upper) {
			return path, nil
		}
	}
	// Hardlink Target is an alias list, not a redirect. Its query parity is
	// checked separately; reproducing the old recursion would cycle.
	if info.LinkType == "HardLink" {
		return "", fmt.Errorf("hardlink resolution is outside the legacy parity cases")
	}
	target := strings.TrimSpace(strings.Join(info.Targets, "\n"))
	if target == "" || isDeviceOrUncPath(target) {
		return path, nil
	}
	if !filepath.IsAbs(target) {
		target = filepath.Join(getUpperPath(upper), target)
	}
	resolved, err := powerShellEvalSymlink(target, query)
	if err != nil {
		return path, err
	}
	return filepath.Join(resolved, base), nil
}

func TestNativeLinkQueriesMatchPowerShell(t *testing.T) {
	query := powerShellLinkQuery(t)
	tests := []struct {
		name, linkTarget, suffix  string
		directory, junction       bool
		hardlink, hidden, missing bool
	}{
		{name: "file"},
		{name: "directory", directory: true},
		{name: "hidden file", hidden: true},
		{name: "hidden directory", directory: true, hidden: true},
		{name: "missing", missing: true},
		{name: "absolute file", linkTarget: "absolute"},
		{name: "relative file", linkTarget: "relative"},
		{name: "absolute directory", linkTarget: "absolute", directory: true},
		{name: "relative directory", linkTarget: "relative", directory: true},
		{name: "directory parent", linkTarget: "relative", directory: true, suffix: "file.cfg"},
		{name: "junction", linkTarget: "absolute", directory: true, junction: true},
		{name: "junction parent", linkTarget: "absolute", directory: true, junction: true, suffix: "file.cfg"},
		{name: "symlink chain", linkTarget: "chain"},
		{name: "dangling symlink", linkTarget: "missing-target"},
		{name: "hardlink aliases", hardlink: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			// LiteralPath must handle metacharacters, spaces and Unicode.
			dir := filepath.Join(root, "target [literal] ' $ ż")
			if err := os.Mkdir(dir, 0700); err != nil {
				t.Fatal(err)
			}
			file := filepath.Join(dir, "file.cfg")
			if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
				t.Fatal(err)
			}
			path := file
			if test.directory {
				path = dir
			}
			if test.hidden {
				p, err := windows.UTF16PtrFromString(path)
				if err != nil {
					t.Fatal(err)
				}
				if err := windows.SetFileAttributes(p, windows.FILE_ATTRIBUTE_HIDDEN); err != nil {
					t.Fatal(err)
				}
			}
			if test.missing {
				path = filepath.Join(root, "missing")
			}
			if test.linkTarget != "" {
				target := path
				switch test.linkTarget {
				case "relative":
					var err error
					target, err = filepath.Rel(root, target)
					if err != nil {
						t.Fatal(err)
					}
				case "chain":
					target = filepath.Join(root, "intermediate")
					nativeTestSymlink(t, file, target)
				case "missing-target":
					target = filepath.Join(root, "missing-target")
				}
				path = filepath.Join(root, "link [literal] ' $ ż")
				if test.junction {
					nativeTestJunction(t, target, path)
				} else {
					nativeTestSymlink(t, target, path)
				}
			}
			paths := []string{path}
			if test.hardlink {
				for _, name := range []string{"alias1.cfg", "alias2.cfg"} {
					alias := filepath.Join(dir, name)
					if err := os.Link(file, alias); err != nil {
						t.Fatal(err)
					}
					paths = append(paths, alias)
				}
			}
			for _, path := range paths {
				info, psErr := query(path)
				kind, nativeErr := nativeLinkType(path)
				if (psErr != nil) != (nativeErr != nil) {
					t.Fatalf("%q: PowerShell error=%v; native error=%v", path, psErr, nativeErr)
				}
				if psErr == nil {
					kinds := map[windowsLinkType]string{windowsNotLink: "", windowsSymbolicLink: "SymbolicLink", windowsJunction: "Junction", windowsHardLink: "HardLink"}
					if got, ok := kinds[kind]; !ok || got != info.LinkType {
						t.Fatalf("%q: native kind=%v; PowerShell LinkType=%q", path, kind, info.LinkType)
					}
					var targets []string
					switch kind {
					case windowsSymbolicLink, windowsJunction:
						target, err := os.Readlink(path)
						if err != nil {
							t.Fatal(err)
						}
						targets = []string{target}
					case windowsHardLink:
						var err error
						targets, err = hardLinkTargets(path)
						if err != nil {
							t.Fatal(err)
						}
					}
					// Alias enumeration order and path casing are not contractual.
					for i := range targets {
						targets[i] = strings.ToLower(targets[i])
					}
					for i := range info.Targets {
						info.Targets[i] = strings.ToLower(info.Targets[i])
					}
					slices.Sort(targets)
					slices.Sort(info.Targets)
					if !slices.Equal(targets, info.Targets) {
						t.Fatalf("%q: native targets=%q; PowerShell Target=%q", path, targets, info.Targets)
					}
				}
				if test.hardlink {
					continue
				}
				input := filepath.Join(path, test.suffix)
				want, psErr := powerShellEvalSymlink(input, query)
				got, nativeErr := evalSymlink(input)
				if (psErr != nil) != (nativeErr != nil) || (psErr == nil && !strings.EqualFold(got, want)) {
					t.Fatalf("resolve %q: native=%q, %v; PowerShell=%q, %v", input, got, nativeErr, want, psErr)
				}
			}
		})
	}
}

// Set these roots to the same writable test share, accessed through UNC and a
// mapped drive in the test process's logon session. Ordinary unit tests must
// not depend on SMB credentials, a server, or administrator-only share setup.
func TestNativeSMBMatchesPowerShell(t *testing.T) {
	uncRoot := os.Getenv("KUBE_SUBPATH_SMB_UNC_ROOT")
	driveRoot := os.Getenv("KUBE_SUBPATH_SMB_DRIVE_ROOT")
	if uncRoot == "" && driveRoot == "" {
		t.Skip("set KUBE_SUBPATH_SMB_UNC_ROOT and KUBE_SUBPATH_SMB_DRIVE_ROOT to run the SMB integration test")
	}
	if !strings.HasPrefix(uncRoot, `\\`) || len(filepath.VolumeName(driveRoot)) != 2 || !filepath.IsAbs(driveRoot) {
		t.Fatal("SMB roots must be a UNC share and an absolute mapped-drive path")
	}
	query := powerShellLinkQuery(t)
	volume, err := os.MkdirTemp(driveRoot, "kube-subpath-smb-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(volume); err != nil {
			t.Errorf("remove SMB fixture: %v", err)
		}
	})
	dir := filepath.Join(volume, "config [literal] ż")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "file.cfg")
	if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"directory", "file", "missing"} {
		t.Run(name, func(t *testing.T) {
			path := dir
			if name == "file" {
				path = file
			} else if name == "missing" {
				path = filepath.Join(dir, "missing")
			}
			relative, err := filepath.Rel(driveRoot, path)
			if err != nil {
				t.Fatal(err)
			}
			for _, input := range []string{path, filepath.Join(uncRoot, relative)} {
				info, psErr := query(input)
				kind, nativeErr := nativeLinkType(input)
				if name == "missing" {
					if psErr == nil || !os.IsNotExist(nativeErr) {
						t.Fatalf("%q: PowerShell error=%v; native error=%v", input, psErr, nativeErr)
					}
				} else if psErr != nil || nativeErr != nil || kind != windowsNotLink || info.LinkType != "" || len(info.Targets) != 0 {
					t.Fatalf("%q: PowerShell=%+v, %v; native kind=%v, %v", input, info, psErr, kind, nativeErr)
				}
			}
			// Direct UNC paths are guarded by evalSymlink; compare traversal on
			// the mapped drive, where the previous implementation also ran it.
			want, psErr := powerShellEvalSymlink(path, query)
			got, nativeErr := evalSymlink(path)
			if (psErr != nil) != (nativeErr != nil) || (psErr == nil && !strings.EqualFold(got, want)) {
				t.Fatalf("resolve %q: native=%q, %v; PowerShell=%q, %v", path, got, nativeErr, want, psErr)
			}
		})
	}
	t.Run("prepare and release", func(t *testing.T) {
		newPath, cleanup, err := (&subpath{}).PrepareSafeSubpath(Subpath{VolumePath: volume, Path: file})
		if cleanup == nil {
			t.Fatal("missing cleanup action")
		}
		defer func() {
			if cleanup != nil {
				cleanup()
			}
		}()
		if err != nil || newPath != file {
			t.Fatalf("prepare SMB file: path=%q error=%v", newPath, err)
		}
		if err := os.Rename(file, file+".renamed"); err == nil {
			t.Fatal("SMB file was renamed while its subpath handle was retained")
		}
		cleanup()
		// Do not call cleanup twice: handles may be reused after being closed.
		cleanup = nil
		if err := os.Rename(file, file+".renamed"); err != nil {
			t.Fatalf("SMB cleanup did not release the handle: %v", err)
		}
	})
	for _, directory := range []bool{false, true} {
		name := "UNC file target"
		if directory {
			name = "UNC directory target"
		}
		t.Run(name, func(t *testing.T) {
			target := filepath.Join(uncRoot, filepath.Base(volume), filepath.Base(dir))
			flags := uint32(windows.SYMBOLIC_LINK_FLAG_DIRECTORY)
			if !directory {
				target = filepath.Join(target, "file.cfg.renamed")
				flags = 0
			}
			link := filepath.Join(t.TempDir(), "unc-link")
			linkp, err := windows.UTF16PtrFromString(link)
			if err != nil {
				t.Fatal(err)
			}
			targetp, err := windows.UTF16PtrFromString(target)
			if err != nil {
				t.Fatal(err)
			}
			if err := windows.CreateSymbolicLink(linkp, targetp, flags); err != nil {
				t.Fatal(err)
			}
			info, err := query(link)
			// Windows PowerShell can expose the stripped NT namespace form
			// UNC\server\share, whereas Readlink returns \\server\share.
			if len(info.Targets) == 1 && strings.HasPrefix(info.Targets[0], `UNC\`) {
				info.Targets[0] = `\\` + strings.TrimPrefix(info.Targets[0], `UNC\`)
			}
			if err != nil || info.LinkType != "SymbolicLink" || len(info.Targets) != 1 || !strings.EqualFold(info.Targets[0], target) {
				t.Fatalf("UNC target property: PowerShell=%+v, %v", info, err)
			}
			kind, nativeErr := nativeLinkType(link)
			readTarget, readErr := os.Readlink(link)
			if nativeErr != nil || kind != windowsSymbolicLink || readErr != nil || !strings.EqualFold(readTarget, target) {
				t.Fatalf("native UNC link query: kind=%v, %v; target=%q, %v", kind, nativeErr, readTarget, readErr)
			}
			got, nativeErr := evalSymlink(link)
			want, psErr := powerShellEvalSymlink(link, query)
			if nativeErr != nil || psErr != nil || got != link || want != link {
				t.Fatalf("UNC target followed: native=%q, %v; PowerShell=%q, %v", got, nativeErr, want, psErr)
			}
			_, cleanup, err := (&subpath{}).PrepareSafeSubpath(Subpath{VolumePath: filepath.Dir(link), Path: link})
			if cleanup != nil {
				cleanup()
			}
			if err == nil {
				t.Fatal("PrepareSafeSubpath accepted a symlink to an SMB target")
			}
		})
	}
}
