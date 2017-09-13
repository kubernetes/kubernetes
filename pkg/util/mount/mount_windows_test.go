// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package mount

import (
	"fmt"
	"os/exec"
	"testing"
)

func TestGetAvailableDriveLetter(t *testing.T) {
	if _, err := getAvailableDriveLetter(); err != nil {
		t.Errorf("getAvailableDriveLetter test failed : %v", err)
	}
}

func TestNormalizeWindowsPath(t *testing.T) {
	path := `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4/volumes/kubernetes.io~azure-disk`
	normalizedPath := normalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk`
	normalizedPath = normalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/`
	normalizedPath = normalizeWindowsPath(path)
	if normalizedPath != `c:\` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}
}

func TestValidateDiskNumber(t *testing.T) {
	diskNum := "0"
	if err := ValidateDiskNumber(diskNum); err != nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}

	diskNum = "99"
	if err := ValidateDiskNumber(diskNum); err != nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}

	diskNum = "ab"
	if err := ValidateDiskNumber(diskNum); err == nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}

	diskNum = "100"
	if err := ValidateDiskNumber(diskNum); err == nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}
}

func makeLink(link, target string) error {
	if output, err := exec.Command("cmd", "/c", "mklink", "/D", link, target).CombinedOutput(); err != nil {
		return fmt.Errorf("mklink failed: %v, link(%q) target(%q) output: %q", err, link, target, string(output))
	}
	return nil
}

func removeLink(link string) error {
	if output, err := exec.Command("cmd", "/c", "rmdir", link).CombinedOutput(); err != nil {
		return fmt.Errorf("rmdir failed: %v, output: %q", err, string(output))
	}
	return nil
}

func setEquivalent(set1, set2 []string) bool {
	map1 := make(map[string]bool)
	map2 := make(map[string]bool)
	for _, s := range set1 {
		map1[s] = true
	}
	for _, s := range set2 {
		map2[s] = true
	}

	for s := range map1 {
		if !map2[s] {
			return false
		}
	}
	for s := range map2 {
		if !map1[s] {
			return false
		}
	}
	return true
}

// this func must run in admin mode, otherwise it will fail
func TestGetMountRefs(t *testing.T) {
	fm := &FakeMounter{MountPoints: []MountPoint{}}
	mountPath := `c:\secondmountpath`
	expectedRefs := []string{`c:\`, `c:\firstmountpath`, mountPath}

	// remove symbolic links first
	for i := 1; i < len(expectedRefs); i++ {
		removeLink(expectedRefs[i])
	}

	// create symbolic links
	for i := 1; i < len(expectedRefs); i++ {
		if err := makeLink(expectedRefs[i], expectedRefs[i-1]); err != nil {
			t.Errorf("makeLink failed: %v", err)
		}
	}

	if refs, err := GetMountRefs(fm, mountPath); err != nil || !setEquivalent(expectedRefs, refs) {
		t.Errorf("getMountRefs(%q) = %v, error: %v; expected %v", mountPath, refs, err, expectedRefs)
	}

	// remove symbolic links
	for i := 1; i < len(expectedRefs); i++ {
		if err := removeLink(expectedRefs[i]); err != nil {
			t.Errorf("removeLink failed: %v", err)
		}
	}
}
