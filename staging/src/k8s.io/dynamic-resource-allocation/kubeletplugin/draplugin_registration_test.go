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

package kubeletplugin

import (
	"path"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/types"
)

func TestRollingUpdateRegistrarSocketFile_lengthBound(t *testing.T) {
	driver := "gpu.dra-example-driver.sigs.k8s.io"
	podUID := types.UID("ad4af911-5bcd-47c5-a554-f35cea869472")
	name := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, driver, podUID)
	full := path.Join(KubeletRegistryDir, name)
	if len(full) > unixPathMax {
		t.Fatalf("registration socket path len %d exceeds %d: %q", len(full), unixPathMax, full)
	}
}

func TestRollingUpdateRegistrarSocketFile_prefersFullUID(t *testing.T) {
	driver := "example.com/driver"
	podUID := types.UID("11111111-2222-3333-4444-555555555555")
	want := driver + "-" + string(podUID) + "-reg.sock"
	got := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, driver, podUID)
	if got != want {
		t.Fatalf("expected full UID basename %q, got %q", want, got)
	}
}

func TestRollingUpdateRegistrarSocketFile_usesHashedUIDForLongDriverName(t *testing.T) {
	driver := "gpu.dra-example-driver.sigs.k8s.io"
	podUID := types.UID("ad4af911-5bcd-47c5-a554-f35cea869472")
	got := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, driver, podUID)

	fullUID := driver + "-" + string(podUID) + "-reg.sock"
	if got == fullUID {
		t.Fatalf("expected hashed UID basename for long driver name, got full UID form %q", got)
	}
	if !strings.HasPrefix(got, driver+"-") || !strings.HasSuffix(got, "-reg.sock") {
		t.Fatalf("expected driver name prefix and -reg.sock suffix, got %q", got)
	}
	if strings.Contains(got, string(podUID)) {
		t.Fatalf("expected hashed UID, not full UID in %q", got)
	}
}

func TestRollingUpdateRegistrarSocketFile_usesShortestFormForVeryLongDriverName(t *testing.T) {
	driver := strings.Repeat("a", 100)
	podUID := types.UID("ad4af911-5bcd-47c5-a554-f35cea869472")
	got := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, driver, podUID)
	if !strings.HasPrefix(got, "dra-") || !strings.HasSuffix(got, "-reg.sock") {
		t.Fatalf("expected shortest hashed basename, got %q", got)
	}
	if strings.Contains(got, driver) {
		t.Fatalf("expected driver name to be omitted from shortest form, got %q", got)
	}
}

func TestRollingUpdateRegistrarSocketFile_respectsCustomRegistryDir(t *testing.T) {
	// A longer registry dir leaves less room for the driver name in the path.
	registryDir := path.Join("/var/lib/kubelet", strings.Repeat("x", 20), "plugins_registry")
	driver := "example.com/driver"
	podUID := types.UID("11111111-2222-3333-4444-555555555555")

	fullUID := driver + "-" + string(podUID) + "-reg.sock"
	if len(path.Join(registryDir, fullUID)) <= unixPathMax {
		t.Fatalf("test setup: expected full UID form to exceed limit with custom registry dir")
	}

	got := RollingUpdateRegistrarSocketFile(registryDir, driver, podUID)
	if len(path.Join(registryDir, got)) > unixPathMax {
		t.Fatalf("expected path within limit, got len %d: %q", len(path.Join(registryDir, got)), got)
	}
	if got == fullUID {
		t.Fatalf("expected fallback from full UID form, got %q", got)
	}
}

func TestRollingUpdateRegistrarSocketFile_deterministic(t *testing.T) {
	driver := "example.com/driver"
	podUID := types.UID("11111111-2222-3333-4444-555555555555")
	a := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, driver, podUID)
	b := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, driver, podUID)
	if a != b {
		t.Fatalf("expected stable name, got %q vs %q", a, b)
	}
}

func TestRollingUpdateRegistrarSocketFile_distinctInputs(t *testing.T) {
	uid := types.UID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	a := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, "driver.one", uid)
	b := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, "driver.two", uid)
	if a == b {
		t.Fatalf("different driver names should not collide: %q", a)
	}
	c := RollingUpdateRegistrarSocketFile(KubeletRegistryDir, "driver.one", "ffffffff-ffff-ffff-ffff-ffffffffffff")
	if a == c {
		t.Fatalf("different pod UIDs should not collide: %q", a)
	}
}
