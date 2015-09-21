package kernel

import (
	"testing"
)

func assertParseRelease(t *testing.T, release string, b *KernelVersionInfo, result int) {
	var (
		a *KernelVersionInfo
	)
	a, _ = ParseRelease(release)

	if r := CompareKernelVersion(a, b); r != result {
		t.Fatalf("Unexpected kernel version comparison result. Found %d, expected %d", r, result)
	}
	if a.Flavor != b.Flavor {
		t.Fatalf("Unexpected parsed kernel flavor.  Found %s, expected %s", a.Flavor, b.Flavor)
	}
}

func TestParseRelease(t *testing.T) {
	assertParseRelease(t, "3.8.0", &KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0}, 0)
	assertParseRelease(t, "3.4.54.longterm-1", &KernelVersionInfo{Kernel: 3, Major: 4, Minor: 54, Flavor: ".longterm-1"}, 0)
	assertParseRelease(t, "3.4.54.longterm-1", &KernelVersionInfo{Kernel: 3, Major: 4, Minor: 54, Flavor: ".longterm-1"}, 0)
	assertParseRelease(t, "3.8.0-19-generic", &KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0, Flavor: "-19-generic"}, 0)
	assertParseRelease(t, "3.12.8tag", &KernelVersionInfo{Kernel: 3, Major: 12, Minor: 8, Flavor: "tag"}, 0)
	assertParseRelease(t, "3.12-1-amd64", &KernelVersionInfo{Kernel: 3, Major: 12, Minor: 0, Flavor: "-1-amd64"}, 0)
}

func assertKernelVersion(t *testing.T, a, b *KernelVersionInfo, result int) {
	if r := CompareKernelVersion(a, b); r != result {
		t.Fatalf("Unexpected kernel version comparison result. Found %d, expected %d", r, result)
	}
}

func TestCompareKernelVersion(t *testing.T) {
	assertKernelVersion(t,
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		0)
	assertKernelVersion(t,
		&KernelVersionInfo{Kernel: 2, Major: 6, Minor: 0},
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		-1)
	assertKernelVersion(t,
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		&KernelVersionInfo{Kernel: 2, Major: 6, Minor: 0},
		1)
	assertKernelVersion(t,
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		0)
	assertKernelVersion(t,
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 5},
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		1)
	assertKernelVersion(t,
		&KernelVersionInfo{Kernel: 3, Major: 0, Minor: 20},
		&KernelVersionInfo{Kernel: 3, Major: 8, Minor: 0},
		-1)
}
