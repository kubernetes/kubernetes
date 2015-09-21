package operatingsystem

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestGetOperatingSystem(t *testing.T) {
	var (
		backup       = etcOsRelease
		ubuntuTrusty = []byte(`NAME="Ubuntu"
VERSION="14.04, Trusty Tahr"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 14.04 LTS"
VERSION_ID="14.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"`)
		gentoo = []byte(`NAME=Gentoo
ID=gentoo
PRETTY_NAME="Gentoo/Linux"
ANSI_COLOR="1;32"
HOME_URL="http://www.gentoo.org/"
SUPPORT_URL="http://www.gentoo.org/main/en/support.xml"
BUG_REPORT_URL="https://bugs.gentoo.org/"
`)
		noPrettyName = []byte(`NAME="Ubuntu"
VERSION="14.04, Trusty Tahr"
ID=ubuntu
ID_LIKE=debian
VERSION_ID="14.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"`)
	)

	dir := os.TempDir()
	etcOsRelease = filepath.Join(dir, "etcOsRelease")

	defer func() {
		os.Remove(etcOsRelease)
		etcOsRelease = backup
	}()

	for expect, osRelease := range map[string][]byte{
		"Ubuntu 14.04 LTS": ubuntuTrusty,
		"Gentoo/Linux":     gentoo,
		"":                 noPrettyName,
	} {
		if err := ioutil.WriteFile(etcOsRelease, osRelease, 0600); err != nil {
			t.Fatalf("failed to write to %s: %v", etcOsRelease, err)
		}
		s, err := GetOperatingSystem()
		if s != expect {
			if expect == "" {
				t.Fatalf("Expected error 'PRETTY_NAME not found', but got %v", err)
			} else {
				t.Fatalf("Expected '%s', but got '%s'. Err=%v", expect, s, err)
			}
		}
	}
}

func TestIsContainerized(t *testing.T) {
	var (
		backup                      = proc1Cgroup
		nonContainerizedProc1Cgroup = []byte(`14:name=systemd:/
13:hugetlb:/
12:net_prio:/
11:perf_event:/
10:bfqio:/
9:blkio:/
8:net_cls:/
7:freezer:/
6:devices:/
5:memory:/
4:cpuacct:/
3:cpu:/
2:cpuset:/
`)
		containerizedProc1Cgroup = []byte(`9:perf_event:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
8:blkio:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
7:net_cls:/
6:freezer:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
5:devices:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
4:memory:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
3:cpuacct:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
2:cpu:/docker/3cef1b53c50b0fa357d994f8a1a8cd783c76bbf4f5dd08b226e38a8bd331338d
1:cpuset:/`)
	)

	dir := os.TempDir()
	proc1Cgroup = filepath.Join(dir, "proc1Cgroup")

	defer func() {
		os.Remove(proc1Cgroup)
		proc1Cgroup = backup
	}()

	if err := ioutil.WriteFile(proc1Cgroup, nonContainerizedProc1Cgroup, 0600); err != nil {
		t.Fatalf("failed to write to %s: %v", proc1Cgroup, err)
	}
	inContainer, err := IsContainerized()
	if err != nil {
		t.Fatal(err)
	}
	if inContainer {
		t.Fatal("Wrongly assuming containerized")
	}

	if err := ioutil.WriteFile(proc1Cgroup, containerizedProc1Cgroup, 0600); err != nil {
		t.Fatalf("failed to write to %s: %v", proc1Cgroup, err)
	}
	inContainer, err = IsContainerized()
	if err != nil {
		t.Fatal(err)
	}
	if !inContainer {
		t.Fatal("Wrongly assuming non-containerized")
	}
}
