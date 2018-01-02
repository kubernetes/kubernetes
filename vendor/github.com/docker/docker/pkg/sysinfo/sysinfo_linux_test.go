package sysinfo

import (
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

func TestReadProcBool(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "test-sysinfo-proc")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	procFile := filepath.Join(tmpDir, "read-proc-bool")
	err = ioutil.WriteFile(procFile, []byte("1"), 0644)
	require.NoError(t, err)

	if !readProcBool(procFile) {
		t.Fatal("expected proc bool to be true, got false")
	}

	if err := ioutil.WriteFile(procFile, []byte("0"), 0644); err != nil {
		t.Fatal(err)
	}
	if readProcBool(procFile) {
		t.Fatal("expected proc bool to be false, got true")
	}

	if readProcBool(path.Join(tmpDir, "no-exist")) {
		t.Fatal("should be false for non-existent entry")
	}

}

func TestCgroupEnabled(t *testing.T) {
	cgroupDir, err := ioutil.TempDir("", "cgroup-test")
	require.NoError(t, err)
	defer os.RemoveAll(cgroupDir)

	if cgroupEnabled(cgroupDir, "test") {
		t.Fatal("cgroupEnabled should be false")
	}

	err = ioutil.WriteFile(path.Join(cgroupDir, "test"), []byte{}, 0644)
	require.NoError(t, err)

	if !cgroupEnabled(cgroupDir, "test") {
		t.Fatal("cgroupEnabled should be true")
	}
}

func TestNew(t *testing.T) {
	sysInfo := New(false)
	require.NotNil(t, sysInfo)
	checkSysInfo(t, sysInfo)

	sysInfo = New(true)
	require.NotNil(t, sysInfo)
	checkSysInfo(t, sysInfo)
}

func checkSysInfo(t *testing.T, sysInfo *SysInfo) {
	// Check if Seccomp is supported, via CONFIG_SECCOMP.then sysInfo.Seccomp must be TRUE , else FALSE
	if err := unix.Prctl(unix.PR_GET_SECCOMP, 0, 0, 0, 0); err != unix.EINVAL {
		// Make sure the kernel has CONFIG_SECCOMP_FILTER.
		if err := unix.Prctl(unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER, 0, 0, 0); err != unix.EINVAL {
			require.True(t, sysInfo.Seccomp)
		}
	} else {
		require.False(t, sysInfo.Seccomp)
	}
}

func TestNewAppArmorEnabled(t *testing.T) {
	// Check if AppArmor is supported. then it must be TRUE , else FALSE
	if _, err := os.Stat("/sys/kernel/security/apparmor"); err != nil {
		t.Skip("App Armor Must be Enabled")
	}

	sysInfo := New(true)
	require.True(t, sysInfo.AppArmor)
}

func TestNewAppArmorDisabled(t *testing.T) {
	// Check if AppArmor is supported. then it must be TRUE , else FALSE
	if _, err := os.Stat("/sys/kernel/security/apparmor"); !os.IsNotExist(err) {
		t.Skip("App Armor Must be Disabled")
	}

	sysInfo := New(true)
	require.False(t, sysInfo.AppArmor)
}

func TestNumCPU(t *testing.T) {
	cpuNumbers := NumCPU()
	if cpuNumbers <= 0 {
		t.Fatal("CPU returned must be greater than zero")
	}
}
