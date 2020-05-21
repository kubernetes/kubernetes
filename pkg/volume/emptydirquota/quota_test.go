package emptydirquota

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	kapiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const expectedDevice = "/dev/sdb2"

func TestParseFSDevice(t *testing.T) {
	tests := map[string]struct {
		dfOutput  string
		expDevice string
		expError  string
	}{
		"happy path": {
			dfOutput:  "Filesystem\n/dev/sdb2",
			expDevice: expectedDevice,
		},
		"happy path multi-token": {
			dfOutput:  "Filesystem\n/dev/sdb2           16444592     8  16444584   1% /var/openshift.local.volumes/",
			expDevice: expectedDevice,
		},
		"invalid tmpfs": {
			dfOutput: "Filesystem\ntmpfs",
			expError: invalidFilesystemError,
		},
		"invalid empty": {
			dfOutput: "",
			expError: unexpectedLineCountError,
		},
		"invalid one line": {
			dfOutput: "Filesystem\n",
			expError: invalidFilesystemError,
		},
		"invalid blank second line": {
			dfOutput: "Filesystem\n\n",
			expError: invalidFilesystemError,
		},
		"invalid too many lines": {
			dfOutput:  "Filesystem\n/dev/sdb2\ntmpfs\nwhatisgoingon",
			expDevice: expectedDevice,
		},
	}
	for name, test := range tests {
		t.Logf("running TestParseFSDevice: %s", name)
		device, err := parseFSDevice(test.dfOutput)
		if test.expDevice != "" && test.expDevice != device {
			t.Errorf("Unexpected filesystem device, expected: %s, got: %s", test.expDevice, device)
		}
		if test.expError != "" && (err == nil || !strings.Contains(err.Error(), test.expError)) {
			t.Errorf("Unexpected filesystem error, expected: %s, got: %s", test.expError, err)
		}
	}
}

// Avoid running actual commands to manage XFS quota:
type mockQuotaCommandRunner struct {
	RunFSDeviceCommandResponse     *cmdResponse
	RunFSTypeCommandResponse       *cmdResponse
	RunMountOptionsCommandResponse *cmdResponse

	RanApplyQuotaFSDevice string
	RanApplyQuota         *resource.Quantity
	RanApplyQuotaFSGroup  int64
}

func (m *mockQuotaCommandRunner) RunFSTypeCommand(dir string) (string, error) {
	if m.RunFSTypeCommandResponse != nil {
		return m.RunFSTypeCommandResponse.Stdout, m.RunFSTypeCommandResponse.Error
	}
	return "xfs", nil
}

func (m *mockQuotaCommandRunner) RunFSDeviceCommand(dir string) (string, error) {
	if m.RunFSDeviceCommandResponse != nil {
		return m.RunFSDeviceCommandResponse.Stdout, m.RunFSDeviceCommandResponse.Error
	}
	return "Filesystem\n/dev/sdb2", nil
}

func (m *mockQuotaCommandRunner) RunApplyQuotaCommand(fsDevice string, quota resource.Quantity, fsGroup int64) (string, string, error) {
	// Store these for assertions in tests:
	m.RanApplyQuotaFSDevice = fsDevice
	m.RanApplyQuota = &quota
	m.RanApplyQuotaFSGroup = fsGroup
	return "", "", nil
}

func (m *mockQuotaCommandRunner) RunMountOptionsCommand() (string, error) {
	return m.RunMountOptionsCommandResponse.Stdout, m.RunMountOptionsCommandResponse.Error
}

// Small struct for specifying how we want the various quota command runners to
// respond in tests:
type cmdResponse struct {
	Stdout string
	Stderr string
	Error  error
}

func TestApplyQuota(t *testing.T) {

	var defaultFSGroup int64
	defaultFSGroup = 1000050000

	tests := map[string]struct {
		FSGroupID *int64
		Quota     string

		FSTypeCmdResponse     *cmdResponse
		FSDeviceCmdResponse   *cmdResponse
		ApplyQuotaCmdResponse *cmdResponse

		ExpFSDevice string
		ExpError    string // sub-string to be searched for in error message
		ExpSkipped  bool
	}{
		"happy path": {
			Quota:     "512",
			FSGroupID: &defaultFSGroup,
		},
		"zero quota": {
			Quota:     "0",
			FSGroupID: &defaultFSGroup,
		},
		"invalid filesystem device": {
			Quota:     "512",
			FSGroupID: &defaultFSGroup,
			FSDeviceCmdResponse: &cmdResponse{
				Stdout: "Filesystem\ntmpfs",
				Stderr: "",
				Error:  nil,
			},
			ExpError:   invalidFilesystemError,
			ExpSkipped: true,
		},
		"error checking filesystem device": {
			Quota:     "512",
			FSGroupID: &defaultFSGroup,
			FSDeviceCmdResponse: &cmdResponse{
				Stdout: "",
				Stderr: "no such file or directory",
				Error:  errors.New("no such file or directory"), // Would be exit error in real life
			},
			ExpError:   "no such file or directory",
			ExpSkipped: true,
		},
		"non-xfs filesystem type": {
			Quota:     "512",
			FSGroupID: &defaultFSGroup,
			FSTypeCmdResponse: &cmdResponse{
				Stdout: "ext4",
				Stderr: "",
				Error:  nil,
			},
			ExpError:   "not on an XFS filesystem",
			ExpSkipped: true,
		},
		"error checking filesystem type": {
			Quota:     "512",
			FSGroupID: &defaultFSGroup,
			FSTypeCmdResponse: &cmdResponse{
				Stdout: "",
				Stderr: "no such file or directory",
				Error:  errors.New("no such file or directory"), // Would be exit error in real life
			},
			ExpError:   "unable to check filesystem type",
			ExpSkipped: true,
		},
		// Should result in success, but no quota actually gets applied:
		"no FSGroup": {
			Quota:      "512",
			ExpSkipped: true,
		},
	}

	for name, test := range tests {
		t.Logf("running TestApplyQuota: %s", name)
		quotaApplicator := xfsQuotaApplicator{}
		// Replace the real command runner with our mock:
		mockCmdRunner := mockQuotaCommandRunner{}
		quotaApplicator.cmdRunner = &mockCmdRunner
		fakeDir := "/var/lib/origin/openshift.local.volumes/pods/d71f6949-cb3f-11e5-aedf-989096de63cb"

		// Configure the default happy path command responses if nothing was specified
		// by the test:
		if test.FSTypeCmdResponse == nil {
			// Configure the default happy path response:
			test.FSTypeCmdResponse = &cmdResponse{
				Stdout: "xfs",
				Stderr: "",
				Error:  nil,
			}
		}
		if test.FSDeviceCmdResponse == nil {
			test.FSDeviceCmdResponse = &cmdResponse{
				Stdout: "Filesystem\n/dev/sdb2",
				Stderr: "",
				Error:  nil,
			}
		}

		if test.ApplyQuotaCmdResponse == nil {
			test.ApplyQuotaCmdResponse = &cmdResponse{
				Stdout: "",
				Stderr: "",
				Error:  nil,
			}
		}

		mockCmdRunner.RunFSDeviceCommandResponse = test.FSDeviceCmdResponse
		mockCmdRunner.RunFSTypeCommandResponse = test.FSTypeCmdResponse

		quota := resource.MustParse(test.Quota)
		err := quotaApplicator.Apply(fakeDir, kapiv1.StorageMediumDefault, &kapiv1.Pod{}, test.FSGroupID, quota)
		if test.ExpError == "" && !test.ExpSkipped {
			// Expecting success case:
			if mockCmdRunner.RanApplyQuotaFSDevice != "/dev/sdb2" {
				t.Errorf("failed: '%s', expected quota applied to: %s, got: %s", name, "/dev/sdb2", mockCmdRunner.RanApplyQuotaFSDevice)
			}
			if mockCmdRunner.RanApplyQuota.Value() != quota.Value() {
				t.Errorf("failed: '%s', expected quota: %d, got: %d", name, quota.Value(),
					mockCmdRunner.RanApplyQuota.Value())
			}
			if mockCmdRunner.RanApplyQuotaFSGroup != *test.FSGroupID {
				t.Errorf("failed: '%s', expected FSGroup: %d, got: %d", name, test.FSGroupID, mockCmdRunner.RanApplyQuotaFSGroup)
			}
		} else if test.ExpError != "" {
			// Expecting error case:
			if err == nil {
				t.Errorf("failed: '%s', expected error but got none", name)
			} else if !strings.Contains(err.Error(), test.ExpError) {
				t.Errorf("failed: '%s', expected error containing '%s', got: '%s'", name, test.ExpError, err)
			}
		}

		if test.ExpSkipped {
			if mockCmdRunner.RanApplyQuota != nil {
				t.Errorf("failed: '%s', expected error but quota was applied", name)
			}
			if mockCmdRunner.RanApplyQuotaFSGroup != 0 {
				t.Errorf("failed: '%s', expected error but quota was applied", name)
			}
			if mockCmdRunner.RanApplyQuotaFSDevice != "" {
				t.Errorf("failed: '%s', expected error but quota was applied", name)
			}
		}
	}
}

func TestIsMountedWithGrpquota(t *testing.T) {
	// Substitute in the actual mount line we're after for each test:
	mountOutput := `/dev/mapper/fedora_system-root on / type ext4 (rw,relatime,seclabel,data=ordered)
selinuxfs on /sys/fs/selinux type selinuxfs (rw,relatime)
systemd-1 on /proc/sys/fs/binfmt_misc type autofs (rw,relatime,fd=26,pgrp=1,timeout=0,minproto=5,maxproto=5,direct)
debugfs on /sys/kernel/debug type debugfs (rw,relatime,seclabel)
mqueue on /dev/mqueue type mqueue (rw,relatime,seclabel)
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,seclabel)
tmpfs on /tmp type tmpfs (rw,seclabel)
%s
nfsd on /proc/fs/nfsd type nfsd (rw,relatime)
/dev/sda1 on /boot type ext4 (rw,relatime,seclabel,data=ordered)
/dev/mapper/fedora_system-home on /home type ext4 (rw,relatime,seclabel,data=ordered)
/dev/sdb1 on /storage type btrfs (rw,relatime,seclabel,space_cache,subvolid=5,subvol=/)
`
	var fsDevice = "/dev/mapper/openshift--vol--dir"
	var volumeDir = "/var/lib/origin/openshift.local.volumes"

	tests := map[string]struct {
		MountLine      string
		ExpectedResult bool
		ExpError       string // sub-string to be searched for in error message
	}{
		"grpquota": {
			MountLine:      fmt.Sprintf("%s on %s type xfs (rw,relatime,seclabel,attr2,inode64,grpquota)", fsDevice, volumeDir),
			ExpectedResult: true,
		},
		"gquota": {
			// May not be possible in the real world (would show up as grpquota) but just in case:
			MountLine:      fmt.Sprintf("%s on %s type xfs (rw,relatime,seclabel,attr2,inode64,gquota)", fsDevice, volumeDir),
			ExpectedResult: true,
		},
		"gqnoenforce": {
			MountLine:      fmt.Sprintf("%s on %s type xfs (rw,relatime,seclabel,attr2,inode64,gqnoenforce)", fsDevice, volumeDir),
			ExpectedResult: false,
		},
		"noquota": {
			MountLine:      fmt.Sprintf("%s on %s type xfs (rw,relatime,seclabel,attr2,inode64,noquota)", fsDevice, volumeDir),
			ExpectedResult: false,
		},
		"device not in output": {
			MountLine:      fmt.Sprintf("/dev/sdb1 on %s type xfs (rw,relatime,seclabel,attr2,inode64,noquota)", volumeDir),
			ExpectedResult: false,
			ExpError:       "unable to find device",
		},
	}

	for name, test := range tests {
		t.Logf("running TestIsMountedWithGrpquota: %s", name)

		mockCmdRunner := &mockQuotaCommandRunner{}
		mockCmdRunner.RunMountOptionsCommandResponse = &cmdResponse{
			Stdout: fmt.Sprintf(mountOutput, test.MountLine),
			Stderr: "",
			Error:  nil,
		}
		mockCmdRunner.RunFSDeviceCommandResponse = &cmdResponse{
			Stdout: fmt.Sprintf("Filesystem\n%s", fsDevice),
			Stderr: "",
			Error:  nil,
		}

		mountedWithQuota, err := isMountedWithGrpquota(mockCmdRunner, volumeDir)
		if len(test.ExpError) > 0 {
			if err == nil {
				t.Errorf("%q, expected error but got none", name)
			} else if !strings.Contains(err.Error(), test.ExpError) {
				t.Errorf("%q, expected error containing %q, got: %q", name, test.ExpError, err)
			}
		}

		if test.ExpectedResult != mountedWithQuota {
			t.Errorf("%q, expected %t, got: %t", name, test.ExpectedResult, mountedWithQuota)
		}

	}
}
