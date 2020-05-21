package emptydirquota

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog"
)

// QuotaApplicator is used to apply quota to an emptyDir volume.
type QuotaApplicator interface {
	// Apply the quota to the given EmptyDir path:
	Apply(dir string, medium v1.StorageMedium, pod *v1.Pod, fsGroup *int64, quota resource.Quantity) error
}

type xfsQuotaApplicator struct {
	cmdRunner quotaCommandRunner
}

// NewQuotaApplicator checks the filesystem type for the configured volume directory
// and returns an appropriate implementation of the quota applicator. If the filesystem
// does not appear to be a type we support quotas on, an error is returned.
func NewQuotaApplicator(volumeDirectory string) (QuotaApplicator, error) {

	cmdRunner := &realQuotaCommandRunner{}
	isXFS, err := isXFS(cmdRunner, volumeDirectory)
	if err != nil {
		return nil, err
	}

	if isXFS {
		mountedWithGrpquota, mountErr := isMountedWithGrpquota(cmdRunner, volumeDirectory)
		if mountErr != nil {
			return nil, mountErr
		}
		if !mountedWithGrpquota {
			return nil, fmt.Errorf("%s is not on a filesystem mounted with the grpquota option", volumeDirectory)
		}

		// Make sure xfs_quota is on the PATH, otherwise we're not going to get very far:
		_, pathErr := exec.LookPath("xfs_quota")
		if pathErr != nil {
			return nil, pathErr
		}

		return &xfsQuotaApplicator{
			cmdRunner: cmdRunner,
		}, nil
	}

	// If we were unable to find a quota supported filesystem type, return an error:
	return nil, fmt.Errorf("%s is not on a supported filesystem for local volume quota", volumeDirectory)
}

// quotaCommandRunner interface is used to abstract the actual running of
// commands so we can unit test more behavior.
type quotaCommandRunner interface {
	RunFSTypeCommand(dir string) (string, error)
	RunFSDeviceCommand(dir string) (string, error)
	RunApplyQuotaCommand(fsDevice string, quota resource.Quantity, fsGroup int64) (string, string, error)
	RunMountOptionsCommand() (string, error)
}

type realQuotaCommandRunner struct {
}

func (cr *realQuotaCommandRunner) RunFSTypeCommand(dir string) (string, error) {
	args := []string{"-f", "-c", "%T", dir}
	outBytes, err := exec.Command("stat", args...).Output()
	return string(outBytes), err
}

func (cr *realQuotaCommandRunner) RunFSDeviceCommand(dir string) (string, error) {
	outBytes, err := exec.Command("df", "--output=source", dir).Output()
	return string(outBytes), err
}

func (cr *realQuotaCommandRunner) RunApplyQuotaCommand(fsDevice string, quota resource.Quantity, fsGroup int64) (string, string, error) {
	args := []string{"-x", "-c",
		fmt.Sprintf("limit -g bsoft=%d bhard=%d %d", quota.Value(), quota.Value(), fsGroup),
		fsDevice,
	}

	cmd := exec.Command("xfs_quota", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	klog.V(5).Infof("Ran: xfs_quota %s", args)
	return "", stderr.String(), err
}

func (cr *realQuotaCommandRunner) RunMountOptionsCommand() (string, error) {
	outBytes, err := exec.Command("mount").Output()
	return string(outBytes), err
}

// Apply sets the actual quota on a device for an emptyDir volume if possible. Will return an error
// if anything goes wrong during the process. (not an XFS filesystem, etc) If the volume medium is set
// to memory, or no FSGroup is provided (indicating the request matched an SCC set to RunAsAny), this
// method will effectively no-op.
func (xqa *xfsQuotaApplicator) Apply(dir string, medium v1.StorageMedium, pod *v1.Pod, fsGroup *int64, quota resource.Quantity) error {

	if medium == v1.StorageMediumMemory {
		klog.V(5).Infof("Skipping quota application due to memory storage medium.")
		return nil
	}
	isXFS, err := isXFS(xqa.cmdRunner, dir)
	if err != nil {
		return err
	}
	if !isXFS {
		return fmt.Errorf("unable to apply quota: %s is not on an XFS filesystem", dir)
	}
	if fsGroup == nil {
		// This indicates the operation matched an SCC with FSGroup strategy RunAsAny.
		// Not an error condition.
		klog.V(5).Infof("Unable to apply XFS quota, no FSGroup specified.")
		return nil
	}

	volDevice, err := xqa.getFSDevice(dir)
	if err != nil {
		return err
	}

	err = xqa.applyQuota(volDevice, quota, *fsGroup)
	if err != nil {
		return err
	}

	return nil
}

func (xqa *xfsQuotaApplicator) applyQuota(volDevice string, quota resource.Quantity, fsGroupID int64) error {
	_, stderr, err := xqa.cmdRunner.RunApplyQuotaCommand(volDevice, quota, fsGroupID)

	// xfs_quota is very happy to fail but return a success code, likely due to its
	// interactive shell approach. Grab stderr, if we see anything written to it we'll
	// consider this an error.
	//
	// If we exit non-zero *and* write to stderr, stderr is likely to have the details on what
	// actually went wrong, so we'll use this as the error message instead.
	if len(stderr) > 0 {
		return fmt.Errorf("error applying quota: %s", stderr)
	}

	if err != nil {
		return fmt.Errorf("error applying quota: %v", err)
	}

	klog.V(4).Infof("XFS quota applied: device=%s, quota=%d, fsGroup=%d", volDevice, quota.Value(), fsGroupID)
	return nil
}

func (xqa *xfsQuotaApplicator) getFSDevice(dir string) (string, error) {
	return getFSDevice(dir, xqa.cmdRunner)
}

// GetFSDevice returns the filesystem device for a given path. To do this we
// run df on the path, returning a header line and the line we're
// interested in. The first string token in that line will be the device name.
func GetFSDevice(dir string) (string, error) {
	return getFSDevice(dir, &realQuotaCommandRunner{})
}

func getFSDevice(dir string, cmdRunner quotaCommandRunner) (string, error) {
	out, err := cmdRunner.RunFSDeviceCommand(dir)
	if err != nil {
		return "", fmt.Errorf("unable to find filesystem device for dir %s: %s", dir, err)
	}
	fsDevice, parseErr := parseFSDevice(out)
	return fsDevice, parseErr
}

func parseFSDevice(dfOutput string) (string, error) {
	// Need to skip the df header line starting with "Filesystem", and grab the first
	// word of the following line which will be our device path.
	lines := strings.Split(dfOutput, "\n")
	if len(lines) < 2 {
		return "", fmt.Errorf("%s: %s", unexpectedLineCountError, dfOutput)
	}

	fsDevice := strings.Split(lines[1], " ")[0]
	// Make sure it looks like a device:
	if !strings.HasPrefix(fsDevice, "/") {
		return "", fmt.Errorf("%s: %s", invalidFilesystemError, fsDevice)
	}

	return fsDevice, nil
}

// isMountedWithGrpquota checks if the device the given directory is on has been
// mounted with the grpquota option.
func isMountedWithGrpquota(cmdRunner quotaCommandRunner, dir string) (bool, error) {
	fsDevice, fsDeviceErr := getFSDevice(dir, cmdRunner)
	if fsDeviceErr != nil {
		return false, fsDeviceErr
	}

	out, err := cmdRunner.RunMountOptionsCommand()
	if err != nil {
		return false, fmt.Errorf("unable to check mount options for dir %s: %s", dir, err)
	}

	// Keep this simple, locate a line in the mount output that matches the fs device
	// the directory is on. Once found, make sure "grpquota" is present in the line.
	// "gquota" is an alias for "grpquota", but if used "grpquota" will be what appears in
	// mount output. (however check for both just in case)
	//
	// Use of "noquota" or "gqnoenforce" is mutually exclusive with the above, if some
	// combination of these are specified, the last one wins, but there should be no way for
	// them to all appear in the mount output, thus we just need to see one of the positive
	// aliases.
	lines := strings.Split(out, "\n")
	for _, line := range lines {
		tokens := strings.Split(line, " ")
		if len(tokens) > 0 && tokens[0] == fsDevice {
			if strings.Contains(line, "grpquota") || strings.Contains(line, "gquota") {
				return true, nil
			}
			return false, nil
		}
	}
	return false, fmt.Errorf("unable to find device %s in mount output", fsDevice)
}

// isXFS checks if a dir is on an XFS filesystem.
func isXFS(cmdRunner quotaCommandRunner, dir string) (bool, error) {
	out, err := cmdRunner.RunFSTypeCommand(dir)
	if err != nil {
		return false, fmt.Errorf("unable to check filesystem type for emptydir volume %s: %s", dir, err)
	}
	if strings.TrimSpace(out) == "xfs" {
		return true, nil
	}
	return false, nil
}

const (
	invalidFilesystemError   = "found invalid filesystem device"
	unexpectedLineCountError = "unexpected line count in df output"
)
