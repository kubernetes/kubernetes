//go:build linux
// +build linux

/*
Copyright 2014 The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/moby/sys/mountinfo"
	"golang.org/x/sys/unix"

	"k8s.io/klog/v2"
	utilexec "k8s.io/utils/exec"
)

const (
	// Number of fields per line in /proc/mounts as per the fstab man page.
	expectedNumFieldsPerLine = 6
	// Location of the mount file to use
	procMountsPath = "/proc/mounts"
	// Location of the mountinfo file
	procMountInfoPath = "/proc/self/mountinfo"
	// 'fsck' found errors and corrected them
	fsckErrorsCorrected = 1
	// 'fsck' found errors but exited without correcting them
	fsckErrorsUncorrected = 4
	// Error thrown by exec cmd.Run() when process spawned by cmd.Start() completes before cmd.Wait() is called (see - k/k issue #103753)
	errNoChildProcesses = "wait: no child processes"
	// Error returned by some `umount` implementations when the specified path is not a mount point
	errNotMounted = "not mounted"
)

var (
	// Error statx support since Linux 4.11, https://man7.org/linux/man-pages/man2/statx.2.html
	errStatxNotSupport = errors.New("the statx syscall is not supported. At least Linux kernel 4.11 is needed")
)

// Mounter provides the default implementation of mount.Interface
// for the linux platform.  This implementation assumes that the
// kubelet is running in the host's root mount namespace.
type Mounter struct {
	mounterPath                string
	withSystemd                *bool
	trySystemd                 bool
	withSafeNotMountedBehavior bool
}

var _ MounterForceUnmounter = &Mounter{}

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	return &Mounter{
		mounterPath:                mounterPath,
		trySystemd:                 true,
		withSafeNotMountedBehavior: detectSafeNotMountedBehavior(),
	}
}

// NewWithoutSystemd returns a Linux specific mount.Interface for the current
// system. It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting. Any
// detection for systemd functionality is disabled with this Mounter.
func NewWithoutSystemd(mounterPath string) Interface {
	return &Mounter{
		mounterPath:                mounterPath,
		trySystemd:                 false,
		withSafeNotMountedBehavior: detectSafeNotMountedBehavior(),
	}
}

// hasSystemd validates that the withSystemd bool is set, if it is not,
// detectSystemd will be called once for this Mounter instance.
func (mounter *Mounter) hasSystemd() bool {
	if !mounter.trySystemd {
		mounter.withSystemd = &mounter.trySystemd
	}

	if mounter.withSystemd == nil {
		withSystemd := detectSystemd()
		mounter.withSystemd = &withSystemd
	}

	return *mounter.withSystemd
}

// Map unix.Statfs mount flags ro, nodev, noexec, nosuid, noatime, relatime,
// nodiratime to mount option flag strings.
func getBindMountOptions(path string, statfs func(path string, buf *unix.Statfs_t) (err error)) ([]string, error) {
	var s unix.Statfs_t
	var mountOpts []string
	if err := statfs(path, &s); err != nil {
		return nil, &os.PathError{Op: "statfs", Path: path, Err: err}
	}
	flagMapping := map[int]string{
		unix.MS_RDONLY:     "ro",
		unix.MS_NODEV:      "nodev",
		unix.MS_NOEXEC:     "noexec",
		unix.MS_NOSUID:     "nosuid",
		unix.MS_NOATIME:    "noatime",
		unix.MS_RELATIME:   "relatime",
		unix.MS_NODIRATIME: "nodiratime",
	}
	for k, v := range flagMapping {
		if int(s.Flags)&k == k {
			mountOpts = append(mountOpts, v)
		}
	}
	return mountOpts, nil
}

// Performs a bind mount with the specified options, and then remounts
// the mount point with the same `nodev`, `nosuid`, `noexec`, `nosuid`, `noatime`,
// `relatime`, `nodiratime` options as the original mount point.
func (mounter *Mounter) bindMountSensitive(mounterPath string, mountCmd string, source string, target string, fstype string, bindOpts []string, bindRemountOpts []string, bindRemountOptsSensitive []string, mountFlags []string, systemdMountRequired bool) error {
	err := mounter.doMount(mounterPath, mountCmd, source, target, fstype, bindOpts, bindRemountOptsSensitive, mountFlags, systemdMountRequired)
	if err != nil {
		return err
	}
	// Check if the source has ro, nodev, noexec, nosuid, noatime, relatime,
	// nodiratime flag...
	fixMountOpts, err := getBindMountOptions(source, unix.Statfs)
	if err != nil {
		return &os.PathError{Op: "statfs", Path: source, Err: err}
	}
	// ... and retry the mount with flags found above.
	bindRemountOpts = append(bindRemountOpts, fixMountOpts...)
	return mounter.doMount(mounterPath, mountCmd, source, target, fstype, bindRemountOpts, bindRemountOptsSensitive, mountFlags, systemdMountRequired)
}

// Mount mounts source to target as fstype with given options. 'source' and 'fstype' must
// be an empty string in case it's not required, e.g. for remount, or for auto filesystem
// type, where kernel handles fstype for you. The mount 'options' is a list of options,
// currently come from mount(8), e.g. "ro", "remount", "bind", etc. If no more option is
// required, call Mount with an empty string list or nil.
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return mounter.MountSensitive(source, target, fstype, options, nil)
}

// MountSensitive is the same as Mount() but this method allows
// sensitiveOptions to be passed in a separate parameter from the normal
// mount options and ensures the sensitiveOptions are never logged. This
// method should be used by callers that pass sensitive material (like
// passwords) as mount options.
func (mounter *Mounter) MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	// Path to mounter binary if containerized mounter is needed. Otherwise, it is set to empty.
	// All Linux distros are expected to be shipped with a mount utility that a support bind mounts.
	mounterPath := ""
	bind, bindOpts, bindRemountOpts, bindRemountOptsSensitive := MakeBindOptsSensitive(options, sensitiveOptions)
	if bind {
		return mounter.bindMountSensitive(mounterPath, defaultMountCommand, source, target, fstype, bindOpts, bindRemountOpts, bindRemountOptsSensitive, nil /* mountFlags */, mounter.trySystemd)
	}
	// The list of filesystems that require containerized mounter on GCI image cluster
	fsTypesNeedMounter := map[string]struct{}{
		"nfs":       {},
		"glusterfs": {},
		"ceph":      {},
		"cifs":      {},
	}
	if _, ok := fsTypesNeedMounter[fstype]; ok {
		mounterPath = mounter.mounterPath
	}
	return mounter.doMount(mounterPath, defaultMountCommand, source, target, fstype, options, sensitiveOptions, nil /* mountFlags */, mounter.trySystemd)
}

// MountSensitiveWithoutSystemd is the same as MountSensitive() but disable using systemd mount.
func (mounter *Mounter) MountSensitiveWithoutSystemd(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return mounter.MountSensitiveWithoutSystemdWithMountFlags(source, target, fstype, options, sensitiveOptions, nil /* mountFlags */)
}

// MountSensitiveWithoutSystemdWithMountFlags is the same as MountSensitiveWithoutSystemd with additional mount flags.
func (mounter *Mounter) MountSensitiveWithoutSystemdWithMountFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	mounterPath := ""
	bind, bindOpts, bindRemountOpts, bindRemountOptsSensitive := MakeBindOptsSensitive(options, sensitiveOptions)
	if bind {
		return mounter.bindMountSensitive(mounterPath, defaultMountCommand, source, target, fstype, bindOpts, bindRemountOpts, bindRemountOptsSensitive, mountFlags, false)
	}
	// The list of filesystems that require containerized mounter on GCI image cluster
	fsTypesNeedMounter := map[string]struct{}{
		"nfs":       {},
		"glusterfs": {},
		"ceph":      {},
		"cifs":      {},
	}
	if _, ok := fsTypesNeedMounter[fstype]; ok {
		mounterPath = mounter.mounterPath
	}
	return mounter.doMount(mounterPath, defaultMountCommand, source, target, fstype, options, sensitiveOptions, mountFlags, false)
}

// doMount runs the mount command. mounterPath is the path to mounter binary if containerized mounter is used.
// sensitiveOptions is an extension of options except they will not be logged (because they may contain sensitive material)
// systemdMountRequired is an extension of option to decide whether uses systemd mount.
func (mounter *Mounter) doMount(mounterPath string, mountCmd string, source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string, systemdMountRequired bool) error {
	mountArgs, mountArgsLogStr := MakeMountArgsSensitiveWithMountFlags(source, target, fstype, options, sensitiveOptions, mountFlags)
	if len(mounterPath) > 0 {
		mountArgs = append([]string{mountCmd}, mountArgs...)
		mountArgsLogStr = mountCmd + " " + mountArgsLogStr
		mountCmd = mounterPath
	}

	if systemdMountRequired && mounter.hasSystemd() {
		// Try to run mount via systemd-run --scope. This will escape the
		// service where kubelet runs and any fuse daemons will be started in a
		// specific scope. kubelet service than can be restarted without killing
		// these fuse daemons.
		//
		// Complete command line (when mounterPath is not used):
		// systemd-run --description=... --scope -- mount -t <type> <what> <where>
		//
		// Expected flow:
		// * systemd-run creates a transient scope (=~ cgroup) and executes its
		//   argument (/bin/mount) there.
		// * mount does its job, forks a fuse daemon if necessary and finishes.
		//   (systemd-run --scope finishes at this point, returning mount's exit
		//   code and stdout/stderr - thats one of --scope benefits).
		// * systemd keeps the fuse daemon running in the scope (i.e. in its own
		//   cgroup) until the fuse daemon dies (another --scope benefit).
		//   Kubelet service can be restarted and the fuse daemon survives.
		// * When the fuse daemon dies (e.g. during unmount) systemd removes the
		//   scope automatically.
		//
		// systemd-mount is not used because it's too new for older distros
		// (CentOS 7, Debian Jessie).
		mountCmd, mountArgs, mountArgsLogStr = AddSystemdScopeSensitive("systemd-run", target, mountCmd, mountArgs, mountArgsLogStr)
		// } else {
		// No systemd-run on the host (or we failed to check it), assume kubelet
		// does not run as a systemd service.
		// No code here, mountCmd and mountArgs are already populated.
	}

	// Logging with sensitive mount options removed.
	klog.V(4).Infof("Mounting cmd (%s) with arguments (%s)", mountCmd, mountArgsLogStr)
	command := exec.Command(mountCmd, mountArgs...)
	output, err := command.CombinedOutput()
	if err != nil {
		if err.Error() == errNoChildProcesses {
			if command.ProcessState.Success() {
				// We don't consider errNoChildProcesses an error if the process itself succeeded (see - k/k issue #103753).
				return nil
			}
			// Rewrite err with the actual exit error of the process.
			err = &exec.ExitError{ProcessState: command.ProcessState}
		}
		klog.Errorf("Mount failed: %v\nMounting command: %s\nMounting arguments: %s\nOutput: %s\n", err, mountCmd, mountArgsLogStr, string(output))
		return fmt.Errorf("mount failed: %v\nMounting command: %s\nMounting arguments: %s\nOutput: %s",
			err, mountCmd, mountArgsLogStr, string(output))
	}
	return err
}

// detectSystemd returns true if OS runs with systemd as init. When not sure
// (permission errors, ...), it returns false.
// There may be different ways how to detect systemd, this one makes sure that
// systemd-runs (needed by Mount()) works.
func detectSystemd() bool {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		klog.V(2).Infof("Detected OS without systemd")
		return false
	}
	// Try to run systemd-run --scope /bin/true, that should be enough
	// to make sure that systemd is really running and not just installed,
	// which happens when running in a container with a systemd-based image
	// but with different pid 1.
	cmd := exec.Command("systemd-run", "--description=Kubernetes systemd probe", "--scope", "true")
	output, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Cannot run systemd-run, assuming non-systemd OS")
		klog.V(4).Infof("systemd-run output: %s, failed with: %v", string(output), err)
		return false
	}
	klog.V(2).Infof("Detected OS with systemd")
	return true
}

// detectSafeNotMountedBehavior returns true if the umount implementation replies "not mounted"
// when the specified path is not mounted. When not sure (permission errors, ...), it returns false.
// When possible, we will trust umount's message and avoid doing our own mount point checks.
// More info: https://github.com/util-linux/util-linux/blob/v2.2/mount/umount.c#L179
func detectSafeNotMountedBehavior() bool {
	return detectSafeNotMountedBehaviorWithExec(utilexec.New())
}

// detectSafeNotMountedBehaviorWithExec is for testing with FakeExec.
func detectSafeNotMountedBehaviorWithExec(exec utilexec.Interface) bool {
	// create a temp dir and try to umount it
	path, err := os.MkdirTemp("", "kubelet-detect-safe-umount")
	if err != nil {
		klog.V(4).Infof("Cannot create temp dir to detect safe 'not mounted' behavior: %v", err)
		return false
	}
	defer os.RemoveAll(path)
	cmd := exec.Command("umount", path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(string(output), errNotMounted) {
			klog.V(4).Infof("Detected umount with safe 'not mounted' behavior")
			return true
		}
		klog.V(4).Infof("'umount %s' failed with: %v, output: %s", path, err, string(output))
	}
	klog.V(4).Infof("Detected umount with unsafe 'not mounted' behavior")
	return false
}

// MakeMountArgs makes the arguments to the mount(8) command.
// options MUST not contain sensitive material (like passwords).
func MakeMountArgs(source, target, fstype string, options []string) (mountArgs []string) {
	mountArgs, _ = MakeMountArgsSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
	return mountArgs
}

// MakeMountArgsSensitive makes the arguments to the mount(8) command.
// sensitiveOptions is an extension of options except they will not be logged (because they may contain sensitive material)
func MakeMountArgsSensitive(source, target, fstype string, options []string, sensitiveOptions []string) (mountArgs []string, mountArgsLogStr string) {
	return MakeMountArgsSensitiveWithMountFlags(source, target, fstype, options, sensitiveOptions, nil /* mountFlags */)
}

// MakeMountArgsSensitiveWithMountFlags makes the arguments to the mount(8) command.
// sensitiveOptions is an extension of options except they will not be logged (because they may contain sensitive material)
// mountFlags are additional mount flags that are not related with the fstype
// and mount options
func MakeMountArgsSensitiveWithMountFlags(source, target, fstype string, options []string, sensitiveOptions []string, mountFlags []string) (mountArgs []string, mountArgsLogStr string) {
	// Build mount command as follows:
	//   mount [$mountFlags] [-t $fstype] [-o $options] [$source] $target
	mountArgs = []string{}
	mountArgsLogStr = ""

	mountArgs = append(mountArgs, mountFlags...)
	mountArgsLogStr += strings.Join(mountFlags, " ")

	if len(fstype) > 0 {
		mountArgs = append(mountArgs, "-t", fstype)
		mountArgsLogStr += strings.Join(mountArgs, " ")
	}
	if len(options) > 0 || len(sensitiveOptions) > 0 {
		combinedOptions := []string{}
		combinedOptions = append(combinedOptions, options...)
		combinedOptions = append(combinedOptions, sensitiveOptions...)
		mountArgs = append(mountArgs, "-o", strings.Join(combinedOptions, ","))
		// exclude sensitiveOptions from log string
		mountArgsLogStr += " -o " + sanitizedOptionsForLogging(options, sensitiveOptions)
	}
	if len(source) > 0 {
		mountArgs = append(mountArgs, source)
		mountArgsLogStr += " " + source
	}
	mountArgs = append(mountArgs, target)
	mountArgsLogStr += " " + target

	return mountArgs, mountArgsLogStr
}

// AddSystemdScope adds "system-run --scope" to given command line
// If args contains sensitive material, use AddSystemdScopeSensitive to construct
// a safe to log string.
func AddSystemdScope(systemdRunPath, mountName, command string, args []string) (string, []string) {
	descriptionArg := fmt.Sprintf("--description=Kubernetes transient mount for %s", mountName)
	systemdRunArgs := []string{descriptionArg, "--scope", "--", command}
	return systemdRunPath, append(systemdRunArgs, args...)
}

// AddSystemdScopeSensitive adds "system-run --scope" to given command line
// It also accepts takes a sanitized string containing mount arguments, mountArgsLogStr,
// and returns the string appended to the systemd command for logging.
func AddSystemdScopeSensitive(systemdRunPath, mountName, command string, args []string, mountArgsLogStr string) (string, []string, string) {
	descriptionArg := fmt.Sprintf("--description=Kubernetes transient mount for %s", mountName)
	systemdRunArgs := []string{descriptionArg, "--scope", "--", command}
	return systemdRunPath, append(systemdRunArgs, args...), strings.Join(systemdRunArgs, " ") + " " + mountArgsLogStr
}

// Unmount unmounts the target.
// If the mounter has safe "not mounted" behavior, no error will be returned when the target is not a mount point.
func (mounter *Mounter) Unmount(target string) error {
	klog.V(4).Infof("Unmounting %s", target)
	command := exec.Command("umount", target)
	output, err := command.CombinedOutput()
	if err != nil {
		return checkUmountError(target, command, output, err, mounter.withSafeNotMountedBehavior)
	}
	return nil
}

// UnmountWithForce unmounts given target but will retry unmounting with force option
// after given timeout.
func (mounter *Mounter) UnmountWithForce(target string, umountTimeout time.Duration) error {
	err := tryUnmount(target, mounter.withSafeNotMountedBehavior, umountTimeout)
	if err != nil {
		if err == context.DeadlineExceeded {
			klog.V(2).Infof("Timed out waiting for unmount of %s, trying with -f", target)
			err = forceUmount(target, mounter.withSafeNotMountedBehavior)
		}
		return err
	}
	return nil
}

// List returns a list of all mounted filesystems.
func (*Mounter) List() ([]MountPoint, error) {
	return ListProcMounts(procMountsPath)
}

func statx(file string) (unix.Statx_t, error) {
	var stat unix.Statx_t
	if err := unix.Statx(unix.AT_FDCWD, file, unix.AT_STATX_DONT_SYNC, 0, &stat); err != nil {
		if err == unix.ENOSYS {
			return stat, errStatxNotSupport
		}

		return stat, err
	}

	return stat, nil
}

func (mounter *Mounter) isLikelyNotMountPointStat(file string) (bool, error) {
	stat, err := os.Stat(file)
	if err != nil {
		return true, err
	}
	rootStat, err := os.Stat(filepath.Dir(strings.TrimSuffix(file, "/")))
	if err != nil {
		return true, err
	}
	// If the directory has a different device as parent, then it is a mountpoint.
	if stat.Sys().(*syscall.Stat_t).Dev != rootStat.Sys().(*syscall.Stat_t).Dev {
		return false, nil
	}

	return true, nil
}

func (mounter *Mounter) isLikelyNotMountPointStatx(file string) (bool, error) {
	var stat, rootStat unix.Statx_t
	var err error

	if stat, err = statx(file); err != nil {
		return true, err
	}

	if stat.Attributes_mask != 0 {
		if stat.Attributes_mask&unix.STATX_ATTR_MOUNT_ROOT != 0 {
			if stat.Attributes&unix.STATX_ATTR_MOUNT_ROOT != 0 {
				// file is a mountpoint
				return false, nil
			} else {
				// no need to check rootStat if unix.STATX_ATTR_MOUNT_ROOT supported
				return true, nil
			}
		}
	}

	root := filepath.Dir(strings.TrimSuffix(file, "/"))
	if rootStat, err = statx(root); err != nil {
		return true, err
	}

	return (stat.Dev_major == rootStat.Dev_major && stat.Dev_minor == rootStat.Dev_minor), nil
}

// IsLikelyNotMountPoint determines if a directory is not a mountpoint.
// It is fast but not necessarily ALWAYS correct. If the path is in fact
// a bind mount from one part of a mount to another it will not be detected.
// It also can not distinguish between mountpoints and symbolic links.
// mkdir /tmp/a /tmp/b; mount --bind /tmp/a /tmp/b; IsLikelyNotMountPoint("/tmp/b")
// will return true. When in fact /tmp/b is a mount point. If this situation
// is of interest to you, don't use this function...
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	notMountPoint, err := mounter.isLikelyNotMountPointStatx(file)
	if errors.Is(err, errStatxNotSupport) {
		// fall back to isLikelyNotMountPointStat
		return mounter.isLikelyNotMountPointStat(file)
	}

	return notMountPoint, err
}

// CanSafelySkipMountPointCheck relies on the detected behavior of umount when given a target that is not a mount point.
func (mounter *Mounter) CanSafelySkipMountPointCheck() bool {
	return mounter.withSafeNotMountedBehavior
}

// GetMountRefs finds all mount references to pathname, returns a
// list of paths. Path could be a mountpoint or a normal
// directory (for bind mount).
func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	pathExists, pathErr := PathExists(pathname)
	if !pathExists {
		return []string{}, nil
	} else if IsCorruptedMnt(pathErr) {
		klog.Warningf("GetMountRefs found corrupted mount at %s, treating as unmounted path", pathname)
		return []string{}, nil
	} else if pathErr != nil {
		return nil, fmt.Errorf("error checking path %s: %v", pathname, pathErr)
	}
	realpath, err := filepath.EvalSymlinks(pathname)
	if err != nil {
		return nil, err
	}
	return SearchMountPoints(realpath, procMountInfoPath)
}

// checkAndRepairFileSystem checks and repairs filesystems using command fsck.
func (mounter *SafeFormatAndMount) checkAndRepairFilesystem(source string) error {
	klog.V(4).Infof("Checking for issues with fsck on disk: %s", source)
	args := []string{"-a", source}
	out, err := mounter.Exec.Command("fsck", args...).CombinedOutput()
	if err != nil {
		ee, isExitError := err.(utilexec.ExitError)
		switch {
		case err == utilexec.ErrExecutableNotFound:
			klog.Warningf("'fsck' not found on system; continuing mount without running 'fsck'.")
		case isExitError && ee.ExitStatus() == fsckErrorsCorrected:
			klog.Infof("Device %s has errors which were corrected by fsck.", source)
		case isExitError && ee.ExitStatus() == fsckErrorsUncorrected:
			return NewMountError(HasFilesystemErrors, "'fsck' found errors on device %s but could not correct them: %s", source, string(out))
		case isExitError && ee.ExitStatus() > fsckErrorsUncorrected:
			klog.Infof("`fsck` error %s", string(out))
		default:
			klog.Warningf("fsck on device %s failed with error %v, output: %v", source, err, string(out))
		}
	}
	return nil
}

// formatAndMount uses unix utils to format and mount the given disk
func (mounter *SafeFormatAndMount) formatAndMountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string, formatOptions []string) error {
	readOnly := false
	for _, option := range options {
		if option == "ro" {
			readOnly = true
			break
		}
	}
	if !readOnly {
		// Check sensitiveOptions for ro
		for _, option := range sensitiveOptions {
			if option == "ro" {
				readOnly = true
				break
			}
		}
	}

	options = append(options, "defaults")
	mountErrorValue := UnknownMountError

	// Check if the disk is already formatted
	existingFormat, err := mounter.GetDiskFormat(source)
	if err != nil {
		return NewMountError(GetDiskFormatFailed, "failed to get disk format of disk %s: %v", source, err)
	}

	// Use 'ext4' as the default
	if len(fstype) == 0 {
		fstype = "ext4"
	}

	if existingFormat == "" {
		// Do not attempt to format the disk if mounting as readonly, return an error to reflect this.
		if readOnly {
			return NewMountError(UnformattedReadOnly, "cannot mount unformatted disk %s as we are manipulating it in read-only mode", source)
		}

		// Disk is unformatted so format it.
		args := []string{source}
		if fstype == "ext4" || fstype == "ext3" {
			args = []string{
				"-F",  // Force flag
				"-m0", // Zero blocks reserved for super-user
				source,
			}
		} else if fstype == "xfs" {
			args = []string{
				"-f", // force flag
				source,
			}
		}
		args = append(formatOptions, args...)

		klog.Infof("Disk %q appears to be unformatted, attempting to format as type: %q with options: %v", source, fstype, args)

		output, err := mounter.format(fstype, args)
		if err != nil {
			// Do not log sensitiveOptions only options
			sensitiveOptionsLog := sanitizedOptionsForLogging(options, sensitiveOptions)
			detailedErr := fmt.Sprintf("format of disk %q failed: type:(%q) target:(%q) options:(%q) errcode:(%v) output:(%v) ", source, fstype, target, sensitiveOptionsLog, err, string(output))
			klog.Error(detailedErr)
			return NewMountError(FormatFailed, "%s", detailedErr)
		}

		klog.Infof("Disk successfully formatted (mkfs): %s - %s %s", fstype, source, target)
	} else {
		if fstype != existingFormat {
			// Verify that the disk is formatted with filesystem type we are expecting
			mountErrorValue = FilesystemMismatch
			klog.Warningf("Configured to mount disk %s as %s but current format is %s, things might break", source, fstype, existingFormat)
		}

		if !readOnly {
			// Run check tools on the disk to fix repairable issues, only do this for formatted volumes requested as rw.
			err := mounter.checkAndRepairFilesystem(source)
			if err != nil {
				return err
			}
		}
	}

	// Mount the disk
	klog.V(4).Infof("Attempting to mount disk %s in %s format at %s", source, fstype, target)
	if err := mounter.MountSensitive(source, target, fstype, options, sensitiveOptions); err != nil {
		return NewMountError(mountErrorValue, "%s", err.Error())
	}

	return nil
}

func (mounter *SafeFormatAndMount) format(fstype string, args []string) ([]byte, error) {
	if mounter.formatSem != nil {
		done := make(chan struct{})
		defer close(done)

		mounter.formatSem <- struct{}{}

		go func() {
			defer func() { <-mounter.formatSem }()

			timeout := time.NewTimer(mounter.formatTimeout)
			defer timeout.Stop()

			select {
			case <-done:
			case <-timeout.C:
			}
		}()
	}

	return mounter.Exec.Command("mkfs."+fstype, args...).CombinedOutput()
}

func getDiskFormat(exec utilexec.Interface, disk string) (string, error) {
	args := []string{"-p", "-s", "TYPE", "-s", "PTTYPE", "-o", "export", disk}
	klog.V(4).Infof("Attempting to determine if disk %q is formatted using blkid with args: (%v)", disk, args)
	dataOut, err := exec.Command("blkid", args...).CombinedOutput()
	output := string(dataOut)
	klog.V(4).Infof("Output: %q", output)

	if err != nil {
		if exit, ok := err.(utilexec.ExitError); ok {
			if exit.ExitStatus() == 2 {
				// Disk device is unformatted.
				// For `blkid`, if the specified token (TYPE/PTTYPE, etc) was
				// not found, or no (specified) devices could be identified, an
				// exit code of 2 is returned.
				return "", nil
			}
		}
		klog.Errorf("Could not determine if disk %q is formatted (%v)", disk, err)
		return "", err
	}

	var fstype, pttype string

	lines := strings.Split(output, "\n")
	for _, l := range lines {
		if len(l) <= 0 {
			// Ignore empty line.
			continue
		}
		cs := strings.Split(l, "=")
		if len(cs) != 2 {
			return "", fmt.Errorf("blkid returns invalid output: %s", output)
		}
		// TYPE is filesystem type, and PTTYPE is partition table type, according
		// to https://www.kernel.org/pub/linux/utils/util-linux/v2.21/libblkid-docs/.
		if cs[0] == "TYPE" {
			fstype = cs[1]
		} else if cs[0] == "PTTYPE" {
			pttype = cs[1]
		}
	}

	if len(pttype) > 0 {
		klog.V(4).Infof("Disk %s detected partition table type: %s", disk, pttype)
		// Returns a special non-empty string as filesystem type, then kubelet
		// will not format it.
		return "unknown data, probably partitions", nil
	}

	return fstype, nil
}

// GetDiskFormat uses 'blkid' to see if the given disk is unformatted
func (mounter *SafeFormatAndMount) GetDiskFormat(disk string) (string, error) {
	return getDiskFormat(mounter.Exec, disk)
}

// ListProcMounts returns a list of all mounted filesystems.
func ListProcMounts(mountFilePath string) ([]MountPoint, error) {
	content, err := readMountInfo(mountFilePath)
	if err != nil {
		return nil, err
	}
	return parseProcMounts(content)
}

func parseProcMounts(content []byte) ([]MountPoint, error) {
	out := []MountPoint{}
	lines := strings.Split(string(content), "\n")
	for _, line := range lines {
		if line == "" {
			// the last split() item is empty string following the last \n
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != expectedNumFieldsPerLine {
			// Do not log line in case it contains sensitive Mount options
			return nil, fmt.Errorf("wrong number of fields (expected %d, got %d)", expectedNumFieldsPerLine, len(fields))
		}

		mp := MountPoint{
			Device: fields[0],
			Path:   fields[1],
			Type:   fields[2],
			Opts:   strings.Split(fields[3], ","),
		}

		freq, err := strconv.Atoi(fields[4])
		if err != nil {
			return nil, err
		}
		mp.Freq = freq

		pass, err := strconv.Atoi(fields[5])
		if err != nil {
			return nil, err
		}
		mp.Pass = pass

		out = append(out, mp)
	}
	return out, nil
}

// SearchMountPoints finds all mount references to the source, returns a list of
// mountpoints.
// The source can be a mount point or a normal directory (bind mount). We
// didn't support device because there is no use case by now.
// Some filesystems may share a source name, e.g. tmpfs. And for bind mounting,
// it's possible to mount a non-root path of a filesystem, so we need to use
// root path and major:minor to represent mount source uniquely.
func SearchMountPoints(hostSource, mountInfoPath string) ([]string, error) {
	mis, err := ParseMountInfo(mountInfoPath)
	if err != nil {
		return nil, err
	}

	mountID := 0
	rootPath := ""
	major := -1
	minor := -1

	// Finding the underlying root path and major:minor if possible.
	// We need search in backward order because it's possible for later mounts
	// to overlap earlier mounts.
	for i := len(mis) - 1; i >= 0; i-- {
		if hostSource == mis[i].MountPoint || PathWithinBase(hostSource, mis[i].MountPoint) {
			// If it's a mount point or path under a mount point.
			mountID = mis[i].ID
			rootPath = filepath.Join(mis[i].Root, strings.TrimPrefix(hostSource, mis[i].MountPoint))
			major = mis[i].Major
			minor = mis[i].Minor
			break
		}
	}

	if rootPath == "" || major == -1 || minor == -1 {
		return nil, fmt.Errorf("failed to get root path and major:minor for %s", hostSource)
	}

	var refs []string
	for i := range mis {
		if mis[i].ID == mountID {
			// Ignore mount entry for mount source itself.
			continue
		}
		if mis[i].Root == rootPath && mis[i].Major == major && mis[i].Minor == minor {
			refs = append(refs, mis[i].MountPoint)
		}
	}

	return refs, nil
}

// IsMountPoint determines if a file is a mountpoint.
// It first detects bind & any other mountpoints using
// MountedFast function. If the MountedFast function returns
// sure as true and err as nil, then a mountpoint is detected
// successfully. When an error is returned by MountedFast, the
// following is true:
// 1. All errors are returned with IsMountPoint as false
// except os.IsPermission.
// 2. When os.IsPermission is returned by MountedFast, List()
// is called to confirm if the given file is a mountpoint are not.
//
// os.ErrNotExist should always be returned if a file does not exist
// as callers have in past relied on this error and not fallback.
//
// When MountedFast returns sure as false and err as nil (eg: in
// case of bindmounts on kernel version 5.10- ); mounter.List()
// endpoint is called to enumerate all the mountpoints and check if
// it is mountpoint match or not.
func (mounter *Mounter) IsMountPoint(file string) (bool, error) {
	isMnt, sure, isMntErr := mountinfo.MountedFast(file)
	if sure && isMntErr == nil {
		return isMnt, nil
	}
	if isMntErr != nil {
		if errors.Is(isMntErr, fs.ErrNotExist) {
			return false, fs.ErrNotExist
		}
		// We were not allowed to do the simple stat() check, e.g. on NFS with
		// root_squash. Fall back to /proc/mounts check below when
		// fs.ErrPermission is returned.
		if !errors.Is(isMntErr, fs.ErrPermission) {
			return false, isMntErr
		}
	}
	// Resolve any symlinks in file, kernel would do the same and use the resolved path in /proc/mounts.
	resolvedFile, err := filepath.EvalSymlinks(file)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, fs.ErrNotExist
		}
		return false, err
	}

	// check all mountpoints since MountedFast is not sure.
	// is not reliable for some mountpoint types.
	mountPoints, mountPointsErr := mounter.List()
	if mountPointsErr != nil {
		return false, mountPointsErr
	}
	for _, mp := range mountPoints {
		if isMountPointMatch(mp, resolvedFile) {
			return true, nil
		}
	}
	return false, nil
}

// tryUnmount calls plain "umount" and waits for unmountTimeout for it to finish.
func tryUnmount(target string, withSafeNotMountedBehavior bool, unmountTimeout time.Duration) error {
	klog.V(4).Infof("Unmounting %s", target)
	ctx, cancel := context.WithTimeout(context.Background(), unmountTimeout)
	defer cancel()

	command := exec.CommandContext(ctx, "umount", target)
	output, err := command.CombinedOutput()

	// CombinedOutput() does not return DeadlineExceeded, make sure it's
	// propagated on timeout.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	if err != nil {
		return checkUmountError(target, command, output, err, withSafeNotMountedBehavior)
	}
	return nil
}

func forceUmount(target string, withSafeNotMountedBehavior bool) error {
	command := exec.Command("umount", "-f", target)
	output, err := command.CombinedOutput()
	if err != nil {
		return checkUmountError(target, command, output, err, withSafeNotMountedBehavior)
	}
	return nil
}

// checkUmountError checks a result of umount command and determine a return value.
func checkUmountError(target string, command *exec.Cmd, output []byte, err error, withSafeNotMountedBehavior bool) error {
	if err.Error() == errNoChildProcesses {
		if command.ProcessState.Success() {
			// We don't consider errNoChildProcesses an error if the process itself succeeded (see - k/k issue #103753).
			return nil
		}
		// Rewrite err with the actual exit error of the process.
		err = &exec.ExitError{ProcessState: command.ProcessState}
	}
	if withSafeNotMountedBehavior && strings.Contains(string(output), errNotMounted) {
		klog.V(4).Infof("ignoring 'not mounted' error for %s", target)
		return nil
	}
	return fmt.Errorf("unmount failed: %v\nUnmounting arguments: %s\nOutput: %s", err, target, string(output))
}
