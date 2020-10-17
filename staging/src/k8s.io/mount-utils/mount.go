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

// TODO(thockin): This whole pkg is pretty linux-centric.  As soon as we have
// an alternate platform, we will need to abstract further.

package mount

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	utilexec "k8s.io/utils/exec"
)

const (
	// Default mount command if mounter path is not specified.
	defaultMountCommand = "mount"
	// Log message where sensitive mount options were removed
	sensitiveOptionsRemoved = "<masked>"
)

// Interface defines the set of methods to allow for mount operations on a system.
type Interface interface {
	// Mount mounts source to target as fstype with given options.
	// options MUST not contain sensitive material (like passwords).
	Mount(source string, target string, fstype string, options []string) error
	// MountSensitive is the same as Mount() but this method allows
	// sensitiveOptions to be passed in a separate parameter from the normal
	// mount options and ensures the sensitiveOptions are never logged. This
	// method should be used by callers that pass sensitive material (like
	// passwords) as mount options.
	MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error
	// MountSensitiveWithoutSystemd is the same as MountSensitive() but this method disable using systemd mount.
	MountSensitiveWithoutSystemd(source string, target string, fstype string, options []string, sensitiveOptions []string) error
	// Unmount unmounts given target.
	Unmount(target string) error
	// List returns a list of all mounted filesystems.  This can be large.
	// On some platforms, reading mounts directly from the OS is not guaranteed
	// consistent (i.e. it could change between chunked reads). This is guaranteed
	// to be consistent.
	List() ([]MountPoint, error)
	// IsLikelyNotMountPoint uses heuristics to determine if a directory
	// is not a mountpoint.
	// It should return ErrNotExist when the directory does not exist.
	// IsLikelyNotMountPoint does NOT properly detect all mountpoint types
	// most notably linux bind mounts and symbolic link. For callers that do not
	// care about such situations, this is a faster alternative to calling List()
	// and scanning that output.
	IsLikelyNotMountPoint(file string) (bool, error)
	// GetMountRefs finds all mount references to pathname, returning a slice of
	// paths. Pathname can be a mountpoint path or a normal	directory
	// (for bind mount). On Linux, pathname is excluded from the slice.
	// For example, if /dev/sdc was mounted at /path/a and /path/b,
	// GetMountRefs("/path/a") would return ["/path/b"]
	// GetMountRefs("/path/b") would return ["/path/a"]
	// On Windows there is no way to query all mount points; as long as pathname is
	// a valid mount, it will be returned.
	GetMountRefs(pathname string) ([]string, error)
}

// Compile-time check to ensure all Mounter implementations satisfy
// the mount interface.
var _ Interface = &Mounter{}

// MountPoint represents a single line in /proc/mounts or /etc/fstab.
type MountPoint struct { // nolint: golint
	Device string
	Path   string
	Type   string
	Opts   []string // Opts may contain sensitive mount options (like passwords) and MUST be treated as such (e.g. not logged).
	Freq   int
	Pass   int
}

type MountErrorType string // nolint: golint

const (
	FilesystemMismatch  MountErrorType = "FilesystemMismatch"
	HasFilesystemErrors MountErrorType = "HasFilesystemErrors"
	UnformattedReadOnly MountErrorType = "UnformattedReadOnly"
	FormatFailed        MountErrorType = "FormatFailed"
	GetDiskFormatFailed MountErrorType = "GetDiskFormatFailed"
	UnknownMountError   MountErrorType = "UnknownMountError"
)

type MountError struct { // nolint: golint
	Type    MountErrorType
	Message string
}

func (mountError MountError) String() string {
	return mountError.Message
}

func (mountError MountError) Error() string {
	return mountError.Message
}

func NewMountError(mountErrorValue MountErrorType, format string, args ...interface{}) error {
	mountError := MountError{
		Type:    mountErrorValue,
		Message: fmt.Sprintf(format, args...),
	}
	return mountError
}

// SafeFormatAndMount probes a device to see if it is formatted.
// Namely it checks to see if a file system is present. If so it
// mounts it otherwise the device is formatted first then mounted.
type SafeFormatAndMount struct {
	Interface
	Exec utilexec.Interface
}

// FormatAndMount formats the given disk, if needed, and mounts it.
// That is if the disk is not formatted and it is not being mounted as
// read-only it will format it first then mount it. Otherwise, if the
// disk is already formatted or it is being mounted as read-only, it
// will be mounted without formatting.
// options MUST not contain sensitive material (like passwords).
func (mounter *SafeFormatAndMount) FormatAndMount(source string, target string, fstype string, options []string) error {
	return mounter.FormatAndMountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
}

// FormatAndMountSensitive is the same as FormatAndMount but this method allows
// sensitiveOptions to be passed in a separate parameter from the normal mount
// options and ensures the sensitiveOptions are never logged. This method should
// be used by callers that pass sensitive material (like passwords) as mount
// options.
func (mounter *SafeFormatAndMount) FormatAndMountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return mounter.formatAndMountSensitive(source, target, fstype, options, sensitiveOptions)
}

// getMountRefsByDev finds all references to the device provided
// by mountPath; returns a list of paths.
// Note that mountPath should be path after the evaluation of any symblolic links.
func getMountRefsByDev(mounter Interface, mountPath string) ([]string, error) {
	mps, err := mounter.List()
	if err != nil {
		return nil, err
	}

	// Finding the device mounted to mountPath.
	diskDev := ""
	for i := range mps {
		if mountPath == mps[i].Path {
			diskDev = mps[i].Device
			break
		}
	}

	// Find all references to the device.
	var refs []string
	for i := range mps {
		if mps[i].Device == diskDev || mps[i].Device == mountPath {
			if mps[i].Path != mountPath {
				refs = append(refs, mps[i].Path)
			}
		}
	}
	return refs, nil
}

// GetDeviceNameFromMount given a mnt point, find the device from /proc/mounts
// returns the device name, reference count, and error code.
func GetDeviceNameFromMount(mounter Interface, mountPath string) (string, int, error) {
	mps, err := mounter.List()
	if err != nil {
		return "", 0, err
	}

	// Find the device name.
	// FIXME if multiple devices mounted on the same mount path, only the first one is returned.
	device := ""
	// If mountPath is symlink, need get its target path.
	slTarget, err := filepath.EvalSymlinks(mountPath)
	if err != nil {
		slTarget = mountPath
	}
	for i := range mps {
		if mps[i].Path == slTarget {
			device = mps[i].Device
			break
		}
	}

	// Find all references to the device.
	refCount := 0
	for i := range mps {
		if mps[i].Device == device {
			refCount++
		}
	}
	return device, refCount, nil
}

// IsNotMountPoint determines if a directory is a mountpoint.
// It should return ErrNotExist when the directory does not exist.
// IsNotMountPoint is more expensive than IsLikelyNotMountPoint.
// IsNotMountPoint detects bind mounts in linux.
// IsNotMountPoint enumerates all the mountpoints using List() and
// the list of mountpoints may be large, then it uses
// isMountPointMatch to evaluate whether the directory is a mountpoint.
func IsNotMountPoint(mounter Interface, file string) (bool, error) {
	// IsLikelyNotMountPoint provides a quick check
	// to determine whether file IS A mountpoint.
	notMnt, notMntErr := mounter.IsLikelyNotMountPoint(file)
	if notMntErr != nil && os.IsPermission(notMntErr) {
		// We were not allowed to do the simple stat() check, e.g. on NFS with
		// root_squash. Fall back to /proc/mounts check below.
		notMnt = true
		notMntErr = nil
	}
	if notMntErr != nil {
		return notMnt, notMntErr
	}
	// identified as mountpoint, so return this fact.
	if notMnt == false {
		return notMnt, nil
	}

	// Resolve any symlinks in file, kernel would do the same and use the resolved path in /proc/mounts.
	resolvedFile, err := filepath.EvalSymlinks(file)
	if err != nil {
		return true, err
	}

	// check all mountpoints since IsLikelyNotMountPoint
	// is not reliable for some mountpoint types.
	mountPoints, mountPointsErr := mounter.List()
	if mountPointsErr != nil {
		return notMnt, mountPointsErr
	}
	for _, mp := range mountPoints {
		if isMountPointMatch(mp, resolvedFile) {
			notMnt = false
			break
		}
	}
	return notMnt, nil
}

// MakeBindOpts detects whether a bind mount is being requested and makes the remount options to
// use in case of bind mount, due to the fact that bind mount doesn't respect mount options.
// The list equals:
//   options - 'bind' + 'remount' (no duplicate)
func MakeBindOpts(options []string) (bool, []string, []string) {
	bind, bindOpts, bindRemountOpts, _ := MakeBindOptsSensitive(options, nil /* sensitiveOptions */)
	return bind, bindOpts, bindRemountOpts
}

// MakeBindOptsSensitive is the same as MakeBindOpts but this method allows
// sensitiveOptions to be passed in a separate parameter from the normal mount
// options and ensures the sensitiveOptions are never logged. This method should
// be used by callers that pass sensitive material (like passwords) as mount
// options.
func MakeBindOptsSensitive(options []string, sensitiveOptions []string) (bool, []string, []string, []string) {
	// Because we have an FD opened on the subpath bind mount, the "bind" option
	// needs to be included, otherwise the mount target will error as busy if you
	// remount as readonly.
	//
	// As a consequence, all read only bind mounts will no longer change the underlying
	// volume mount to be read only.
	bindRemountOpts := []string{"bind", "remount"}
	bindRemountSensitiveOpts := []string{}
	bind := false
	bindOpts := []string{"bind"}

	// _netdev is a userspace mount option and does not automatically get added when
	// bind mount is created and hence we must carry it over.
	if checkForNetDev(options, sensitiveOptions) {
		bindOpts = append(bindOpts, "_netdev")
	}

	for _, option := range options {
		switch option {
		case "bind":
			bind = true
		case "remount":
		default:
			bindRemountOpts = append(bindRemountOpts, option)
		}
	}

	for _, sensitiveOption := range sensitiveOptions {
		switch sensitiveOption {
		case "bind":
			bind = true
		case "remount":
		default:
			bindRemountSensitiveOpts = append(bindRemountSensitiveOpts, sensitiveOption)
		}
	}

	return bind, bindOpts, bindRemountOpts, bindRemountSensitiveOpts
}

func checkForNetDev(options []string, sensitiveOptions []string) bool {
	for _, option := range options {
		if option == "_netdev" {
			return true
		}
	}
	for _, sensitiveOption := range sensitiveOptions {
		if sensitiveOption == "_netdev" {
			return true
		}
	}
	return false
}

// PathWithinBase checks if give path is within given base directory.
func PathWithinBase(fullPath, basePath string) bool {
	rel, err := filepath.Rel(basePath, fullPath)
	if err != nil {
		return false
	}
	if StartsWithBackstep(rel) {
		// Needed to escape the base path.
		return false
	}
	return true
}

// StartsWithBackstep checks if the given path starts with a backstep segment.
func StartsWithBackstep(rel string) bool {
	// normalize to / and check for ../
	return rel == ".." || strings.HasPrefix(filepath.ToSlash(rel), "../")
}

// sanitizedOptionsForLogging will return a comma separated string containing
// options and sensitiveOptions. Each entry in sensitiveOptions will be
// replaced with the string sensitiveOptionsRemoved
// e.g. o1,o2,<masked>,<masked>
func sanitizedOptionsForLogging(options []string, sensitiveOptions []string) string {
	separator := ""
	if len(options) > 0 && len(sensitiveOptions) > 0 {
		separator = ","
	}

	sensitiveOptionsStart := ""
	sensitiveOptionsEnd := ""
	if len(sensitiveOptions) > 0 {
		sensitiveOptionsStart = strings.Repeat(sensitiveOptionsRemoved+",", len(sensitiveOptions)-1)
		sensitiveOptionsEnd = sensitiveOptionsRemoved
	}

	return strings.Join(options, ",") +
		separator +
		sensitiveOptionsStart +
		sensitiveOptionsEnd
}
