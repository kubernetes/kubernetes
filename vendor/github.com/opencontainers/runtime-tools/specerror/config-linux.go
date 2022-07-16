package specerror

import (
	"fmt"

	rfc2119 "github.com/opencontainers/runtime-tools/error"
)

// define error codes
const (
	// DefaultFilesystems represents "The following filesystems SHOULD be made available in each container's filesystem:"
	DefaultFilesystems Code = 0xc001 + iota
	// NSPathAbs represents "This value MUST be an absolute path in the runtime mount namespace."
	NSPathAbs
	// NSProcInPath represents "The runtime MUST place the container process in the namespace associated with that `path`."
	NSProcInPath
	// NSPathMatchTypeError represents "The runtime MUST generate an error if `path` is not associated with a namespace of type `type`."
	NSPathMatchTypeError
	// NSNewNSWithoutPath represents "If `path` is not specified, the runtime MUST create a new container namespace of type `type`."
	NSNewNSWithoutPath
	// NSInheritWithoutType represents "If a namespace type is not specified in the `namespaces` array, the container MUST inherit the runtime namespace of that type."
	NSInheritWithoutType
	// NSErrorOnDup represents "If a `namespaces` field contains duplicated namespaces with same `type`, the runtime MUST generate an error."
	NSErrorOnDup
	// UserNSMapOwnershipRO represents "The runtime SHOULD NOT modify the ownership of referenced filesystems to realize the mapping."
	UserNSMapOwnershipRO
	// DevicesAvailable represents "devices (array of objects, OPTIONAL) lists devices that MUST be available in the container."
	DevicesAvailable
	// DevicesFileNotMatch represents "If a file already exists at `path` that does not match the requested device, the runtime MUST generate an error."
	DevicesFileNotMatch
	// DevicesMajMinRequired represents "`major, minor` (int64, REQUIRED unless `type` is `p`) - major, minor numbers for the device."
	DevicesMajMinRequired
	// DevicesErrorOnDup represents "The same `type`, `major` and `minor` SHOULD NOT be used for multiple devices."
	DevicesErrorOnDup
	// DefaultDevices represents "In addition to any devices configured with this setting, the runtime MUST also supply default devices."
	DefaultDevices
	// CgroupsPathAbsOrRel represents "The value of `cgroupsPath` MUST be either an absolute path or a relative path."
	CgroupsPathAbsOrRel
	// CgroupsAbsPathRelToMount represents "In the case of an absolute path (starting with `/`), the runtime MUST take the path to be relative to the cgroups mount point."
	CgroupsAbsPathRelToMount
	// CgroupsPathAttach represents "If the value is specified, the runtime MUST consistently attach to the same place in the cgroups hierarchy given the same value of `cgroupsPath`."
	CgroupsPathAttach
	// CgroupsPathError represents "Runtimes MAY consider certain `cgroupsPath` values to be invalid, and MUST generate an error if this is the case."
	CgroupsPathError
	// DevicesApplyInOrder represents "The runtime MUST apply entries in the listed order."
	DevicesApplyInOrder
	// BlkIOWeightOrLeafWeightExist represents "You MUST specify at least one of `weight` or `leafWeight` in a given entry, and MAY specify both."
	BlkIOWeightOrLeafWeightExist
	// IntelRdtPIDWrite represents "If `intelRdt` is set, the runtime MUST write the container process ID to the `<container-id>/tasks` file in a mounted `resctrl` pseudo-filesystem, using the container ID from `start` and creating the `container-id` directory if necessary."
	IntelRdtPIDWrite
	// IntelRdtNoMountedResctrlError represents "If no mounted `resctrl` pseudo-filesystem is available in the runtime mount namespace, the runtime MUST generate an error."
	IntelRdtNoMountedResctrlError
	// NotManipResctrlWithoutIntelRdt represents "If `intelRdt` is not set, the runtime MUST NOT manipulate any `resctrl` pseudo-filesystems."
	NotManipResctrlWithoutIntelRdt
	// IntelRdtL3CacheSchemaWrite represents "If `l3CacheSchema` is set, runtimes MUST write the value to the `schemata` file in the `<container-id>` directory discussed in `intelRdt`."
	IntelRdtL3CacheSchemaWrite
	// IntelRdtL3CacheSchemaNotWrite represents "If `l3CacheSchema` is not set, runtimes MUST NOT write to `schemata` files in any `resctrl` pseudo-filesystems."
	IntelRdtL3CacheSchemaNotWrite
	// SeccSyscallsNamesRequired represents "`names` MUST contain at least one entry."
	SeccSyscallsNamesRequired
	// MaskedPathsAbs represents "maskedPaths (array of strings, OPTIONAL) will mask over the provided paths inside the container so that they cannot be read. The values MUST be absolute paths in the container namespace."
	MaskedPathsAbs
	// ReadonlyPathsAbs represents "readonlyPaths (array of strings, OPTIONAL) will set the provided paths as readonly inside the container. The values MUST be absolute paths in the container namespace."
	ReadonlyPathsAbs
)

var (
	defaultFilesystemsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#default-filesystems"), nil
	}
	namespacesRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#namespaces"), nil
	}
	userNamespaceMappingsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#user-namespace-mappings"), nil
	}
	devicesRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#devices"), nil
	}
	defaultDevicesRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#default-devices"), nil
	}
	cgroupsPathRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#cgroups-path"), nil
	}
	deviceWhitelistRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#device-whitelist"), nil
	}
	blockIoRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#block-io"), nil
	}
	intelrdtRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#intelrdt"), nil
	}
	seccompRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#seccomp"), nil
	}
	maskedPathsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#masked-paths"), nil
	}
	readonlyPathsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config-linux.md#readonly-paths"), nil
	}
)

func init() {
	register(DefaultFilesystems, rfc2119.Should, defaultFilesystemsRef)
	register(NSPathAbs, rfc2119.Must, namespacesRef)
	register(NSProcInPath, rfc2119.Must, namespacesRef)
	register(NSPathMatchTypeError, rfc2119.Must, namespacesRef)
	register(NSNewNSWithoutPath, rfc2119.Must, namespacesRef)
	register(NSInheritWithoutType, rfc2119.Must, namespacesRef)
	register(NSErrorOnDup, rfc2119.Must, namespacesRef)
	register(UserNSMapOwnershipRO, rfc2119.Should, userNamespaceMappingsRef)
	register(DevicesAvailable, rfc2119.Must, devicesRef)
	register(DevicesFileNotMatch, rfc2119.Must, devicesRef)
	register(DevicesMajMinRequired, rfc2119.Required, devicesRef)
	register(DevicesErrorOnDup, rfc2119.Should, devicesRef)
	register(DefaultDevices, rfc2119.Must, defaultDevicesRef)
	register(CgroupsPathAbsOrRel, rfc2119.Must, cgroupsPathRef)
	register(CgroupsAbsPathRelToMount, rfc2119.Must, cgroupsPathRef)
	register(CgroupsPathAttach, rfc2119.Must, cgroupsPathRef)
	register(CgroupsPathError, rfc2119.Must, cgroupsPathRef)
	register(DevicesApplyInOrder, rfc2119.Must, deviceWhitelistRef)
	register(BlkIOWeightOrLeafWeightExist, rfc2119.Must, blockIoRef)
	register(IntelRdtPIDWrite, rfc2119.Must, intelrdtRef)
	register(IntelRdtNoMountedResctrlError, rfc2119.Must, intelrdtRef)
	register(NotManipResctrlWithoutIntelRdt, rfc2119.Must, intelrdtRef)
	register(IntelRdtL3CacheSchemaWrite, rfc2119.Must, intelrdtRef)
	register(IntelRdtL3CacheSchemaNotWrite, rfc2119.Must, intelrdtRef)
	register(SeccSyscallsNamesRequired, rfc2119.Must, seccompRef)
	register(MaskedPathsAbs, rfc2119.Must, maskedPathsRef)
	register(ReadonlyPathsAbs, rfc2119.Must, readonlyPathsRef)
}
