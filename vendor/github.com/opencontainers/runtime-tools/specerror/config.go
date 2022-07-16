package specerror

import (
	"fmt"

	rfc2119 "github.com/opencontainers/runtime-tools/error"
)

// define error codes
const (
	// SpecVersionInSemVer represents "`ociVersion` (string, REQUIRED) MUST be in SemVer v2.0.0 format and specifies the version of the Open Container Initiative Runtime Specification with which the bundle complies."
	SpecVersionInSemVer Code = 0xb001 + iota
	// RootOnWindowsRequired represents "On Windows, for Windows Server Containers, this field is REQUIRED."
	RootOnWindowsRequired
	// RootOnHyperVNotSet represents "For Hyper-V Containers, this field MUST NOT be set."
	RootOnHyperVNotSet
	// RootOnNonWindowsRequired represents "On all other platforms, this field is REQUIRED."
	RootOnNonWindowsRequired
	// RootPathOnWindowsGUID represents "On Windows, `path` MUST be a volume GUID path."
	RootPathOnWindowsGUID
	// RootPathOnPosixConvention represents "The value SHOULD be the conventional `rootfs`."
	RootPathOnPosixConvention
	// RootPathExist represents "A directory MUST exist at the path declared by the field."
	RootPathExist
	// RootReadonlyImplement represents "`readonly` (bool, OPTIONAL) If true then the root filesystem MUST be read-only inside the container, defaults to false."
	RootReadonlyImplement
	// RootReadonlyOnWindowsFalse represents "* On Windows, this field MUST be omitted or false."
	RootReadonlyOnWindowsFalse
	// MountsInOrder represents "The runtime MUST mount entries in the listed order."
	MountsInOrder
	// MountsDestAbs represents "Destination of mount point: path inside container. This value MUST be an absolute path."
	MountsDestAbs
	// MountsDestOnWindowsNotNested represents "Windows: one mount destination MUST NOT be nested within another mount (e.g., c:\\foo and c:\\foo\\bar)."
	MountsDestOnWindowsNotNested
	// MountsOptionsOnWindowsROSupport represents "Windows: runtimes MUST support `ro`, mounting the filesystem read-only when `ro` is given."
	MountsOptionsOnWindowsROSupport
	// ProcRequiredAtStart represents "This property is REQUIRED when `start` is called."
	ProcRequiredAtStart
	// ProcConsoleSizeIgnore represents "Runtimes MUST ignore `consoleSize` if `terminal` is `false` or unset."
	ProcConsoleSizeIgnore
	// ProcCwdAbs represents "cwd (string, REQUIRED) is the working directory that will be set for the executable. This value MUST be an absolute path."
	ProcCwdAbs
	// ProcArgsOneEntryRequired represents "This specification extends the IEEE standard in that at least one entry is REQUIRED, and that entry is used with the same semantics as `execvp`'s *file*."
	ProcArgsOneEntryRequired
	// PosixProcRlimitsTypeGenError represents "The runtime MUST generate an error for any values which cannot be mapped to a relevant kernel interface."
	PosixProcRlimitsTypeGenError
	// PosixProcRlimitsTypeGet represents "For each entry in `rlimits`, a `getrlimit(3)` on `type` MUST succeed."
	PosixProcRlimitsTypeGet
	// PosixProcRlimitsTypeValueError represents "valid values are defined in the ... man page"
	PosixProcRlimitsTypeValueError
	// PosixProcRlimitsSoftMatchCur represents "`rlim.rlim_cur` MUST match the configured value."
	PosixProcRlimitsSoftMatchCur
	// PosixProcRlimitsHardMatchMax represents "`rlim.rlim_max` MUST match the configured value."
	PosixProcRlimitsHardMatchMax
	// PosixProcRlimitsErrorOnDup represents "If `rlimits` contains duplicated entries with same `type`, the runtime MUST generate an error."
	PosixProcRlimitsErrorOnDup
	// LinuxProcCapError represents "Any value which cannot be mapped to a relevant kernel interface MUST cause an error."
	LinuxProcCapError
	// LinuxProcOomScoreAdjSet represents "If `oomScoreAdj` is set, the runtime MUST set `oom_score_adj` to the given value."
	LinuxProcOomScoreAdjSet
	// LinuxProcOomScoreAdjNotSet represents "If `oomScoreAdj` is not set, the runtime MUST NOT change the value of `oom_score_adj`."
	LinuxProcOomScoreAdjNotSet
	// PlatformSpecConfOnWindowsSet represents "This MUST be set if the target platform of this spec is `windows`."
	PlatformSpecConfOnWindowsSet
	// PosixHooksPathAbs represents "This specification extends the IEEE standard in that `path` MUST be absolute."
	PosixHooksPathAbs
	// PosixHooksTimeoutPositive represents "If set, `timeout` MUST be greater than zero."
	PosixHooksTimeoutPositive
	// PosixHooksCalledInOrder represents "Hooks MUST be called in the listed order."
	PosixHooksCalledInOrder
	// PosixHooksStateToStdin represents "The state of the container MUST be passed to hooks over stdin so that they may do work appropriate to the current state of the container."
	PosixHooksStateToStdin
	// PrestartTiming represents "The pre-start hooks MUST be called after the `start` operation is called but before the user-specified program command is executed."
	PrestartTiming
	// PoststartTiming represents "The post-start hooks MUST be called after the user-specified process is executed but before the `start` operation returns."
	PoststartTiming
	// PoststopTiming represents "The post-stop hooks MUST be called after the container is deleted but before the `delete` operation returns."
	PoststopTiming
	// AnnotationsKeyValueMap represents "Annotations MUST be a key-value map."
	AnnotationsKeyValueMap
	// AnnotationsKeyString represents "Keys MUST be strings."
	AnnotationsKeyString
	// AnnotationsKeyRequired represents "Keys MUST NOT be an empty string."
	AnnotationsKeyRequired
	// AnnotationsKeyReversedDomain represents "Keys SHOULD be named using a reverse domain notation - e.g. `com.example.myKey`."
	AnnotationsKeyReversedDomain
	// AnnotationsKeyReservedNS represents "Keys using the `org.opencontainers` namespace are reserved and MUST NOT be used by subsequent specifications."
	AnnotationsKeyReservedNS
	// AnnotationsKeyIgnoreUnknown represents "Implementations that are reading/processing this configuration file MUST NOT generate an error if they encounter an unknown annotation key."
	AnnotationsKeyIgnoreUnknown
	// AnnotationsValueString represents "Values MUST be strings."
	AnnotationsValueString
	// ExtensibilityIgnoreUnknownProp represents "Runtimes that are reading or processing this configuration file MUST NOT generate an error if they encounter an unknown property."
	ExtensibilityIgnoreUnknownProp
	// ValidValues represents "Runtimes that are reading or processing this configuration file MUST generate an error when invalid or unsupported values are encountered."
	ValidValues
)

var (
	specificationVersionRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#specification-version"), nil
	}
	rootRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#root"), nil
	}
	mountsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#mounts"), nil
	}
	processRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#process"), nil
	}
	posixProcessRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#posix-process"), nil
	}
	linuxProcessRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#linux-process"), nil
	}
	platformSpecificConfigurationRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#platform-specific-configuration"), nil
	}
	posixPlatformHooksRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#posix-platform-hooks"), nil
	}
	prestartRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#prestart"), nil
	}
	poststartRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#poststart"), nil
	}
	poststopRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#poststop"), nil
	}
	annotationsRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#annotations"), nil
	}
	extensibilityRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#extensibility"), nil
	}
	validValuesRef = func(version string) (reference string, err error) {
		return fmt.Sprintf(referenceTemplate, version, "config.md#valid-values"), nil
	}
)

func init() {
	register(SpecVersionInSemVer, rfc2119.Must, specificationVersionRef)
	register(RootOnWindowsRequired, rfc2119.Required, rootRef)
	register(RootOnHyperVNotSet, rfc2119.Must, rootRef)
	register(RootOnNonWindowsRequired, rfc2119.Required, rootRef)
	register(RootPathOnWindowsGUID, rfc2119.Must, rootRef)
	register(RootPathOnPosixConvention, rfc2119.Should, rootRef)
	register(RootPathExist, rfc2119.Must, rootRef)
	register(RootReadonlyImplement, rfc2119.Must, rootRef)
	register(RootReadonlyOnWindowsFalse, rfc2119.Must, rootRef)
	register(MountsInOrder, rfc2119.Must, mountsRef)
	register(MountsDestAbs, rfc2119.Must, mountsRef)
	register(MountsDestOnWindowsNotNested, rfc2119.Must, mountsRef)
	register(MountsOptionsOnWindowsROSupport, rfc2119.Must, mountsRef)
	register(ProcRequiredAtStart, rfc2119.Required, processRef)
	register(ProcConsoleSizeIgnore, rfc2119.Must, processRef)
	register(ProcCwdAbs, rfc2119.Must, processRef)
	register(ProcArgsOneEntryRequired, rfc2119.Required, processRef)
	register(PosixProcRlimitsTypeGenError, rfc2119.Must, posixProcessRef)
	register(PosixProcRlimitsTypeGet, rfc2119.Must, posixProcessRef)
	register(PosixProcRlimitsTypeValueError, rfc2119.Should, posixProcessRef)
	register(PosixProcRlimitsSoftMatchCur, rfc2119.Must, posixProcessRef)
	register(PosixProcRlimitsHardMatchMax, rfc2119.Must, posixProcessRef)
	register(PosixProcRlimitsErrorOnDup, rfc2119.Must, posixProcessRef)
	register(LinuxProcCapError, rfc2119.Must, linuxProcessRef)
	register(LinuxProcOomScoreAdjSet, rfc2119.Must, linuxProcessRef)
	register(LinuxProcOomScoreAdjNotSet, rfc2119.Must, linuxProcessRef)
	register(PlatformSpecConfOnWindowsSet, rfc2119.Must, platformSpecificConfigurationRef)
	register(PosixHooksPathAbs, rfc2119.Must, posixPlatformHooksRef)
	register(PosixHooksTimeoutPositive, rfc2119.Must, posixPlatformHooksRef)
	register(PosixHooksCalledInOrder, rfc2119.Must, posixPlatformHooksRef)
	register(PosixHooksStateToStdin, rfc2119.Must, posixPlatformHooksRef)
	register(PrestartTiming, rfc2119.Must, prestartRef)
	register(PoststartTiming, rfc2119.Must, poststartRef)
	register(PoststopTiming, rfc2119.Must, poststopRef)
	register(AnnotationsKeyValueMap, rfc2119.Must, annotationsRef)
	register(AnnotationsKeyString, rfc2119.Must, annotationsRef)
	register(AnnotationsKeyRequired, rfc2119.Must, annotationsRef)
	register(AnnotationsKeyReversedDomain, rfc2119.Should, annotationsRef)
	register(AnnotationsKeyReservedNS, rfc2119.Must, annotationsRef)
	register(AnnotationsKeyIgnoreUnknown, rfc2119.Must, annotationsRef)
	register(AnnotationsValueString, rfc2119.Must, annotationsRef)
	register(ExtensibilityIgnoreUnknownProp, rfc2119.Must, extensibilityRef)
	register(ValidValues, rfc2119.Must, validValuesRef)
}
