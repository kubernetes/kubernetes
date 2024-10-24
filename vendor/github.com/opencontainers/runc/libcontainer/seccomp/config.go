package seccomp

import (
	"fmt"
	"sort"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runtime-spec/specs-go"
)

// flagTsync is recognized but ignored by runc, and it is not defined
// in the runtime-spec.
const flagTsync = "SECCOMP_FILTER_FLAG_TSYNC"

var operators = map[string]configs.Operator{
	"SCMP_CMP_NE":        configs.NotEqualTo,
	"SCMP_CMP_LT":        configs.LessThan,
	"SCMP_CMP_LE":        configs.LessThanOrEqualTo,
	"SCMP_CMP_EQ":        configs.EqualTo,
	"SCMP_CMP_GE":        configs.GreaterThanOrEqualTo,
	"SCMP_CMP_GT":        configs.GreaterThan,
	"SCMP_CMP_MASKED_EQ": configs.MaskEqualTo,
}

// KnownOperators returns the list of the known operations.
// Used by `runc features`.
func KnownOperators() []string {
	var res []string
	for k := range operators {
		res = append(res, k)
	}
	sort.Strings(res)
	return res
}

var actions = map[string]configs.Action{
	"SCMP_ACT_KILL":         configs.Kill,
	"SCMP_ACT_ERRNO":        configs.Errno,
	"SCMP_ACT_TRAP":         configs.Trap,
	"SCMP_ACT_ALLOW":        configs.Allow,
	"SCMP_ACT_TRACE":        configs.Trace,
	"SCMP_ACT_LOG":          configs.Log,
	"SCMP_ACT_NOTIFY":       configs.Notify,
	"SCMP_ACT_KILL_THREAD":  configs.KillThread,
	"SCMP_ACT_KILL_PROCESS": configs.KillProcess,
}

// KnownActions returns the list of the known actions.
// Used by `runc features`.
func KnownActions() []string {
	var res []string
	for k := range actions {
		res = append(res, k)
	}
	sort.Strings(res)
	return res
}

var archs = map[string]string{
	"SCMP_ARCH_X86":         "x86",
	"SCMP_ARCH_X86_64":      "amd64",
	"SCMP_ARCH_X32":         "x32",
	"SCMP_ARCH_ARM":         "arm",
	"SCMP_ARCH_AARCH64":     "arm64",
	"SCMP_ARCH_MIPS":        "mips",
	"SCMP_ARCH_MIPS64":      "mips64",
	"SCMP_ARCH_MIPS64N32":   "mips64n32",
	"SCMP_ARCH_MIPSEL":      "mipsel",
	"SCMP_ARCH_MIPSEL64":    "mipsel64",
	"SCMP_ARCH_MIPSEL64N32": "mipsel64n32",
	"SCMP_ARCH_PPC":         "ppc",
	"SCMP_ARCH_PPC64":       "ppc64",
	"SCMP_ARCH_PPC64LE":     "ppc64le",
	"SCMP_ARCH_RISCV64":     "riscv64",
	"SCMP_ARCH_S390":        "s390",
	"SCMP_ARCH_S390X":       "s390x",
}

// KnownArchs returns the list of the known archs.
// Used by `runc features`.
func KnownArchs() []string {
	var res []string
	for k := range archs {
		res = append(res, k)
	}
	sort.Strings(res)
	return res
}

// ConvertStringToOperator converts a string into a Seccomp comparison operator.
// Comparison operators use the names they are assigned by Libseccomp's header.
// Attempting to convert a string that is not a valid operator results in an
// error.
func ConvertStringToOperator(in string) (configs.Operator, error) {
	if op, ok := operators[in]; ok {
		return op, nil
	}
	return 0, fmt.Errorf("string %s is not a valid operator for seccomp", in)
}

// ConvertStringToAction converts a string into a Seccomp rule match action.
// Actions use the names they are assigned in Libseccomp's header.
// Attempting to convert a string that is not a valid action results in an
// error.
func ConvertStringToAction(in string) (configs.Action, error) {
	if act, ok := actions[in]; ok {
		return act, nil
	}
	return 0, fmt.Errorf("string %s is not a valid action for seccomp", in)
}

// ConvertStringToArch converts a string into a Seccomp comparison arch.
func ConvertStringToArch(in string) (string, error) {
	if arch, ok := archs[in]; ok {
		return arch, nil
	}
	return "", fmt.Errorf("string %s is not a valid arch for seccomp", in)
}

// List of flags known to this version of runc.
var flags = []string{
	flagTsync,
	string(specs.LinuxSeccompFlagSpecAllow),
	string(specs.LinuxSeccompFlagLog),
}

// KnownFlags returns the list of the known filter flags.
// Used by `runc features`.
func KnownFlags() []string {
	return flags
}

// SupportedFlags returns the list of the supported filter flags.
// This list may be a subset of one returned by KnownFlags due to
// some flags not supported by the current kernel and/or libseccomp.
// Used by `runc features`.
func SupportedFlags() []string {
	if !Enabled {
		return nil
	}

	var res []string
	for _, flag := range flags {
		if FlagSupported(specs.LinuxSeccompFlag(flag)) == nil {
			res = append(res, flag)
		}
	}

	return res
}
