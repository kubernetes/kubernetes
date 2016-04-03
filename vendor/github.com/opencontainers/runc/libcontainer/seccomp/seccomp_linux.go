// +build linux,cgo,seccomp

package seccomp

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"syscall"

	"github.com/opencontainers/runc/libcontainer/configs"
	libseccomp "github.com/seccomp/libseccomp-golang"
)

var (
	actAllow = libseccomp.ActAllow
	actTrap  = libseccomp.ActTrap
	actKill  = libseccomp.ActKill
	actTrace = libseccomp.ActTrace.SetReturnCode(int16(syscall.EPERM))
	actErrno = libseccomp.ActErrno.SetReturnCode(int16(syscall.EPERM))

	// SeccompModeFilter refers to the syscall argument SECCOMP_MODE_FILTER.
	SeccompModeFilter = uintptr(2)
)

// Filters given syscalls in a container, preventing them from being used
// Started in the container init process, and carried over to all child processes
// Setns calls, however, require a separate invocation, as they are not children
// of the init until they join the namespace
func InitSeccomp(config *configs.Seccomp) error {
	if config == nil {
		return fmt.Errorf("cannot initialize Seccomp - nil config passed")
	}

	defaultAction, err := getAction(config.DefaultAction)
	if err != nil {
		return fmt.Errorf("error initializing seccomp - invalid default action")
	}

	filter, err := libseccomp.NewFilter(defaultAction)
	if err != nil {
		return fmt.Errorf("error creating filter: %s", err)
	}

	// Add extra architectures
	for _, arch := range config.Architectures {
		scmpArch, err := libseccomp.GetArchFromString(arch)
		if err != nil {
			return err
		}

		if err := filter.AddArch(scmpArch); err != nil {
			return err
		}
	}

	// Unset no new privs bit
	if err := filter.SetNoNewPrivsBit(false); err != nil {
		return fmt.Errorf("error setting no new privileges: %s", err)
	}

	// Add a rule for each syscall
	for _, call := range config.Syscalls {
		if call == nil {
			return fmt.Errorf("encountered nil syscall while initializing Seccomp")
		}

		if err = matchCall(filter, call); err != nil {
			return err
		}
	}

	if err = filter.Load(); err != nil {
		return fmt.Errorf("error loading seccomp filter into kernel: %s", err)
	}

	return nil
}

// IsEnabled returns if the kernel has been configured to support seccomp.
func IsEnabled() bool {
	// Try to read from /proc/self/status for kernels > 3.8
	s, err := parseStatusFile("/proc/self/status")
	if err != nil {
		// Check if Seccomp is supported, via CONFIG_SECCOMP.
		if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_GET_SECCOMP, 0, 0); err != syscall.EINVAL {
			// Make sure the kernel has CONFIG_SECCOMP_FILTER.
			if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_SET_SECCOMP, SeccompModeFilter, 0); err != syscall.EINVAL {
				return true
			}
		}
		return false
	}
	_, ok := s["Seccomp"]
	return ok
}

// Convert Libcontainer Action to Libseccomp ScmpAction
func getAction(act configs.Action) (libseccomp.ScmpAction, error) {
	switch act {
	case configs.Kill:
		return actKill, nil
	case configs.Errno:
		return actErrno, nil
	case configs.Trap:
		return actTrap, nil
	case configs.Allow:
		return actAllow, nil
	case configs.Trace:
		return actTrace, nil
	default:
		return libseccomp.ActInvalid, fmt.Errorf("invalid action, cannot use in rule")
	}
}

// Convert Libcontainer Operator to Libseccomp ScmpCompareOp
func getOperator(op configs.Operator) (libseccomp.ScmpCompareOp, error) {
	switch op {
	case configs.EqualTo:
		return libseccomp.CompareEqual, nil
	case configs.NotEqualTo:
		return libseccomp.CompareNotEqual, nil
	case configs.GreaterThan:
		return libseccomp.CompareGreater, nil
	case configs.GreaterThanOrEqualTo:
		return libseccomp.CompareGreaterEqual, nil
	case configs.LessThan:
		return libseccomp.CompareLess, nil
	case configs.LessThanOrEqualTo:
		return libseccomp.CompareLessOrEqual, nil
	case configs.MaskEqualTo:
		return libseccomp.CompareMaskedEqual, nil
	default:
		return libseccomp.CompareInvalid, fmt.Errorf("invalid operator, cannot use in rule")
	}
}

// Convert Libcontainer Arg to Libseccomp ScmpCondition
func getCondition(arg *configs.Arg) (libseccomp.ScmpCondition, error) {
	cond := libseccomp.ScmpCondition{}

	if arg == nil {
		return cond, fmt.Errorf("cannot convert nil to syscall condition")
	}

	op, err := getOperator(arg.Op)
	if err != nil {
		return cond, err
	}

	return libseccomp.MakeCondition(arg.Index, op, arg.Value, arg.ValueTwo)
}

// Add a rule to match a single syscall
func matchCall(filter *libseccomp.ScmpFilter, call *configs.Syscall) error {
	if call == nil || filter == nil {
		return fmt.Errorf("cannot use nil as syscall to block")
	}

	if len(call.Name) == 0 {
		return fmt.Errorf("empty string is not a valid syscall")
	}

	// If we can't resolve the syscall, assume it's not supported on this kernel
	// Ignore it, don't error out
	callNum, err := libseccomp.GetSyscallFromName(call.Name)
	if err != nil {
		log.Printf("Error resolving syscall name %s: %s - ignoring syscall.", call.Name, err)
		return nil
	}

	// Convert the call's action to the libseccomp equivalent
	callAct, err := getAction(call.Action)
	if err != nil {
		return err
	}

	// Unconditional match - just add the rule
	if len(call.Args) == 0 {
		if err = filter.AddRule(callNum, callAct); err != nil {
			return err
		}
	} else {
		// Conditional match - convert the per-arg rules into library format
		conditions := []libseccomp.ScmpCondition{}

		for _, cond := range call.Args {
			newCond, err := getCondition(cond)
			if err != nil {
				return err
			}

			conditions = append(conditions, newCond)
		}

		if err = filter.AddRuleConditional(callNum, callAct, conditions); err != nil {
			return err
		}
	}

	return nil
}

func parseStatusFile(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	status := make(map[string]string)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := s.Text()
		parts := strings.Split(text, ":")

		if len(parts) <= 1 {
			continue
		}

		status[parts[0]] = parts[1]
	}
	return status, nil
}
