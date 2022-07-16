package seccomp

import (
	"fmt"
	"strconv"
	"strings"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

// SyscallOpts contain options for parsing syscall rules
type SyscallOpts struct {
	Action   string
	Syscall  string
	Index    string
	Value    string
	ValueTwo string
	Operator string
}

// ParseSyscallFlag takes a SyscallOpts struct and the seccomp configuration
// and sets the new syscall rule accordingly
func ParseSyscallFlag(args SyscallOpts, config *rspec.LinuxSeccomp) error {
	var arguments []string
	if args.Index != "" && args.Value != "" && args.ValueTwo != "" && args.Operator != "" {
		arguments = []string{args.Action, args.Syscall, args.Index, args.Value,
			args.ValueTwo, args.Operator}
	} else {
		arguments = []string{args.Action, args.Syscall}
	}

	action, _ := parseAction(arguments[0])
	if action == config.DefaultAction && args.argsAreEmpty() {
		// default already set, no need to make changes
		return nil
	}

	var newSyscall rspec.LinuxSyscall
	numOfArgs := len(arguments)
	if numOfArgs == 6 || numOfArgs == 2 {
		argStruct, err := parseArguments(arguments[1:])
		if err != nil {
			return err
		}
		newSyscall = newSyscallStruct(arguments[1], action, argStruct)
	} else {
		return fmt.Errorf("incorrect number of arguments to ParseSyscall: %d", numOfArgs)
	}

	descison, err := decideCourseOfAction(&newSyscall, config.Syscalls)
	if err != nil {
		return err
	}
	delimDescison := strings.Split(descison, ":")

	if delimDescison[0] == seccompAppend {
		config.Syscalls = append(config.Syscalls, newSyscall)
	}

	if delimDescison[0] == seccompOverwrite {
		indexForOverwrite, err := strconv.ParseInt(delimDescison[1], 10, 32)
		if err != nil {
			return err
		}
		config.Syscalls[indexForOverwrite] = newSyscall
	}

	return nil
}

var actions = map[string]rspec.LinuxSeccompAction{
	"allow": rspec.ActAllow,
	"errno": rspec.ActErrno,
	"kill":  rspec.ActKill,
	"trace": rspec.ActTrace,
	"trap":  rspec.ActTrap,
}

// Take passed action, return the SCMP_ACT_<ACTION> version of it
func parseAction(action string) (rspec.LinuxSeccompAction, error) {
	a, ok := actions[action]
	if !ok {
		return "", fmt.Errorf("unrecognized action: %s", action)
	}
	return a, nil
}

// ParseDefaultAction sets the default action of the seccomp configuration
// and then removes any rules that were already specified with this action
func ParseDefaultAction(action string, config *rspec.LinuxSeccomp) error {
	if action == "" {
		return nil
	}

	defaultAction, err := parseAction(action)
	if err != nil {
		return err
	}
	config.DefaultAction = defaultAction
	err = RemoveAllMatchingRules(config, defaultAction)
	if err != nil {
		return err
	}
	return nil
}

// ParseDefaultActionForce simply sets the default action of the seccomp configuration
func ParseDefaultActionForce(action string, config *rspec.LinuxSeccomp) error {
	if action == "" {
		return nil
	}

	defaultAction, err := parseAction(action)
	if err != nil {
		return err
	}
	config.DefaultAction = defaultAction
	return nil
}

func newSyscallStruct(name string, action rspec.LinuxSeccompAction, args []rspec.LinuxSeccompArg) rspec.LinuxSyscall {
	syscallStruct := rspec.LinuxSyscall{
		Names:  []string{name},
		Action: action,
		Args:   args,
	}
	return syscallStruct
}

func (s SyscallOpts) argsAreEmpty() bool {
	return (s.Index == "" &&
		s.Value == "" &&
		s.ValueTwo == "" &&
		s.Operator == "")
}
