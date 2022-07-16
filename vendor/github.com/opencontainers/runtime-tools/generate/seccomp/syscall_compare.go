package seccomp

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

// Determine if a new syscall rule should be appended, overwrite an existing rule
// or if no action should be taken at all
func decideCourseOfAction(newSyscall *rspec.LinuxSyscall, syscalls []rspec.LinuxSyscall) (string, error) {
	ruleForSyscallAlreadyExists := false

	var sliceOfDeterminedActions []string
	for i, syscall := range syscalls {
		if sameName(&syscall, newSyscall) {
			ruleForSyscallAlreadyExists = true

			if identical(newSyscall, &syscall) {
				sliceOfDeterminedActions = append(sliceOfDeterminedActions, nothing)
			}

			if sameAction(newSyscall, &syscall) {
				if bothHaveArgs(newSyscall, &syscall) {
					sliceOfDeterminedActions = append(sliceOfDeterminedActions, seccompAppend)
				}
				if onlyOneHasArgs(newSyscall, &syscall) {
					if firstParamOnlyHasArgs(newSyscall, &syscall) {
						sliceOfDeterminedActions = append(sliceOfDeterminedActions, "overwrite:"+strconv.Itoa(i))
					} else {
						sliceOfDeterminedActions = append(sliceOfDeterminedActions, nothing)
					}
				}
			}

			if !sameAction(newSyscall, &syscall) {
				if bothHaveArgs(newSyscall, &syscall) {
					if sameArgs(newSyscall, &syscall) {
						sliceOfDeterminedActions = append(sliceOfDeterminedActions, "overwrite:"+strconv.Itoa(i))
					}
					if !sameArgs(newSyscall, &syscall) {
						sliceOfDeterminedActions = append(sliceOfDeterminedActions, seccompAppend)
					}
				}
				if onlyOneHasArgs(newSyscall, &syscall) {
					sliceOfDeterminedActions = append(sliceOfDeterminedActions, seccompAppend)
				}
				if neitherHasArgs(newSyscall, &syscall) {
					sliceOfDeterminedActions = append(sliceOfDeterminedActions, "overwrite:"+strconv.Itoa(i))
				}
			}
		}
	}

	if !ruleForSyscallAlreadyExists {
		sliceOfDeterminedActions = append(sliceOfDeterminedActions, seccompAppend)
	}

	// Nothing has highest priority
	for _, determinedAction := range sliceOfDeterminedActions {
		if determinedAction == nothing {
			return determinedAction, nil
		}
	}

	// Overwrite has second highest priority
	for _, determinedAction := range sliceOfDeterminedActions {
		if strings.Contains(determinedAction, seccompOverwrite) {
			return determinedAction, nil
		}
	}

	// Append has the lowest priority
	for _, determinedAction := range sliceOfDeterminedActions {
		if determinedAction == seccompAppend {
			return determinedAction, nil
		}
	}

	return "", fmt.Errorf("Trouble determining action: %s", sliceOfDeterminedActions)
}

func hasArguments(config *rspec.LinuxSyscall) bool {
	nilSyscall := new(rspec.LinuxSyscall)
	return !sameArgs(nilSyscall, config)
}

func identical(config1, config2 *rspec.LinuxSyscall) bool {
	return reflect.DeepEqual(config1, config2)
}

func identicalExceptAction(config1, config2 *rspec.LinuxSyscall) bool {
	samename := sameName(config1, config2)
	sameAction := sameAction(config1, config2)
	sameArgs := sameArgs(config1, config2)

	return samename && !sameAction && sameArgs
}

func identicalExceptArgs(config1, config2 *rspec.LinuxSyscall) bool {
	samename := sameName(config1, config2)
	sameAction := sameAction(config1, config2)
	sameArgs := sameArgs(config1, config2)

	return samename && sameAction && !sameArgs
}

func sameName(config1, config2 *rspec.LinuxSyscall) bool {
	return reflect.DeepEqual(config1.Names, config2.Names)
}

func sameAction(config1, config2 *rspec.LinuxSyscall) bool {
	return config1.Action == config2.Action
}

func sameArgs(config1, config2 *rspec.LinuxSyscall) bool {
	return reflect.DeepEqual(config1.Args, config2.Args)
}

func bothHaveArgs(config1, config2 *rspec.LinuxSyscall) bool {
	return hasArguments(config1) && hasArguments(config2)
}

func onlyOneHasArgs(config1, config2 *rspec.LinuxSyscall) bool {
	conf1 := hasArguments(config1)
	conf2 := hasArguments(config2)

	return (conf1 && !conf2) || (!conf1 && conf2)
}

func neitherHasArgs(config1, config2 *rspec.LinuxSyscall) bool {
	return !hasArguments(config1) && !hasArguments(config2)
}

func firstParamOnlyHasArgs(config1, config2 *rspec.LinuxSyscall) bool {
	return !hasArguments(config1) && hasArguments(config2)
}
