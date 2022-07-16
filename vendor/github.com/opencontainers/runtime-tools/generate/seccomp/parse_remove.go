package seccomp

import (
	"fmt"
	"reflect"
	"strings"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

// RemoveAction takes the argument string that was passed with the --remove flag,
// parses it, and updates the Seccomp config accordingly
func RemoveAction(arguments string, config *rspec.LinuxSeccomp) error {
	if config == nil {
		return fmt.Errorf("Cannot remove action from nil Seccomp pointer")
	}

	syscallsToRemove := strings.Split(arguments, ",")

	for counter, syscallStruct := range config.Syscalls {
		if reflect.DeepEqual(syscallsToRemove, syscallStruct.Names) {
			config.Syscalls = append(config.Syscalls[:counter], config.Syscalls[counter+1:]...)
		}
	}

	return nil
}

// RemoveAllSeccompRules removes all seccomp syscall rules
func RemoveAllSeccompRules(config *rspec.LinuxSeccomp) error {
	if config == nil {
		return fmt.Errorf("Cannot remove action from nil Seccomp pointer")
	}
	newSyscallSlice := []rspec.LinuxSyscall{}
	config.Syscalls = newSyscallSlice
	return nil
}

// RemoveAllMatchingRules will remove any syscall rules that match the specified action
func RemoveAllMatchingRules(config *rspec.LinuxSeccomp, seccompAction rspec.LinuxSeccompAction) error {
	if config == nil {
		return fmt.Errorf("Cannot remove action from nil Seccomp pointer")
	}

	for _, syscall := range config.Syscalls {
		if reflect.DeepEqual(syscall.Action, seccompAction) {
			RemoveAction(strings.Join(syscall.Names, ","), config)
		}
	}

	return nil
}
