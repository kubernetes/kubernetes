/*
Copyright 2016 The Kubernetes Authors.

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

package kubeadm

import (
	"bytes"
	"fmt"
	"os/exec"
	"testing"

	"github.com/spf13/cobra"
)

// Forked from test/e2e/framework because the e2e framework is quite bloated
// for our purposes here, and modified to remove undesired logging.
func RunCmd(command string, args ...string) (string, string, error) {
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	cmd.Stdout = &bout
	cmd.Stderr = &berr
	err := cmd.Run()
	stdout, stderr := bout.String(), berr.String()
	if err != nil {
		return "", "", fmt.Errorf("error running %s %v; \ngot error %v, \nstdout %q, \nstderr %q",
			command, args, err, stdout, stderr)
	}
	return stdout, stderr, nil
}

// RunSubCommand is a utility function for kubeadm testing that executes a Cobra sub command
func RunSubCommand(t *testing.T, subCmds []*cobra.Command, command string, args ...string) {
	subCmd := getSubCommand(t, subCmds, command)
	subCmd.SetArgs(args)
	if err := subCmd.Execute(); err != nil {
		t.Fatalf("Could not execute subcommand: %s", command)
	}
}

// AssertSubCommandHasFlags is a utility function for kubeadm testing that assert if a Cobra sub command has expected flags
func AssertSubCommandHasFlags(t *testing.T, subCmds []*cobra.Command, command string, flags ...string) {
	subCmd := getSubCommand(t, subCmds, command)

	for _, flag := range flags {
		if subCmd.Flags().Lookup(flag) == nil {
			t.Errorf("Could not find expecte flag %s for command %s", flag, command)
		}
	}
}

func getSubCommand(t *testing.T, subCmds []*cobra.Command, name string) *cobra.Command {
	for _, subCmd := range subCmds {
		if subCmd.Name() == name {
			return subCmd
		}
	}
	t.Fatalf("Unable to find sub command %s", name)

	return nil
}
