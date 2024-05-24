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
	"io"
	"os"
	"os/exec"
	"testing"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
)

// Forked from test/e2e/framework because the e2e framework is quite bloated
// for our purposes here, and modified to remove undesired logging.

func runCmdNoWrap(command string, args ...string) (string, string, int, error) {
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	cmd.Stdout = &bout
	cmd.Stderr = &berr
	err := cmd.Run()
	stdout, stderr := bout.String(), berr.String()
	return stdout, stderr, cmd.ProcessState.ExitCode(), err
}

// RunCmd is a utility function for kubeadm testing that executes a specified command
func RunCmd(command string, args ...string) (string, string, int, error) {
	stdout, stderr, retcode, err := runCmdNoWrap(command, args...)
	if err != nil {
		return stdout, stderr, retcode, errors.Wrapf(err, "error running %s %v; \nretcode %d, \nstdout %q, \nstderr %q, \ngot error",
			command, args, retcode, stdout, stderr)
	}
	return stdout, stderr, retcode, nil
}

// RunSubCommand is a utility function for kubeadm testing that executes a Cobra sub command
func RunSubCommand(t *testing.T, subCmds []*cobra.Command, command string, output io.Writer, args ...string) error {
	subCmd := getSubCommand(t, subCmds, command)
	subCmd.SetOut(output)
	subCmd.SetArgs(args)
	if err := subCmd.Execute(); err != nil {
		return err
	}
	return nil
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

// getKubeadmPath returns the contents of the environment variable KUBEADM_PATH
// or panics if it's empty
func getKubeadmPath() string {
	kubeadmPath := os.Getenv("KUBEADM_PATH")
	if len(kubeadmPath) == 0 {
		panic("the environment variable KUBEADM_PATH must point to the kubeadm binary path")
	}
	return kubeadmPath
}
