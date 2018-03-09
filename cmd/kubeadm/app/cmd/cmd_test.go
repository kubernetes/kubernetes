/*
Copyright 2018 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"testing"
)

func TestNewKubeadmCommand(t *testing.T) {
	var buf bytes.Buffer
	found := []string{}
	commandUses := []string{
		"alpha",
		"completion SHELL",
		"config",
		"init",
		"join",
		"reset",
		"token",
		"upgrade",
		"version",
	}

	// obtain the list of command from NewKubeadmCommand()
	// and then match the command uses in the `commandUses` slice,
	// while filling the `found` slice at the same time.
	// report differences.
	kubeadmCommands := NewKubeadmCommand(&buf, &buf, &buf)
	cmdList := kubeadmCommands.Commands()
	if len(commandUses) != len(cmdList) {
		reportLengthError(t, len(commandUses), len(cmdList))
	}

	for _, cmd := range cmdList {
		use := cmd.Use
		if contains(found, use) {
			t.Fatalf("Multiple definitions of command: %s", use)
		}
		if contains(commandUses, use) {
			found = append(found, use)
		} else {
			t.Fatalf("Unknown command %q", use)
		}
	}

	if len(found) != len(commandUses) {
		reportLengthError(t, len(commandUses), len(found))
	}
}

func reportLengthError(t *testing.T, lenE, lenF int) {
	t.Fatalf("Expected %d commands, found %d", lenE, lenF)
}

func contains(slice []string, element string) bool {
	for _, v := range slice {
		if element == v {
			return true
		}
	}
	return false
}
