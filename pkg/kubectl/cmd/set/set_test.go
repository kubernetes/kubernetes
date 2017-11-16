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

package set

import (
	"bytes"
	"testing"

	"github.com/spf13/cobra"

	clientcmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"os"
)

func TestLocalAndDryRunFlags(t *testing.T) {
	out := &bytes.Buffer{}
	errout := &bytes.Buffer{}
	f := clientcmdutil.NewFactory(nil)
	setCmd := NewCmdSet(f, os.Stdin, out, errout)
	ensureLocalAndDryRunFlagsOnChildren(t, setCmd, "")
}

func ensureLocalAndDryRunFlagsOnChildren(t *testing.T, c *cobra.Command, prefix string) {
	for _, cmd := range c.Commands() {
		name := prefix + cmd.Name()
		if localFlag := cmd.Flag("local"); localFlag == nil {
			t.Errorf("Command %s does not implement the --local flag", name)
		}
		if dryRunFlag := cmd.Flag("dry-run"); dryRunFlag == nil {
			t.Errorf("Command %s does not implement the --dry-run flag", name)
		}
		ensureLocalAndDryRunFlagsOnChildren(t, cmd, name+".")
	}
}
