/*
Copyright 2020 The Kubernetes Authors.

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

package version

import (
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

func TestNewCmdVersionWithoutConfigFile(t *testing.T) {
	tf := cmdutil.NewFactory(&genericclioptions.ConfigFlags{})
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdVersion(tf, streams)
	cmd.SetOutput(buf)
	if err := cmd.Execute(); err != nil {
		t.Errorf("Cannot execute version command: %v", err)
	}
	if !strings.Contains(buf.String(), "Client Version") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
