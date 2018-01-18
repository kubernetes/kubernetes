/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"testing"

	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
)

type mockPluginRunner struct {
	success bool
}

func (r *mockPluginRunner) Run(p *plugins.Plugin, ctx plugins.RunningContext) error {
	if !r.success {
		return fmt.Errorf("oops %s", p.Name)
	}
	ctx.Out.Write([]byte(fmt.Sprintf("ok: %s", p.Name)))
	return nil
}

func TestPluginCmd(t *testing.T) {
	tests := []struct {
		name            string
		plugin          *plugins.Plugin
		expectedSuccess bool
		expectedNilCmd  bool
	}{
		{
			name: "success",
			plugin: &plugins.Plugin{
				Description: plugins.Description{
					Name:      "success",
					ShortDesc: "The Test Plugin",
					Command:   "echo ok",
				},
			},
			expectedSuccess: true,
		},
		{
			name: "incomplete",
			plugin: &plugins.Plugin{
				Description: plugins.Description{
					Name:      "incomplete",
					ShortDesc: "The Incomplete Plugin",
				},
			},
			expectedNilCmd: true,
		},
		{
			name: "failure",
			plugin: &plugins.Plugin{
				Description: plugins.Description{
					Name:      "failure",
					ShortDesc: "The Failing Plugin",
					Command:   "false",
				},
			},
			expectedSuccess: false,
		},
	}

	for _, test := range tests {
		inBuf := bytes.NewBuffer([]byte{})
		outBuf := bytes.NewBuffer([]byte{})
		errBuf := bytes.NewBuffer([]byte{})

		cmdutil.BehaviorOnFatal(func(str string, code int) {
			errBuf.Write([]byte(str))
		})

		runner := &mockPluginRunner{
			success: test.expectedSuccess,
		}

		f, _, _, _ := cmdtesting.NewAPIFactory()
		cmd := NewCmdForPlugin(f, test.plugin, runner, inBuf, outBuf, errBuf)
		if cmd == nil {
			if !test.expectedNilCmd {
				t.Fatalf("%s: command was unexpectedly not registered", test.name)
			}
			continue
		}
		cmd.Run(cmd, []string{})

		if test.expectedSuccess && outBuf.String() != fmt.Sprintf("ok: %s", test.plugin.Name) {
			t.Errorf("%s: unexpected output: %q", test.name, outBuf.String())
		}

		if !test.expectedSuccess && errBuf.String() != fmt.Sprintf("error: oops %s", test.plugin.Name) {
			t.Errorf("%s: unexpected err output: %q", test.name, errBuf.String())
		}
	}
}
