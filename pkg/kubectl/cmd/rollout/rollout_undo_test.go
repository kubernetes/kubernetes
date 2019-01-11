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

package rollout

import (
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func TestResourceErrors(t *testing.T) {
	testCases := map[string]struct {
		args    []string
		errFn   func(string) bool
		allFlag string
		file    string
	}{
		"no args": {
			args: []string{},
			errFn: func(err string) bool {
				return (strings.Contains(err, "required resource not specified") ||
					strings.Contains(err, "You must provide one or more resources by argument or filename"))
			},
		},
		"resource with all flag": {
			args: []string{""},
			errFn: func(err string) bool {
				return (strings.Contains(err, "cannot set --all and --filename at the same time") ||
					strings.Contains(err, "the path \"/test.yaml\" does not exist"))
			},
			allFlag: "true",
			file:    "/test.yaml",
		},
	}

	for k, testCase := range testCases {
		t.Run(k, func(t *testing.T) {
			cmdutil.BehaviorOnFatal(func(str string, code int) {
				if !testCase.errFn(str) {
					t.Errorf("%s: unexpected error: %v", k, str)
					return
				}
			})
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			streams, _, _, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdRolloutUndo(tf, streams)
			if testCase.allFlag != "" {
				cmd.Flags().Set("all", testCase.allFlag)
			}
			if testCase.file != "" {
				cmd.Flags().Set("filename", testCase.file)
			}
			cmd.Run(cmd, testCase.args)
		})
	}
}
