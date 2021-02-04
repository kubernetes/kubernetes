/*
Copyright 2021 The Kubernetes Authors.

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

package lint

import (
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func ExampleNewCmdLint() {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdLint(tf, streams)
	err := cmd.Flags().Set("filename", "./testdata/container_running_as_root")
	checkErr(err)
	cmd.SetOutput(buf)
	err = cmd.RunE(cmd, nil)
	checkErr(err)
	// testdata/container_running_as_root/pod1.yaml: Pod/myapp2/container nginx is running as root
	// Error: resources had linting errors
	// exit status 1
}

func checkErr(e error) {
	if e != nil {
		panic(e)
	}
}
