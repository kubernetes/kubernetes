/*
Copyright 2015 The Kubernetes Authors.

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

package app

import (
	"github.com/docker/docker/pkg/term"

	"k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

/*
WARNING: this logic is duplicated, with minor changes, in cmd/hyperkube/kubectl.go
Any salient changes here will need to be manually reflected in that file.
*/
func Run() error {
	// need to use term.StdStreams to get the right IO refs on Windows
	stdin, stdout, stderr := term.StdStreams()
	cmd := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), stdin, stdout, stderr)
	return cmd.Execute()
}
