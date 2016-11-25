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

package plugins

import (
	"io"
	"os/exec"

	"github.com/golang/glog"
)

type Plugin struct {
	Path string
	PluginMetadata
}

type PluginMetadata struct {
	Use     string `json:"use"`
	Short   string `json:"short"`
	Long    string `json:"long,omitempty"`
	Example string `json:"example,omitempty"`
	Tunnel  bool   `json:"tunnel,omitempty"`
}

type Plugins []*Plugin

func (p *Plugin) Run(in io.Reader, out, errout io.Writer, env []string, args ...string) error {
	cmd := exec.Command(p.Path, args...)

	cmd.Stdin = in
	cmd.Stdout = out
	cmd.Stderr = errout

	cmd.Env = env

	glog.V(9).Infof("Running plugin with %s", cmd.Args)
	return cmd.Run()
}
