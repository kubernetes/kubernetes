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

package plugins

import (
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/golang/glog"
)

// PluginRunner is capable of running a plugin in a given running context.
type PluginRunner interface {
	Run(plugin *Plugin, ctx RunningContext) error
}

// RunningContext holds the context in which a given plugin is running - the
// in, out, and err streams, arguments and environment passed to it, and the
// working directory.
type RunningContext struct {
	In          io.Reader
	Out         io.Writer
	ErrOut      io.Writer
	Args        []string
	EnvProvider EnvProvider
	WorkingDir  string
}

// ExecPluginRunner is a PluginRunner that uses Go's os/exec to run plugins.
type ExecPluginRunner struct{}

// Run takes a given plugin and runs it in a given context using os/exec, returning
// any error found while running.
func (r *ExecPluginRunner) Run(plugin *Plugin, ctx RunningContext) error {
	command := strings.Split(os.ExpandEnv(plugin.Command), " ")
	base := command[0]
	args := []string{}
	if len(command) > 1 {
		args = command[1:]
	}
	args = append(args, ctx.Args...)

	cmd := exec.Command(base, args...)

	cmd.Stdin = ctx.In
	cmd.Stdout = ctx.Out
	cmd.Stderr = ctx.ErrOut

	env, err := ctx.EnvProvider.Env()
	if err != nil {
		return err
	}
	cmd.Env = env.Slice()
	cmd.Dir = ctx.WorkingDir

	glog.V(9).Infof("Running plugin %q as base command %q with args %v", plugin.Name, base, args)
	return cmd.Run()
}
