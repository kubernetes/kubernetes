// Copyright 2015 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package invoke

import (
	"fmt"
	"os"
	"strings"
)

type CNIArgs interface {
	// For use with os/exec; i.e., return nil to inherit the
	// environment from this process
	// For use in delegation; inherit the environment from this
	// process and allow overrides
	AsEnv() []string
}

type inherited struct{}

var inheritArgsFromEnv inherited

func (*inherited) AsEnv() []string {
	return nil
}

func ArgsFromEnv() CNIArgs {
	return &inheritArgsFromEnv
}

type Args struct {
	Command       string
	ContainerID   string
	NetNS         string
	PluginArgs    [][2]string
	PluginArgsStr string
	IfName        string
	Path          string
}

// Args implements the CNIArgs interface
var _ CNIArgs = &Args{}

func (args *Args) AsEnv() []string {
	env := os.Environ()
	pluginArgsStr := args.PluginArgsStr
	if pluginArgsStr == "" {
		pluginArgsStr = stringify(args.PluginArgs)
	}

	// Duplicated values which come first will be overridden, so we must put the
	// custom values in the end to avoid being overridden by the process environments.
	env = append(env,
		"CNI_COMMAND="+args.Command,
		"CNI_CONTAINERID="+args.ContainerID,
		"CNI_NETNS="+args.NetNS,
		"CNI_ARGS="+pluginArgsStr,
		"CNI_IFNAME="+args.IfName,
		"CNI_PATH="+args.Path,
	)
	return dedupEnv(env)
}

// taken from rkt/networking/net_plugin.go
func stringify(pluginArgs [][2]string) string {
	entries := make([]string, len(pluginArgs))

	for i, kv := range pluginArgs {
		entries[i] = strings.Join(kv[:], "=")
	}

	return strings.Join(entries, ";")
}

// DelegateArgs implements the CNIArgs interface
// used for delegation to inherit from environments
// and allow some overrides like CNI_COMMAND
var _ CNIArgs = &DelegateArgs{}

type DelegateArgs struct {
	Command string
}

func (d *DelegateArgs) AsEnv() []string {
	env := os.Environ()

	// The custom values should come in the end to override the existing
	// process environment of the same key.
	env = append(env,
		"CNI_COMMAND="+d.Command,
	)
	return dedupEnv(env)
}

// dedupEnv returns a copy of env with any duplicates removed, in favor of later values.
// Items not of the normal environment "key=value" form are preserved unchanged.
func dedupEnv(env []string) []string {
	out := make([]string, 0, len(env))
	envMap := map[string]string{}

	for _, kv := range env {
		// find the first "=" in environment, if not, just keep it
		eq := strings.Index(kv, "=")
		if eq < 0 {
			out = append(out, kv)
			continue
		}
		envMap[kv[:eq]] = kv[eq+1:]
	}

	for k, v := range envMap {
		out = append(out, fmt.Sprintf("%s=%s", k, v))
	}

	return out
}
