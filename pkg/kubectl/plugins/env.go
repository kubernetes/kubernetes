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
	"fmt"
	"os"
	"strings"

	"github.com/spf13/pflag"
)

// Env represents an environment variable with its name and value.
type Env struct {
	N string
	V string
}

// String returns "name=value" string.
func (e Env) String() string {
	return fmt.Sprintf("%s=%s", e.N, e.V)
}

// EnvList is a list of Env.
type EnvList []Env

// Slice returns a slice of "name=value" strings.
func (e EnvList) Slice() []string {
	envs := []string{}
	for _, env := range e {
		envs = append(envs, env.String())
	}
	return envs
}

// Merge converts "name=value" strings into Env values and merges them into e.
func (e EnvList) Merge(s ...string) EnvList {
	newList := e
	newList = append(newList, fromSlice(s)...)
	return newList
}

// EnvProvider provides the environment in which the plugin will run.
type EnvProvider interface {
	// Env returns the env list.
	Env() (EnvList, error)
}

// MultiEnvProvider satisfies the EnvProvider interface for multiple env providers.
type MultiEnvProvider []EnvProvider

// Env returns the combined env list of multiple env providers, returns on first error.
func (p MultiEnvProvider) Env() (EnvList, error) {
	env := EnvList{}
	for _, provider := range p {
		pEnv, err := provider.Env()
		if err != nil {
			return EnvList{}, err
		}
		env = append(env, pEnv...)
	}
	return env, nil
}

// PluginCallerEnvProvider satisfies the EnvProvider interface.
type PluginCallerEnvProvider struct{}

// Env returns env with the path to the caller binary (usually full path to 'kubectl').
func (p *PluginCallerEnvProvider) Env() (EnvList, error) {
	caller, err := os.Executable()
	if err != nil {
		return EnvList{}, err
	}
	return EnvList{
		{"KUBECTL_PLUGINS_CALLER", caller},
	}, nil
}

// PluginDescriptorEnvProvider satisfies the EnvProvider interface.
type PluginDescriptorEnvProvider struct {
	Plugin *Plugin
}

// Env returns env with information about the running plugin.
func (p *PluginDescriptorEnvProvider) Env() (EnvList, error) {
	if p.Plugin == nil {
		return []Env{}, fmt.Errorf("plugin not present to extract env")
	}
	prefix := "KUBECTL_PLUGINS_DESCRIPTOR_"
	env := EnvList{
		{prefix + "NAME", p.Plugin.Name},
		{prefix + "SHORT_DESC", p.Plugin.ShortDesc},
		{prefix + "LONG_DESC", p.Plugin.LongDesc},
		{prefix + "EXAMPLE", p.Plugin.Example},
		{prefix + "COMMAND", p.Plugin.Command},
	}
	return env, nil
}

// OSEnvProvider satisfies the EnvProvider interface.
type OSEnvProvider struct{}

// Env returns the current environment from the operating system.
func (p *OSEnvProvider) Env() (EnvList, error) {
	return fromSlice(os.Environ()), nil
}

// EmptyEnvProvider satisfies the EnvProvider interface.
type EmptyEnvProvider struct{}

// Env returns an empty environment.
func (p *EmptyEnvProvider) Env() (EnvList, error) {
	return EnvList{}, nil
}

// FlagToEnvName converts a flag string into a UNIX like environment variable name.
//  e.g --some-flag => "PREFIX_SOME_FLAG"
func FlagToEnvName(flagName, prefix string) string {
	envName := strings.TrimPrefix(flagName, "--")
	envName = strings.ToUpper(envName)
	envName = strings.Replace(envName, "-", "_", -1)
	envName = prefix + envName
	return envName
}

// FlagToEnv converts a flag and its value into an Env.
//  e.g --some-flag some-value => Env{N: "PREFIX_SOME_FLAG", V="SOME_VALUE"}
func FlagToEnv(flag *pflag.Flag, prefix string) Env {
	envName := FlagToEnvName(flag.Name, prefix)
	return Env{envName, flag.Value.String()}
}

func fromSlice(envs []string) EnvList {
	list := EnvList{}
	for _, env := range envs {
		list = append(list, parseEnv(env))
	}
	return list
}

func parseEnv(env string) Env {
	if !strings.Contains(env, "=") {
		env = env + "="
	}
	parsed := strings.SplitN(env, "=", 2)
	return Env{parsed[0], parsed[1]}
}
