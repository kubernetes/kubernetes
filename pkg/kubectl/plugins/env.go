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

// Env represents an environment variable with its name and value
type Env struct {
	N string
	V string
}

func (e Env) String() string {
	return fmt.Sprintf("%s=%s", e.N, e.V)
}

// EnvList is a list of Env
type EnvList []Env

func (e EnvList) Slice() []string {
	envs := []string{}
	for _, env := range e {
		envs = append(envs, env.String())
	}
	return envs
}

func (e EnvList) Merge(s ...string) EnvList {
	newList := e
	newList = append(newList, fromSlice(s)...)
	return newList
}

// EnvProvider provides the environment in which the plugin will run.
type EnvProvider interface {
	Env() (EnvList, error)
}

// MultiEnvProvider is an EnvProvider for multiple env providers, returns on first error.
type MultiEnvProvider []EnvProvider

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

// PluginCallerEnvProvider provides env with the path to the caller binary (usually full path to 'kubectl').
type PluginCallerEnvProvider struct{}

func (p *PluginCallerEnvProvider) Env() (EnvList, error) {
	caller, err := os.Executable()
	if err != nil {
		return EnvList{}, err
	}
	return EnvList{
		{"KUBECTL_PLUGINS_CALLER", caller},
	}, nil
}

// PluginDescriptorEnvProvider provides env vars with information about the running plugin.
type PluginDescriptorEnvProvider struct {
	Plugin *Plugin
}

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

// OSEnvProvider provides current environment from the operating system.
type OSEnvProvider struct{}

func (p *OSEnvProvider) Env() (EnvList, error) {
	return fromSlice(os.Environ()), nil
}

type EmptyEnvProvider struct{}

func (p *EmptyEnvProvider) Env() (EnvList, error) {
	return EnvList{}, nil
}

func FlagToEnvName(flagName, prefix string) string {
	envName := strings.TrimPrefix(flagName, "--")
	envName = strings.ToUpper(envName)
	envName = strings.Replace(envName, "-", "_", -1)
	envName = prefix + envName
	return envName
}

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
