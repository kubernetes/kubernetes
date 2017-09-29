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
	"fmt"
	"io"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	plugin_long = templates.LongDesc(`
		Runs a command-line plugin.

		Plugins are subcommands that are not part of the major command-line distribution 
		and can even be provided by third-parties. Please refer to the documentation and 
		examples for more information about how to install and write your own plugins.`)
)

// NewCmdPlugin creates the command that is the top-level for plugin commands.
func NewCmdPlugin(f cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	// Loads plugins and create commands for each plugin identified
	loadedPlugins, loadErr := f.PluginLoader().Load()
	if loadErr != nil {
		glog.V(1).Infof("Unable to load plugins: %v", loadErr)
	}

	cmd := &cobra.Command{
		Use:   "plugin NAME",
		Short: i18n.T("Runs a command-line plugin"),
		Long:  plugin_long,
		Run: func(cmd *cobra.Command, args []string) {
			if len(loadedPlugins) == 0 {
				cmdutil.CheckErr(fmt.Errorf("no plugins installed."))
			}
			cmdutil.DefaultSubCommandRun(err)(cmd, args)
		},
	}

	if len(loadedPlugins) > 0 {
		pluginRunner := f.PluginRunner()
		for _, p := range loadedPlugins {
			cmd.AddCommand(NewCmdForPlugin(f, p, pluginRunner, in, out, err))
		}
	}

	return cmd
}

// NewCmdForPlugin creates a command capable of running the provided plugin.
func NewCmdForPlugin(f cmdutil.Factory, plugin *plugins.Plugin, runner plugins.PluginRunner, in io.Reader, out, errout io.Writer) *cobra.Command {
	if !plugin.IsValid() {
		return nil
	}

	cmd := &cobra.Command{
		Use:     plugin.Name,
		Short:   plugin.ShortDesc,
		Long:    templates.LongDesc(plugin.LongDesc),
		Example: templates.Examples(plugin.Example),
		Run: func(cmd *cobra.Command, args []string) {
			if len(plugin.Command) == 0 {
				cmdutil.DefaultSubCommandRun(errout)(cmd, args)
				return
			}

			envProvider := &plugins.MultiEnvProvider{
				&plugins.PluginCallerEnvProvider{},
				&plugins.OSEnvProvider{},
				&plugins.PluginDescriptorEnvProvider{
					Plugin: plugin,
				},
				&flagsPluginEnvProvider{
					cmd: cmd,
				},
				&factoryAttrsPluginEnvProvider{
					factory: f,
				},
			}

			runningContext := plugins.RunningContext{
				In:          in,
				Out:         out,
				ErrOut:      errout,
				Args:        args,
				EnvProvider: envProvider,
				WorkingDir:  plugin.Dir,
			}

			if err := runner.Run(plugin, runningContext); err != nil {
				cmdutil.CheckErr(err)
			}
		},
	}

	for _, flag := range plugin.Flags {
		cmd.Flags().StringP(flag.Name, flag.Shorthand, flag.DefValue, flag.Desc)
	}

	for _, childPlugin := range plugin.Tree {
		cmd.AddCommand(NewCmdForPlugin(f, childPlugin, runner, in, out, errout))
	}

	return cmd
}

type flagsPluginEnvProvider struct {
	cmd *cobra.Command
}

func (p *flagsPluginEnvProvider) Env() (plugins.EnvList, error) {
	globalPrefix := "KUBECTL_PLUGINS_GLOBAL_FLAG_"
	env := plugins.EnvList{}
	p.cmd.InheritedFlags().VisitAll(func(flag *pflag.Flag) {
		env = append(env, plugins.FlagToEnv(flag, globalPrefix))
	})
	localPrefix := "KUBECTL_PLUGINS_LOCAL_FLAG_"
	p.cmd.LocalFlags().VisitAll(func(flag *pflag.Flag) {
		env = append(env, plugins.FlagToEnv(flag, localPrefix))
	})
	return env, nil
}

type factoryAttrsPluginEnvProvider struct {
	factory cmdutil.Factory
}

func (p *factoryAttrsPluginEnvProvider) Env() (plugins.EnvList, error) {
	cmdNamespace, _, err := p.factory.DefaultNamespace()
	if err != nil {
		return plugins.EnvList{}, err
	}
	return plugins.EnvList{
		plugins.Env{N: "KUBECTL_PLUGINS_CURRENT_NAMESPACE", V: cmdNamespace},
	}, nil
}
