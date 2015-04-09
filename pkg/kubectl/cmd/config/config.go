/*
Copyright 2014 Google Inc. All rights reserved.

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

package config

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"reflect"
	"strconv"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

type PathOptions struct {
	Local     bool
	Global    bool
	UseEnvVar bool

	LocalFile  string
	GlobalFile string
	EnvVarFile string

	EnvVar           string
	ExplicitFileFlag string

	LoadingRules *clientcmd.ClientConfigLoadingRules
}

func NewCmdConfig(pathOptions *PathOptions, out io.Writer) *cobra.Command {
	if len(pathOptions.ExplicitFileFlag) == 0 {
		pathOptions.ExplicitFileFlag = clientcmd.RecommendedConfigPathFlag
	}
	if len(pathOptions.EnvVar) > 0 {
		pathOptions.EnvVarFile = os.Getenv(pathOptions.EnvVar)
	}

	cmd := &cobra.Command{
		Use:   "config SUBCOMMAND",
		Short: "config modifies kubeconfig files",
		Long:  `config modifies kubeconfig files using subcommands like "kubectl config set current-context my-context"`,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	// file paths are common to all sub commands
	cmd.PersistentFlags().BoolVar(&pathOptions.Local, "local", pathOptions.Local, "use the kubeconfig in the current directory")
	cmd.PersistentFlags().BoolVar(&pathOptions.Global, "global", pathOptions.Global, "use the kubeconfig from "+pathOptions.GlobalFile)
	cmd.PersistentFlags().BoolVar(&pathOptions.UseEnvVar, "envvar", pathOptions.UseEnvVar, "use the kubeconfig from $"+pathOptions.EnvVar)
	cmd.PersistentFlags().StringVar(&pathOptions.LoadingRules.ExplicitPath, pathOptions.ExplicitFileFlag, pathOptions.LoadingRules.ExplicitPath, "use a particular kubeconfig file")

	cmd.AddCommand(NewCmdConfigView(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetCluster(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetAuthInfo(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSetContext(out, pathOptions))
	cmd.AddCommand(NewCmdConfigSet(out, pathOptions))
	cmd.AddCommand(NewCmdConfigUnset(out, pathOptions))
	cmd.AddCommand(NewCmdConfigUseContext(out, pathOptions))

	return cmd
}

func NewDefaultPathOptions() *PathOptions {
	ret := &PathOptions{
		LocalFile:  ".kubeconfig",
		GlobalFile: path.Join(os.Getenv("HOME"), "/.kube/.kubeconfig"),
		EnvVar:     clientcmd.RecommendedConfigPathEnvVar,
		EnvVarFile: os.Getenv(clientcmd.RecommendedConfigPathEnvVar),

		ExplicitFileFlag: clientcmd.RecommendedConfigPathFlag,

		LoadingRules: clientcmd.NewDefaultClientConfigLoadingRules(),
	}
	ret.LoadingRules.DoNotResolvePaths = true

	return ret
}

func (o PathOptions) Validate() error {
	if len(o.LoadingRules.ExplicitPath) > 0 {
		if o.Global {
			return errors.New("cannot specify both --" + o.ExplicitFileFlag + " and --global")
		}
		if o.Local {
			return errors.New("cannot specify both --" + o.ExplicitFileFlag + " and --local")
		}
		if o.UseEnvVar {
			return errors.New("cannot specify both --" + o.ExplicitFileFlag + " and --envvar")
		}
	}

	if o.Global {
		if o.Local {
			return errors.New("cannot specify both --global and --local")
		}
		if o.UseEnvVar {
			return errors.New("cannot specify both --global and --envvar")
		}
	}

	if o.Local {
		if o.UseEnvVar {
			return errors.New("cannot specify both --local and --envvar")
		}
	}

	if o.UseEnvVar {
		if len(o.EnvVarFile) == 0 {
			return fmt.Errorf("environment variable %v does not have a value", o.EnvVar)
		}

	}

	return nil
}

func (o *PathOptions) getStartingConfig() (*clientcmdapi.Config, error) {
	if err := o.Validate(); err != nil {
		return nil, err
	}

	config := clientcmdapi.NewConfig()

	switch {
	case o.Global:
		config = getConfigFromFileOrDie(o.GlobalFile)

	case o.UseEnvVar:
		config = getConfigFromFileOrDie(o.EnvVarFile)

	case o.Local:
		config = getConfigFromFileOrDie(o.LocalFile)

		// no specific flag was set, load according to the loading rules
	default:
		clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(o.LoadingRules, &clientcmd.ConfigOverrides{})
		rawConfig, err := clientConfig.RawConfig()
		if err != nil {
			return nil, err
		}
		config = &rawConfig

	}

	return config, nil
}

// GetDefaultFilename returns the name of the file you should write into (create if necessary), if you're trying to create
// a new stanza as opposed to updating an existing one.
func (o *PathOptions) GetDefaultFilename() string {
	if o.IsExplicitFile() {
		return o.GetExplicitFile()
	}

	if len(o.EnvVarFile) > 0 {
		return o.EnvVarFile
	}

	if _, err := os.Stat(o.LocalFile); err == nil {
		return o.LocalFile
	}

	return o.GlobalFile

}

func (o *PathOptions) IsExplicitFile() bool {
	switch {
	case len(o.LoadingRules.ExplicitPath) > 0 ||
		o.Global ||
		o.UseEnvVar ||
		o.Local:
		return true
	}
	return false
}

func (o *PathOptions) GetExplicitFile() string {
	if !o.IsExplicitFile() {
		return ""
	}

	switch {
	case len(o.LoadingRules.ExplicitPath) > 0:
		return o.LoadingRules.ExplicitPath

	case o.Global:
		return o.GlobalFile

	case o.UseEnvVar:
		return o.EnvVarFile

	case o.Local:
		return o.LocalFile
	}

	return ""
}

// ModifyConfig takes a Config object, iterates through Clusters, AuthInfos, and Contexts, uses the LocationOfOrigin if specified or
// uses the default destination file to write the results into.  This results in multiple file reads, but it's very easy to follow.
// Preferences and CurrentContext should always be set in the default destination file.  Since we can't distinguish between empty and missing values
// (no nil strings), we're forced have separate handling for them.  In all the currently known cases, newConfig should have, at most, one difference,
// that means that this code will only write into a single file.
func (o *PathOptions) ModifyConfig(newConfig clientcmdapi.Config) error {
	startingConfig, err := o.getStartingConfig()
	if err != nil {
		return err
	}

	// at this point, config and startingConfig should have, at most, one difference.  We need to chase the difference until we find it
	// then we'll build a partial config object to call write upon.  Special case the test for current context and preferences since those
	// always write to the default file.
	switch {
	case reflect.DeepEqual(*startingConfig, newConfig):
		// nothing to do

	case startingConfig.CurrentContext != newConfig.CurrentContext:
		if err := o.writeCurrentContext(newConfig.CurrentContext); err != nil {
			return err
		}

	case !reflect.DeepEqual(startingConfig.Preferences, newConfig.Preferences):
		if err := o.writePreferences(newConfig.Preferences); err != nil {
			return err
		}

	default:
		// something is different. Search every cluster, authInfo, and context.  First from new to old for differences, then from old to new for deletions
		for key, cluster := range newConfig.Clusters {
			startingCluster, exists := startingConfig.Clusters[key]
			if !reflect.DeepEqual(cluster, startingCluster) || !exists {
				destinationFile := cluster.LocationOfOrigin
				if len(destinationFile) == 0 {
					destinationFile = o.GetDefaultFilename()
				}

				configToWrite := getConfigFromFileOrDie(destinationFile)
				configToWrite.Clusters[key] = cluster

				if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
					return err
				}
			}
		}

		for key, context := range newConfig.Contexts {
			startingContext, exists := startingConfig.Contexts[key]
			if !reflect.DeepEqual(context, startingContext) || !exists {
				destinationFile := context.LocationOfOrigin
				if len(destinationFile) == 0 {
					destinationFile = o.GetDefaultFilename()
				}

				configToWrite := getConfigFromFileOrDie(destinationFile)
				configToWrite.Contexts[key] = context

				if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
					return err
				}
			}
		}

		for key, authInfo := range newConfig.AuthInfos {
			startingAuthInfo, exists := startingConfig.AuthInfos[key]
			if !reflect.DeepEqual(authInfo, startingAuthInfo) || !exists {
				destinationFile := authInfo.LocationOfOrigin
				if len(destinationFile) == 0 {
					destinationFile = o.GetDefaultFilename()
				}

				configToWrite := getConfigFromFileOrDie(destinationFile)
				configToWrite.AuthInfos[key] = authInfo

				if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
					return err
				}
			}
		}

		for key, cluster := range startingConfig.Clusters {
			if _, exists := newConfig.Clusters[key]; !exists {
				destinationFile := cluster.LocationOfOrigin
				if len(destinationFile) == 0 {
					destinationFile = o.GetDefaultFilename()
				}

				configToWrite := getConfigFromFileOrDie(destinationFile)
				delete(configToWrite.Clusters, key)

				if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
					return err
				}
			}
		}

		for key, context := range startingConfig.Contexts {
			if _, exists := newConfig.Contexts[key]; !exists {
				destinationFile := context.LocationOfOrigin
				if len(destinationFile) == 0 {
					destinationFile = o.GetDefaultFilename()
				}

				configToWrite := getConfigFromFileOrDie(destinationFile)
				delete(configToWrite.Contexts, key)

				if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
					return err
				}
			}
		}

		for key, authInfo := range startingConfig.AuthInfos {
			if _, exists := newConfig.AuthInfos[key]; !exists {
				destinationFile := authInfo.LocationOfOrigin
				if len(destinationFile) == 0 {
					destinationFile = o.GetDefaultFilename()
				}

				configToWrite := getConfigFromFileOrDie(destinationFile)
				delete(configToWrite.AuthInfos, key)

				if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
					return err
				}
			}
		}

	}

	return nil
}

// writeCurrentContext takes three possible paths.
// If newCurrentContext is the same as the startingConfig's current context, then we exit.
// If newCurrentContext has a value, then that value is written into the default destination file.
// If newCurrentContext is empty, then we find the config file that is setting the CurrentContext and clear the value from that file
func (o *PathOptions) writeCurrentContext(newCurrentContext string) error {
	if startingConfig, err := o.getStartingConfig(); err != nil {
		return err
	} else if startingConfig.CurrentContext == newCurrentContext {
		return nil
	}

	if len(newCurrentContext) > 0 {
		destinationFile := o.GetDefaultFilename()
		config := getConfigFromFileOrDie(destinationFile)
		config.CurrentContext = newCurrentContext

		if err := clientcmd.WriteToFile(*config, destinationFile); err != nil {
			return err
		}

		return nil
	}

	if o.IsExplicitFile() {
		file := o.GetExplicitFile()
		currConfig := getConfigFromFileOrDie(file)
		currConfig.CurrentContext = newCurrentContext
		if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
			return err
		}

		return nil
	}

	filesToCheck := make([]string, 0, len(o.LoadingRules.Precedence)+1)
	filesToCheck = append(filesToCheck, o.LoadingRules.ExplicitPath)
	filesToCheck = append(filesToCheck, o.LoadingRules.Precedence...)

	for _, file := range filesToCheck {
		currConfig := getConfigFromFileOrDie(file)

		if len(currConfig.CurrentContext) > 0 {
			currConfig.CurrentContext = newCurrentContext
			if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
				return err
			}

			return nil
		}
	}

	return nil
}

func (o *PathOptions) writePreferences(newPrefs clientcmdapi.Preferences) error {
	if startingConfig, err := o.getStartingConfig(); err != nil {
		return err
	} else if reflect.DeepEqual(startingConfig.Preferences, newPrefs) {
		return nil
	}

	if o.IsExplicitFile() {
		file := o.GetExplicitFile()
		currConfig := getConfigFromFileOrDie(file)
		currConfig.Preferences = newPrefs
		if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
			return err
		}

		return nil
	}

	filesToCheck := make([]string, 0, len(o.LoadingRules.Precedence)+1)
	filesToCheck = append(filesToCheck, o.LoadingRules.ExplicitPath)
	filesToCheck = append(filesToCheck, o.LoadingRules.Precedence...)

	for _, file := range filesToCheck {
		currConfig := getConfigFromFileOrDie(file)

		if !reflect.DeepEqual(currConfig.Preferences, newPrefs) {
			currConfig.Preferences = newPrefs
			if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
				return err
			}

			return nil
		}
	}

	return nil
}

// getConfigFromFileOrDie tries to read a kubeconfig file and if it can't, it calls exit.  One exception, missing files result in empty configs, not an exit
func getConfigFromFileOrDie(filename string) *clientcmdapi.Config {
	config, err := clientcmd.LoadFromFile(filename)
	if err != nil && !os.IsNotExist(err) {
		glog.FatalDepth(1, err)
	}

	if config == nil {
		return clientcmdapi.NewConfig()
	}

	return config
}

func toBool(propertyValue string) (bool, error) {
	boolValue := false
	if len(propertyValue) != 0 {
		var err error
		boolValue, err = strconv.ParseBool(propertyValue)
		if err != nil {
			return false, err
		}
	}

	return boolValue, nil
}
