/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strconv"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

type PathOptions struct {
	// GlobalFile is the full path to the file to load as the global (final) option
	GlobalFile string
	// EnvVar is the env var name that points to the list of kubeconfig files to load
	EnvVar string
	// ExplicitFileFlag is the name of the flag to use for prompting for the kubeconfig file
	ExplicitFileFlag string

	// GlobalFileSubpath is an optional value used for displaying help
	GlobalFileSubpath string

	LoadingRules *clientcmd.ClientConfigLoadingRules
}

// ConfigAccess is used by subcommands and methods in this package to load and modify the appropriate config files
type ConfigAccess interface {
	// GetLoadingPrecedence returns the slice of files that should be used for loading and inspecting the config
	GetLoadingPrecedence() []string
	// GetStartingConfig returns the config that subcommands should being operating against.  It may or may not be merged depending on loading rules
	GetStartingConfig() (*clientcmdapi.Config, error)
	// GetDefaultFilename returns the name of the file you should write into (create if necessary), if you're trying to create a new stanza as opposed to updating an existing one.
	GetDefaultFilename() string
	// IsExplicitFile indicates whether or not this command is interested in exactly one file.  This implementation only ever does that  via a flag, but implementations that handle local, global, and flags may have more
	IsExplicitFile() bool
	// GetExplicitFile returns the particular file this command is operating against.  This implementation only ever has one, but implementations that handle local, global, and flags may have more
	GetExplicitFile() string
}

func NewCmdConfig(pathOptions *PathOptions, out io.Writer) *cobra.Command {
	if len(pathOptions.ExplicitFileFlag) == 0 {
		pathOptions.ExplicitFileFlag = clientcmd.RecommendedConfigPathFlag
	}

	cmd := &cobra.Command{
		Use:   "config SUBCOMMAND",
		Short: "config modifies kubeconfig files",
		Long: `config modifies kubeconfig files using subcommands like "kubectl config set current-context my-context"

The loading order follows these rules:
    1. If the --` + pathOptions.ExplicitFileFlag + ` flag is set, then only that file is loaded.  The flag may only be set once and no merging takes place.
    2. If $` + pathOptions.EnvVar + ` environment variable is set, then it is used a list of paths (normal path delimitting rules for your system).  These paths are merged together.  When a value is modified, it is modified in the file that defines the stanza.  When a value is created, it is created in the first file that exists.  If no files in the chain exist, then it creates the last file in the list.
    3. Otherwise, ` + path.Join("${HOME}", pathOptions.GlobalFileSubpath) + ` is used and no merging takes place.
`,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	// file paths are common to all sub commands
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
		GlobalFile:       clientcmd.RecommendedHomeFile,
		EnvVar:           clientcmd.RecommendedConfigPathEnvVar,
		ExplicitFileFlag: clientcmd.RecommendedConfigPathFlag,

		GlobalFileSubpath: path.Join(clientcmd.RecommendedHomeDir, clientcmd.RecommendedFileName),

		LoadingRules: clientcmd.NewDefaultClientConfigLoadingRules(),
	}
	ret.LoadingRules.DoNotResolvePaths = true

	return ret
}

func (o *PathOptions) GetEnvVarFiles() []string {
	if len(o.EnvVar) == 0 {
		return []string{}
	}

	envVarValue := os.Getenv(o.EnvVar)
	if len(envVarValue) == 0 {
		return []string{}
	}

	return filepath.SplitList(envVarValue)
}

func (o *PathOptions) GetLoadingPrecedence() []string {
	if envVarFiles := o.GetEnvVarFiles(); len(envVarFiles) > 0 {
		return envVarFiles
	}

	return []string{o.GlobalFile}
}

func (o *PathOptions) GetStartingConfig() (*clientcmdapi.Config, error) {
	// don't mutate the original
	loadingRules := *o.LoadingRules
	loadingRules.Precedence = o.GetLoadingPrecedence()

	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(&loadingRules, &clientcmd.ConfigOverrides{})
	rawConfig, err := clientConfig.RawConfig()
	if os.IsNotExist(err) {
		return clientcmdapi.NewConfig(), nil
	}
	if err != nil {
		return nil, err
	}

	return &rawConfig, nil
}

func (o *PathOptions) GetDefaultFilename() string {
	if o.IsExplicitFile() {
		return o.GetExplicitFile()
	}

	if envVarFiles := o.GetEnvVarFiles(); len(envVarFiles) > 0 {
		if len(envVarFiles) == 1 {
			return envVarFiles[0]
		}

		// if any of the envvar files already exists, return it
		for _, envVarFile := range envVarFiles {
			if _, err := os.Stat(envVarFile); err == nil {
				return envVarFile
			}
		}

		// otherwise, return the last one in the list
		return envVarFiles[len(envVarFiles)-1]
	}

	return o.GlobalFile
}

func (o *PathOptions) IsExplicitFile() bool {
	if len(o.LoadingRules.ExplicitPath) > 0 {
		return true
	}

	return false
}

func (o *PathOptions) GetExplicitFile() string {
	return o.LoadingRules.ExplicitPath
}

// ModifyConfig takes a Config object, iterates through Clusters, AuthInfos, and Contexts, uses the LocationOfOrigin if specified or
// uses the default destination file to write the results into.  This results in multiple file reads, but it's very easy to follow.
// Preferences and CurrentContext should always be set in the default destination file.  Since we can't distinguish between empty and missing values
// (no nil strings), we're forced have separate handling for them.  In the kubeconfig cases, newConfig should have at most one difference,
// that means that this code will only write into a single file.  If you want to relativizePaths, you must provide a fully qualified path in any
// modified element.
func ModifyConfig(configAccess ConfigAccess, newConfig clientcmdapi.Config, relativizePaths bool) error {
	startingConfig, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	// We need to find all differences, locate their original files, read a partial config to modify only that stanza and write out the file.
	// Special case the test for current context and preferences since those always write to the default file.
	if reflect.DeepEqual(*startingConfig, newConfig) {
		// nothing to do
		return nil
	}

	if startingConfig.CurrentContext != newConfig.CurrentContext {
		if err := writeCurrentContext(configAccess, newConfig.CurrentContext); err != nil {
			return err
		}
	}

	if !reflect.DeepEqual(startingConfig.Preferences, newConfig.Preferences) {
		if err := writePreferences(configAccess, newConfig.Preferences); err != nil {
			return err
		}
	}

	// Search every cluster, authInfo, and context.  First from new to old for differences, then from old to new for deletions
	for key, cluster := range newConfig.Clusters {
		startingCluster, exists := startingConfig.Clusters[key]
		if !reflect.DeepEqual(cluster, startingCluster) || !exists {
			destinationFile := cluster.LocationOfOrigin
			if len(destinationFile) == 0 {
				destinationFile = configAccess.GetDefaultFilename()
			}

			configToWrite := getConfigFromFileOrDie(destinationFile)
			t := *cluster
			configToWrite.Clusters[key] = &t
			configToWrite.Clusters[key].LocationOfOrigin = destinationFile
			if relativizePaths {
				if err := clientcmd.RelativizeClusterLocalPaths(configToWrite.Clusters[key]); err != nil {
					return err
				}
			}

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
				destinationFile = configAccess.GetDefaultFilename()
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
				destinationFile = configAccess.GetDefaultFilename()
			}

			configToWrite := getConfigFromFileOrDie(destinationFile)
			t := *authInfo
			configToWrite.AuthInfos[key] = &t
			configToWrite.AuthInfos[key].LocationOfOrigin = destinationFile
			if relativizePaths {
				if err := clientcmd.RelativizeAuthInfoLocalPaths(configToWrite.AuthInfos[key]); err != nil {
					return err
				}
			}

			if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
				return err
			}
		}
	}

	for key, cluster := range startingConfig.Clusters {
		if _, exists := newConfig.Clusters[key]; !exists {
			destinationFile := cluster.LocationOfOrigin
			if len(destinationFile) == 0 {
				destinationFile = configAccess.GetDefaultFilename()
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
				destinationFile = configAccess.GetDefaultFilename()
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
				destinationFile = configAccess.GetDefaultFilename()
			}

			configToWrite := getConfigFromFileOrDie(destinationFile)
			delete(configToWrite.AuthInfos, key)

			if err := clientcmd.WriteToFile(*configToWrite, destinationFile); err != nil {
				return err
			}
		}
	}

	return nil
}

// writeCurrentContext takes three possible paths.
// If newCurrentContext is the same as the startingConfig's current context, then we exit.
// If newCurrentContext has a value, then that value is written into the default destination file.
// If newCurrentContext is empty, then we find the config file that is setting the CurrentContext and clear the value from that file
func writeCurrentContext(configAccess ConfigAccess, newCurrentContext string) error {
	if startingConfig, err := configAccess.GetStartingConfig(); err != nil {
		return err
	} else if startingConfig.CurrentContext == newCurrentContext {
		return nil
	}

	if configAccess.IsExplicitFile() {
		file := configAccess.GetExplicitFile()
		currConfig := getConfigFromFileOrDie(file)
		currConfig.CurrentContext = newCurrentContext
		if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
			return err
		}

		return nil
	}

	if len(newCurrentContext) > 0 {
		destinationFile := configAccess.GetDefaultFilename()
		config := getConfigFromFileOrDie(destinationFile)
		config.CurrentContext = newCurrentContext

		if err := clientcmd.WriteToFile(*config, destinationFile); err != nil {
			return err
		}

		return nil
	}

	// we're supposed to be clearing the current context.  We need to find the first spot in the chain that is setting it and clear it
	for _, file := range configAccess.GetLoadingPrecedence() {
		if _, err := os.Stat(file); err == nil {
			currConfig := getConfigFromFileOrDie(file)

			if len(currConfig.CurrentContext) > 0 {
				currConfig.CurrentContext = newCurrentContext
				if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
					return err
				}

				return nil
			}
		}
	}

	return errors.New("no config found to write context")
}

func writePreferences(configAccess ConfigAccess, newPrefs clientcmdapi.Preferences) error {
	if startingConfig, err := configAccess.GetStartingConfig(); err != nil {
		return err
	} else if reflect.DeepEqual(startingConfig.Preferences, newPrefs) {
		return nil
	}

	if configAccess.IsExplicitFile() {
		file := configAccess.GetExplicitFile()
		currConfig := getConfigFromFileOrDie(file)
		currConfig.Preferences = newPrefs
		if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
			return err
		}

		return nil
	}

	for _, file := range configAccess.GetLoadingPrecedence() {
		currConfig := getConfigFromFileOrDie(file)

		if !reflect.DeepEqual(currConfig.Preferences, newPrefs) {
			currConfig.Preferences = newPrefs
			if err := clientcmd.WriteToFile(*currConfig, file); err != nil {
				return err
			}

			return nil
		}
	}

	return errors.New("no config found to write preferences")
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
