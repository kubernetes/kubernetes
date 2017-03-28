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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	configMapLong = templates.LongDesc(`
		Create a configmap based on a file, directory, or specified literal value.

		A single configmap may package one or more key/value pairs.

		When creating a configmap based on a file, the key will default to the basename of the file, and the value will
		default to the file content.  If the basename is an invalid key, you may specify an alternate key.

		When creating a configmap based on a directory, each file whose basename is a valid key in the directory will be
		packaged into the configmap.  Any directory entries except regular files are ignored (e.g. subdirectories,
		symlinks, devices, pipes, etc).`)

	configMapExample = templates.Examples(`
		  # Create a new configmap named my-config based on folder bar
		  kubectl create configmap my-config --from-file=path/to/bar

		  # Create a new configmap named my-config with specified keys instead of file basenames on disk
		  kubectl create configmap my-config --from-file=key1=/path/to/bar/file1.txt --from-file=key2=/path/to/bar/file2.txt

		  # Create a new configmap named my-config with key1=config1 and key2=config2
		  kubectl create configmap my-config --from-literal=key1=config1 --from-literal=key2=config2

		  # Create a new configmap named my-config from the key=value pairs in the file
		  kubectl create configmap my-config --from-file=path/to/bar

		  # Create a new configmap named my-config from an env file
		  kubectl create configmap my-config --from-env-file=path/to/bar.env`)
)

// ConfigMap is a command to ease creating ConfigMaps.
func NewCmdCreateConfigMap(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "configmap NAME [--from-file=[key=]source] [--from-literal=key1=value1] [--dry-run]",
		Aliases: []string{"cm"},
		Short:   i18n.T("Create a configmap from a local file, directory or literal value"),
		Long:    configMapLong,
		Example: configMapExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateConfigMap(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ConfigMapV1GeneratorName)
	cmd.Flags().StringSlice("from-file", []string{}, "Key file can be specified using its file path, in which case file basename will be used as configmap key, or optionally with a key and file path, in which case the given key will be used.  Specifying a directory will iterate each named file in the directory whose basename is a valid configmap key.")
	cmd.Flags().StringArray("from-literal", []string{}, "Specify a key and literal value to insert in configmap (i.e. mykey=somevalue)")
	cmd.Flags().String("from-env-file", "", "Specify the path to a file to read lines of key=val pairs to create a configmap (i.e. a Docker .env file).")
	return cmd
}

// CreateConfigMap is the implementation of the create configmap command.
func CreateConfigMap(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.ConfigMapV1GeneratorName:
		generator = &kubectl.ConfigMapGeneratorV1{
			Name:           name,
			FileSources:    cmdutil.GetFlagStringSlice(cmd, "from-file"),
			LiteralSources: cmdutil.GetFlagStringArray(cmd, "from-literal"),
			EnvFileSource:  cmdutil.GetFlagString(cmd, "from-env-file"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
