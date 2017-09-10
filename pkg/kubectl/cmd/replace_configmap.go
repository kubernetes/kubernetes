/*
Copyright 2014 The Kubernetes Authors.

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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	replaceConfigMapLong = templates.LongDesc(i18n.T(`
		Replace a configmap based on a file, directory, or specified literal value.

		A single configmap may package one or more key/value pairs.

		When replacing a configmap based on a file, the key will default to the basename of the file, and the value will
		default to the file content.  If the basename is an invalid key, you may specify an alternate key.

		When replacing a configmap based on a directory, each file whose basename is a valid key in the directory will be
		packaged into the configmap.  Any directory entries except regular files are ignored (e.g. subdirectories,
		symlinks, devices, pipes, etc).`))

	replaceConfigMapExample = templates.Examples(i18n.T(`
		  # Replace a configmap named my-config based on folder bar
		  kubectl replace configmap my-config --from-file=path/to/bar

		  # Replace a configmap named my-config with specified keys instead of file basenames on disk
		  kubectl replace configmap my-config --from-file=key1=/path/to/bar/file1.txt --from-file=key2=/path/to/bar/file2.txt

		  # Replace a configmap named my-config with key1=config1 and key2=config2
		  kubectl replace configmap my-config --from-literal=key1=config1 --from-literal=key2=config2

		  # Replace a configmap named my-config from the key=value pairs in the file
		  kubectl replace configmap my-config --from-file=path/to/bar

		  # Replace a configmap named my-config from an env file
		  kubectl replace configmap my-config --from-env-file=path/to/bar.env`))
)

// ConfigMap is a command to ease replacing ConfigMaps.
func NewCmdReplaceConfigMap(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "configmap NAME [--from-file=[key=]source] [--from-literal=key1=value1]",
		Aliases: []string{"cm"},
		Short:   i18n.T("Replace a configmap from a local file, directory or literal value"),
		Long:    replaceConfigMapLong,
		Example: replaceConfigMapExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ReplaceConfigMap(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ConfigMapV1GeneratorName)
	cmd.Flags().StringSlice("from-file", []string{}, "Key file can be specified using its file path, in which case file basename will be used as configmap key, or optionally with a key and file path, in which case the given key will be used.  Specifying a directory will iterate each named file in the directory whose basename is a valid configmap key.")
	cmd.Flags().StringArray("from-literal", []string{}, "Specify a key and literal value to insert in configmap (i.e. mykey=somevalue)")
	cmd.Flags().String("from-env-file", "", "Specify the path to a file to read lines of key=val pairs to replace a configmap (i.e. a Docker .env file).")
	cmd.Flags().Bool("append-hash", false, "Append a hash of the configmap to its name.")
	return cmd
}

// ReplaceConfigMap is the implementation of the replace configmap command.
func ReplaceConfigMap(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
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
			AppendHash:     cmdutil.GetFlagBool(cmd, "append-hash"),
		}
	default:
		return errUnsupportedGenerator(cmd, generatorName)
	}
	return RunReplaceSubcommand(f, cmd, cmdOut, &ReplaceSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}
