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
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	replaceConfigMapLong = dedent.Dedent(`
		Replace a configmap based on a file, directory, or specified literal value.

		A single configmap may package one or more key/value pairs.

		When replacing a configmap based on a file, the key will default to the basename of the file, and the value will
		default to the file content.  If the basename is an invalid key, you may specify an alternate key.

		When replacing a configmap based on a directory, each file whose basename is a valid key in the directory will be
		packaged into the configmap.  Any directory entries except regular files are ignored (e.g. subdirectories,
		symlinks, devices, pipes, etc).
		`)

	replaceConfigMapExample = dedent.Dedent(`
		  # Replace a configmap named my-config with keys for each file in folder bar
		  kubectl replace configmap my-config --from-file=path/to/bar

		  # Replace a configmap named my-config with specified keys instead of names on disk
		  kubectl replace configmap my-config --from-file=key1=/path/to/bar/file1.txt --from-file=key2=/path/to/bar/file2.txt

		  # Replace a configmap named my-config with key1=config1 and key2=config2
		  kubectl replace configmap my-config --from-literal=key1=config1 --from-literal=key2=config2`)
)

// ConfigMap is a command to ease replacing ConfigMaps.
func NewCmdReplaceConfigMap(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "configmap NAME [--from-file=[key=]source] [--from-literal=key1=value1] [--dry-run]",
		Aliases: []string{"cm"},
		Short:   "Replace a configmap from a local file, directory or literal value",
		Long:    replaceConfigMapLong,
		Example: replaceConfigMapExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ReplaceConfigMap(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddForceReplaceFlags(cmd, false)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddOutputFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmd.Flags().StringSlice("from-file", []string{}, "Key files can be specified using their file path, in which case a default name will be given to them, or optionally with a name and file path, in which case the given name will be used.  Specifying a directory will iterate each named file in the directory that is a valid configmap key.")
	cmd.Flags().StringSlice("from-literal", []string{}, "Specify a key and literal value to insert in configmap (i.e. mykey=somevalue)")
	return cmd
}

// ReplaceConfigMap is the implementation of the replace configmap command.
func ReplaceConfigMap(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	return RunReplaceSubcommand(f, cmd, cmdOut, args, &ReplaceSubcommandOptions{
		Name:         name,
		Resource:     "configmaps",
		OutputFormat: cmdutil.GetFlagString(cmd, "output"),
	})
}
