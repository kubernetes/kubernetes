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
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// NewCmdEditConfigMap is a macro command to edit config maps
func NewCmdEditConfigMap(f cmdutil.Factory, cmdOut, errOut io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	cmd := &cobra.Command{
		Use:     "configmap",
		Aliases: []string{"cm"},
		Short:   "Edit a config map object.",
		Long:    "Edit and update a config map object",
		Run: func(cmd *cobra.Command, args []string) {
			RunEditConfigMap(cmd, f, args, cmdOut, errOut, options)
		},
	}

	addEditFlags(cmd, options)
	cmd.Flags().String("config-map-data", "", "If non-empty, specify the name of a data slot in a config map to edit.")
	return cmd
}

// RunEditConfigMap runs the edit command for config maps. It either edits the complete map
// or it edits individual files inside the config map.
func RunEditConfigMap(cmd *cobra.Command, f cmdutil.Factory, args []string, cmdOut, errOut io.Writer, options *resource.FilenameOptions) error {
	dataFile := cmdutil.GetFlagString(cmd, "config-map-data")
	if len(dataFile) == 0 {
		// We need to add the resource type back on to the front
		args = append([]string{"configmap"}, args...)
		return RunEdit(f, cmdOut, errOut, cmd, args, options)
	}
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	cs, err := f.ClientSet()
	if err != nil {
		return err
	}
	configMap, err := cs.Core().ConfigMaps(cmdNamespace).Get(args[0], v1.GetOptions{})
	if err != nil {
		return err
	}

	value, found := configMap.Data[dataFile]
	if !found {
		keys := []string{}
		for key := range configMap.Data {
			keys = append(keys, key)
		}
		return fmt.Errorf("No such data file (%s), filenames are: %v\n", dataFile, keys)
	}
	edit := editor.NewDefaultEditor(os.Environ())
	data, file, err := edit.LaunchTempFile(fmt.Sprintf("%s-edit-", dataFile), "", bytes.NewBuffer([]byte(value)))
	defer func() {
		os.Remove(file)
	}()
	if err != nil {
		return err
	}
	configMap.Data[dataFile] = string(data)

	if _, err := cs.Core().ConfigMaps(cmdNamespace).Update(configMap); err != nil {
		return err
	}
	return nil
}
