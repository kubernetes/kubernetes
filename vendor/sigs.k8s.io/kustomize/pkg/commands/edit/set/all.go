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

package set

import (
	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

// NewCmdSet returns an instance of 'set' subcommand.
func NewCmdSet(fsys fs.FileSystem, v ifc.Validator) *cobra.Command {
	c := &cobra.Command{
		Use:   "set",
		Short: "Sets the value of different fields in kustomization file.",
		Long:  "",
		Example: `
	# Sets the nameprefix field
	kustomize edit set nameprefix <prefix-value>

	# Sets the namesuffix field
	kustomize edit set namesuffix <suffix-value>
`,
		Args: cobra.MinimumNArgs(1),
	}

	c.AddCommand(
		newCmdSetNamePrefix(fsys),
		newCmdSetNameSuffix(fsys),
		newCmdSetNamespace(fsys, v),
		newCmdSetImage(fsys),
	)
	return c
}
