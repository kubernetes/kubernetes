/*
Copyright 2018 The Kubernetes Authors.

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
	"errors"

	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/commands/kustfile"
	"sigs.k8s.io/kustomize/pkg/fs"
)

type setNameSuffixOptions struct {
	suffix string
}

// newCmdSetNameSuffix sets the value of the nameSuffix field in the kustomization.
func newCmdSetNameSuffix(fsys fs.FileSystem) *cobra.Command {
	var o setNameSuffixOptions

	cmd := &cobra.Command{
		Use:   "namesuffix",
		Short: "Sets the value of the nameSuffix field in the kustomization file.",
		Example: `
The command
  set namesuffix -- -acme
will add the field "nameSuffix: -acme" to the kustomization file if it doesn't exist,
and overwrite the value with "-acme" if the field does exist.
`,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			err = o.Complete(cmd, args)
			if err != nil {
				return err
			}
			return o.RunSetNameSuffix(fsys)
		},
	}
	return cmd
}

// Validate validates setNameSuffix command.
func (o *setNameSuffixOptions) Validate(args []string) error {
	if len(args) != 1 {
		return errors.New("must specify exactly one suffix value")
	}
	// TODO: add further validation on the value.
	o.suffix = args[0]
	return nil
}

// Complete completes setNameSuffix command.
func (o *setNameSuffixOptions) Complete(cmd *cobra.Command, args []string) error {
	return nil
}

// RunSetNameSuffix runs setNameSuffix command (does real work).
func (o *setNameSuffixOptions) RunSetNameSuffix(fSys fs.FileSystem) error {
	mf, err := kustfile.NewKustomizationFile(fSys)
	if err != nil {
		return err
	}
	m, err := mf.Read()
	if err != nil {
		return err
	}
	m.NameSuffix = o.suffix
	return mf.Write(m)
}
