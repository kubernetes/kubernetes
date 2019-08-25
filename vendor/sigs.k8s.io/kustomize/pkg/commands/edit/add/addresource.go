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

package add

import (
	"errors"
	"log"

	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/commands/kustfile"
	"sigs.k8s.io/kustomize/pkg/fs"
)

type addResourceOptions struct {
	resourceFilePaths []string
}

// newCmdAddResource adds the name of a file containing a resource to the kustomization file.
func newCmdAddResource(fsys fs.FileSystem) *cobra.Command {
	var o addResourceOptions

	cmd := &cobra.Command{
		Use:   "resource",
		Short: "Add the name of a file containing a resource to the kustomization file.",
		Example: `
		add resource {filepath}`,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			err = o.Complete(cmd, args)
			if err != nil {
				return err
			}
			return o.RunAddResource(fsys)
		},
	}
	return cmd
}

// Validate validates addResource command.
func (o *addResourceOptions) Validate(args []string) error {
	if len(args) == 0 {
		return errors.New("must specify a resource file")
	}
	o.resourceFilePaths = args
	return nil
}

// Complete completes addResource command.
func (o *addResourceOptions) Complete(cmd *cobra.Command, args []string) error {
	return nil
}

// RunAddResource runs addResource command (do real work).
func (o *addResourceOptions) RunAddResource(fSys fs.FileSystem) error {
	resources, err := globPatterns(fSys, o.resourceFilePaths)
	if err != nil {
		return err
	}
	if len(resources) == 0 {
		return nil
	}

	mf, err := kustfile.NewKustomizationFile(fSys)
	if err != nil {
		return err
	}

	m, err := mf.Read()
	if err != nil {
		return err
	}

	for _, resource := range resources {
		if kustfile.StringInSlice(resource, m.Resources) {
			log.Printf("resource %s already in kustomization file", resource)
			continue
		}
		m.Resources = append(m.Resources, resource)
	}

	return mf.Write(m)
}
