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
	"sigs.k8s.io/kustomize/pkg/patch"
)

type addPatchOptions struct {
	patchFilePaths []string
}

// newCmdAddPatch adds the name of a file containing a patch to the kustomization file.
func newCmdAddPatch(fsys fs.FileSystem) *cobra.Command {
	var o addPatchOptions

	cmd := &cobra.Command{
		Use:   "patch",
		Short: "Add the name of a file containing a patch to the kustomization file.",
		Example: `
		add patch {filepath}`,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			err = o.Complete(cmd, args)
			if err != nil {
				return err
			}
			return o.RunAddPatch(fsys)
		},
	}
	return cmd
}

// Validate validates addPatch command.
func (o *addPatchOptions) Validate(args []string) error {
	if len(args) == 0 {
		return errors.New("must specify a patch file")
	}
	o.patchFilePaths = args
	return nil
}

// Complete completes addPatch command.
func (o *addPatchOptions) Complete(cmd *cobra.Command, args []string) error {
	return nil
}

// RunAddPatch runs addPatch command (do real work).
func (o *addPatchOptions) RunAddPatch(fSys fs.FileSystem) error {
	patches, err := globPatterns(fSys, o.patchFilePaths)
	if err != nil {
		return err
	}
	if len(patches) == 0 {
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

	for _, p := range patches {
		if patch.Exist(m.PatchesStrategicMerge, p) {
			log.Printf("patch %s already in kustomization file", p)
			continue
		}
		m.PatchesStrategicMerge = patch.Append(m.PatchesStrategicMerge, p)
	}

	return mf.Write(m)
}
