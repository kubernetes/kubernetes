/*
Copyright 2019 The Kubernetes Authors.

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

package kustomize

import (
	"errors"
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/kustomize"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	"sigs.k8s.io/kustomize/pkg/fs"
)

type kustomizeOptions struct {
	kustomizationDir string
}

var (
	kustomizeLong = templates.LongDesc(i18n.T(`
Print a set of API resources generated from instructions in a kustomization.yaml file.

The argument must be the path to the directory containing
the file, or a git repository
URL with a path suffix specifying same with respect to the
repository root.

  kubectl kustomize somedir
	`))

	kustomizeExample = templates.Examples(i18n.T(`
# Use the current working directory
  kubectl kustomize .

# Use some shared configuration directory
  kubectl kustomize /home/configuration/production

# Use a URL
  kubectl kustomize github.com/kubernetes-sigs/kustomize.git/examples/helloWorld?ref=v1.0.6
`))
)

// NewCmdKustomize returns a kustomize command
func NewCmdKustomize(streams genericclioptions.IOStreams) *cobra.Command {
	var o kustomizeOptions

	cmd := &cobra.Command{
		Use:     "kustomize <dir>",
		Short:   i18n.T("Build a kustomization target from a directory or a remote url."),
		Long:    kustomizeLong,
		Example: kustomizeExample,

		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			return kustomize.RunKustomizeBuild(streams.Out, fs.MakeRealFS(), o.kustomizationDir)
		},
	}

	return cmd
}

// Validate validates build command.
func (o *kustomizeOptions) Validate(args []string) error {
	if len(args) > 1 {
		return errors.New("specify one path to a kustomization directory")
	}
	if len(args) == 0 {
		o.kustomizationDir = "./"
	} else {
		o.kustomizationDir = args[0]
	}

	return nil
}
