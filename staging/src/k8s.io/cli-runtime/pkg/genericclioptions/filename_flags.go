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

package genericclioptions

import (
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/cli-runtime/pkg/resource"
)

// FileNameFlags are flags for processing files.
// Usage of this struct by itself is discouraged.
// These flags are composed by ResourceBuilderFlags
// which should be used instead.
type FileNameFlags struct {
	Usage string

	Filenames *[]string
	Kustomize *string
	Recursive *bool
}

// ToOptions creates a new FileNameOptions struct and sets FilenameOptions based on FileNameflags
func (o *FileNameFlags) ToOptions() resource.FilenameOptions {
	options := resource.FilenameOptions{}

	if o == nil {
		return options
	}

	if o.Recursive != nil {
		options.Recursive = *o.Recursive
	}
	if o.Filenames != nil {
		options.Filenames = *o.Filenames
	}
	if o.Kustomize != nil {
		options.Kustomize = *o.Kustomize
	}

	return options
}

// AddFlags binds file name flags to a given flagset
func (o *FileNameFlags) AddFlags(flags *pflag.FlagSet) {
	if o == nil {
		return
	}

	if o.Recursive != nil {
		flags.BoolVarP(o.Recursive, "recursive", "R", *o.Recursive, "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.")
	}
	if o.Filenames != nil {
		flags.StringSliceVarP(o.Filenames, "filename", "f", *o.Filenames, o.Usage)
		annotations := make([]string, 0, len(resource.FileExtensions))
		for _, ext := range resource.FileExtensions {
			annotations = append(annotations, strings.TrimLeft(ext, "."))
		}
		flags.SetAnnotation("filename", cobra.BashCompFilenameExt, annotations)
	}
	if o.Kustomize != nil {
		flags.StringVarP(o.Kustomize, "kustomize", "k", *o.Kustomize,
			"Process a kustomization directory. This flag can't be used together with -f or -R.")
	}
}
