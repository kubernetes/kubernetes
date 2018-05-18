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

package wait

import (
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
)

// ResourceBuilderFlags are flags for finding resources
type ResourceBuilderFlags struct {
	FilenameOptions resource.FilenameOptions

	LabelSelector     string
	FieldSelector     string
	AllNamespaces     bool
	Namespace         string
	ExplicitNamespace bool

	// TODO add conditional support.  These are false for now.
	All   bool
	Local bool
}

// NewResourceBuilderFlags returns a default ResourceBuilderFlags
func NewResourceBuilderFlags() *ResourceBuilderFlags {
	return &ResourceBuilderFlags{
		FilenameOptions: resource.FilenameOptions{
			Recursive: true,
		},
	}
}

// AddFlags registers flags for finding resources
func (o *ResourceBuilderFlags) AddFlags(flagset *pflag.FlagSet) {
	flagset.StringSliceVarP(&o.FilenameOptions.Filenames, "filename", "f", o.FilenameOptions.Filenames, "Filename, directory, or URL to files identifying the resource.")
	annotations := make([]string, 0, len(resource.FileExtensions))
	for _, ext := range resource.FileExtensions {
		annotations = append(annotations, strings.TrimLeft(ext, "."))
	}
	flagset.SetAnnotation("filename", cobra.BashCompFilenameExt, annotations)
	flagset.BoolVar(&o.FilenameOptions.Recursive, "recursive", o.FilenameOptions.Recursive, "Process the directory used in -f, --filename recursively. Useful when you want to manage related manifests organized within the same directory.")

	flagset.StringVarP(&o.LabelSelector, "selector", "l", o.LabelSelector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	flagset.StringVar(&o.FieldSelector, "field-selector", o.FieldSelector, "Selector (field query) to filter on, supports '=', '==', and '!='.(e.g. --field-selector key1=value1,key2=value2). The server only supports a limited number of field queries per type.")
	flagset.BoolVar(&o.AllNamespaces, "all-namespaces", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
}

// ToBuilder gives you back a resource finder to visit resources that are located
func (o *ResourceBuilderFlags) ToBuilder(restClientGetter genericclioptions.RESTClientGetter, resources []string) ResourceFinder {
	namespace, enforceNamespace, namespaceErr := restClientGetter.ToRawKubeConfigLoader().Namespace()

	return &ResourceFindBuilderWrapper{
		builder: resource.NewBuilder(restClientGetter).
			Unstructured().
			NamespaceParam(namespace).DefaultNamespace().
			FilenameParam(enforceNamespace, &o.FilenameOptions).
			LabelSelectorParam(o.LabelSelector).
			FieldSelectorParam(o.FieldSelector).
			ResourceTypeOrNameArgs(o.All, resources...).
			Latest().
			Flatten().
			AddError(namespaceErr),
	}
}

// ResourceFindBuilderWrapper wraps a builder in an interface
type ResourceFindBuilderWrapper struct {
	builder *resource.Builder
}

// Do finds you resources to check
func (b *ResourceFindBuilderWrapper) Do() resource.Visitor {
	return b.builder.Do()
}

// ResourceFinder allows mocking the resource builder
// TODO resource builders needs to become more interfacey
type ResourceFinder interface {
	Do() resource.Visitor
}

// ResourceFinderFunc is a handy way to make a  ResourceFinder
type ResourceFinderFunc func() resource.Visitor

// Do implements ResourceFinder
func (fn ResourceFinderFunc) Do() resource.Visitor {
	return fn()
}

// ResourceFinderForResult skins a visitor for re-use as a ResourceFinder
func ResourceFinderForResult(result resource.Visitor) ResourceFinder {
	return ResourceFinderFunc(func() resource.Visitor {
		return result
	})
}
