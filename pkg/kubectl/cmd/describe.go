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
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
)

var (
	describeLong = templates.LongDesc(`
		Show details of a specific resource or group of resources

		Print a detailed description of the selected resources, including related resources such
		as events or controllers. You may select a single object by name, all objects of that 
		type, provide a name prefix, or label selector. For example:

		    $ kubectl describe TYPE NAME_PREFIX

		will first check for an exact match on TYPE and NAME_PREFIX. If no such resource
		exists, it will output details for every resource that has a name prefixed with NAME_PREFIX.`)

	describeExample = templates.Examples(i18n.T(`
		# Describe a node
		kubectl describe nodes kubernetes-node-emt8.c.myproject.internal

		# Describe a pod
		kubectl describe pods/nginx

		# Describe a pod identified by type and name in "pod.json"
		kubectl describe -f pod.json

		# Describe all pods
		kubectl describe pods

		# Describe pods by label name=myLabel
		kubectl describe po -l name=myLabel

		# Describe all pods managed by the 'frontend' replication controller (rc-created pods
		# get the name of the rc as a prefix in the pod the name).
		kubectl describe pods frontend`))
)

type DescribeOptions struct {
	CmdParent string
	Selector  string
	Namespace string

	Describer  func(*meta.RESTMapping) (printers.Describer, error)
	NewBuilder func() *resource.Builder

	BuilderArgs []string

	EnforceNamespace     bool
	AllNamespaces        bool
	IncludeUninitialized bool

	DescriberSettings *printers.DescriberSettings
	FilenameOptions   *resource.FilenameOptions

	genericclioptions.IOStreams
}

func NewCmdDescribe(parent string, f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := &DescribeOptions{
		FilenameOptions: &resource.FilenameOptions{},
		DescriberSettings: &printers.DescriberSettings{
			ShowEvents: true,
		},

		CmdParent: parent,

		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use: "describe (-f FILENAME | TYPE [NAME_PREFIX | -l label] | TYPE/NAME)",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Show details of a specific resource or group of resources"),
		Long:    describeLong + "\n\n" + cmdutil.SuggestApiResources(parent),
		Example: describeExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}
	usage := "containing the resource to describe"
	cmdutil.AddFilenameOptionFlags(cmd, o.FilenameOptions, usage)
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.AllNamespaces, "all-namespaces", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&o.DescriberSettings.ShowEvents, "show-events", o.DescriberSettings.ShowEvents, "If true, display events related to the described object.")
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

func (o *DescribeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	if o.AllNamespaces {
		o.EnforceNamespace = false
	}

	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(o.FilenameOptions.Filenames) {
		return fmt.Errorf("You must specify the type of resource to describe. %s\n", cmdutil.SuggestApiResources(o.CmdParent))
	}

	o.BuilderArgs = args

	o.Describer = func(mapping *meta.RESTMapping) (printers.Describer, error) {
		return cmdutil.DescriberFn(f, mapping)
	}

	o.NewBuilder = f.NewBuilder

	// include the uninitialized objects by default
	// unless user explicitly set --include-uninitialized=false
	o.IncludeUninitialized = cmdutil.ShouldIncludeUninitialized(cmd, true)
	return nil
}

func (o *DescribeOptions) Validate(args []string) error {
	return nil
}

func (o *DescribeOptions) Run() error {
	r := o.NewBuilder().
		Unstructured().
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().AllNamespaces(o.AllNamespaces).
		FilenameParam(o.EnforceNamespace, o.FilenameOptions).
		LabelSelectorParam(o.Selector).
		IncludeUninitialized(o.IncludeUninitialized).
		ResourceTypeOrNameArgs(true, o.BuilderArgs...).
		Flatten().
		Do()
	err := r.Err()
	if err != nil {
		return err
	}

	allErrs := []error{}
	infos, err := r.Infos()
	if err != nil {
		if apierrors.IsNotFound(err) && len(o.BuilderArgs) == 2 {
			return o.DescribeMatchingResources(err, o.BuilderArgs[0], o.BuilderArgs[1])
		}
		allErrs = append(allErrs, err)
	}

	errs := sets.NewString()
	first := true
	for _, info := range infos {
		mapping := info.ResourceMapping()
		describer, err := o.Describer(mapping)
		if err != nil {
			if errs.Has(err.Error()) {
				continue
			}
			allErrs = append(allErrs, err)
			errs.Insert(err.Error())
			continue
		}
		s, err := describer.Describe(info.Namespace, info.Name, *o.DescriberSettings)
		if err != nil {
			if errs.Has(err.Error()) {
				continue
			}
			allErrs = append(allErrs, err)
			errs.Insert(err.Error())
			continue
		}
		if first {
			first = false
			fmt.Fprint(o.Out, s)
		} else {
			fmt.Fprintf(o.Out, "\n\n%s", s)
		}
	}

	return utilerrors.NewAggregate(allErrs)
}

func (o *DescribeOptions) DescribeMatchingResources(originalError error, resource, prefix string) error {
	r := o.NewBuilder().
		Unstructured().
		NamespaceParam(o.Namespace).DefaultNamespace().
		ResourceTypeOrNameArgs(true, resource).
		SingleResourceType().
		Flatten().
		Do()
	mapping, err := r.ResourceMapping()
	if err != nil {
		return err
	}
	describer, err := o.Describer(mapping)
	if err != nil {
		return err
	}
	infos, err := r.Infos()
	if err != nil {
		return err
	}
	isFound := false
	for ix := range infos {
		info := infos[ix]
		if strings.HasPrefix(info.Name, prefix) {
			isFound = true
			s, err := describer.Describe(info.Namespace, info.Name, *o.DescriberSettings)
			if err != nil {
				return err
			}
			fmt.Fprintf(o.Out, "%s\n", s)
		}
	}
	if !isFound {
		return originalError
	}
	return nil
}
