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

package describe

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/describe"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	describeLong = templates.LongDesc(i18n.T(`
		Show details of a specific resource or group of resources.

		Print a detailed description of the selected resources, including related resources such
		as events or controllers. You may select a single object by name, all objects of that
		type, provide a name prefix, or label selector. For example:

		    $ kubectl describe TYPE NAME_PREFIX

		will first check for an exact match on TYPE and NAME_PREFIX. If no such resource
		exists, it will output details for every resource that has a name prefixed with NAME_PREFIX.`))

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
		kubectl describe pods -l name=myLabel

		# Describe all pods managed by the 'frontend' replication controller
		# (rc-created pods get the name of the rc as a prefix in the pod name)
		kubectl describe pods frontend`))
)

// DescribeFlags directly reflect the information that CLI is gathering via flags. They will be converted to Options,
// which reflect the runtime requirements for the command.
type DescribeFlags struct {
	Factory           cmdutil.Factory
	Selector          string
	AllNamespaces     bool
	FilenameOptions   *resource.FilenameOptions
	DescriberSettings *describe.DescriberSettings
	genericiooptions.IOStreams
}

// NewDescribeFlags returns a default DescribeFlags
func NewDescribeFlags(f cmdutil.Factory, streams genericiooptions.IOStreams) *DescribeFlags {
	return &DescribeFlags{
		Factory:         f,
		FilenameOptions: &resource.FilenameOptions{},
		DescriberSettings: &describe.DescriberSettings{
			ShowEvents: true,
			ChunkSize:  cmdutil.DefaultChunkSize,
		},
		IOStreams: streams,
	}
}

// AddFlags registers flags for a cli
func (flags *DescribeFlags) AddFlags(cmd *cobra.Command) {
	cmdutil.AddFilenameOptionFlags(cmd, flags.FilenameOptions, "containing the resource to describe")
	cmdutil.AddLabelSelectorFlagVar(cmd, &flags.Selector)
	cmd.Flags().BoolVarP(&flags.AllNamespaces, "all-namespaces", "A", flags.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&flags.DescriberSettings.ShowEvents, "show-events", flags.DescriberSettings.ShowEvents, "If true, display events related to the described object.")
	cmdutil.AddChunkSizeFlag(cmd, &flags.DescriberSettings.ChunkSize)
}

// ToOptions converts from CLI inputs to runtime input
func (flags *DescribeFlags) ToOptions(parent string, args []string) (*DescribeOptions, error) {

	var err error
	namespace, enforceNamespace, err := flags.Factory.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return nil, err
	}

	if flags.AllNamespaces {
		enforceNamespace = false
	}

	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(flags.FilenameOptions.Filenames, flags.FilenameOptions.Kustomize) {
		return nil, fmt.Errorf("You must specify the type of resource to describe. %s\n", cmdutil.SuggestAPIResources(parent))
	}

	builderArgs := args

	describer := func(mapping *meta.RESTMapping) (describe.ResourceDescriber, error) {
		return describe.DescriberFn(flags.Factory, mapping)
	}

	o := &DescribeOptions{
		Selector:          flags.Selector,
		Namespace:         namespace,
		Describer:         describer,
		NewBuilder:        flags.Factory.NewBuilder,
		BuilderArgs:       builderArgs,
		EnforceNamespace:  enforceNamespace,
		AllNamespaces:     flags.AllNamespaces,
		FilenameOptions:   flags.FilenameOptions,
		DescriberSettings: flags.DescriberSettings,
		IOStreams:         flags.IOStreams,
	}

	return o, nil
}

func NewCmdDescribe(parent string, f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	flags := NewDescribeFlags(f, streams)

	cmd := &cobra.Command{
		Use:                   "describe (-f FILENAME | TYPE [NAME_PREFIX | -l label] | TYPE/NAME)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Show details of a specific resource or group of resources"),
		Long:                  describeLong + "\n\n" + cmdutil.SuggestAPIResources(parent),
		Example:               describeExample,
		ValidArgsFunction:     completion.ResourceTypeAndNameCompletionFunc(f),
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(parent, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	flags.AddFlags(cmd)

	return cmd
}

func (o *DescribeOptions) Validate() error {
	return nil
}

func (o *DescribeOptions) Run() error {
	r := o.NewBuilder().
		Unstructured().
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().AllNamespaces(o.AllNamespaces).
		FilenameParam(o.EnforceNamespace, o.FilenameOptions).
		LabelSelectorParam(o.Selector).
		ResourceTypeOrNameArgs(true, o.BuilderArgs...).
		RequestChunksOf(o.DescriberSettings.ChunkSize).
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

	if len(infos) == 0 && len(allErrs) == 0 {
		// if we wrote no output, and had no errors, be sure we output something.
		if o.AllNamespaces {
			fmt.Fprintln(o.ErrOut, "No resources found")
		} else {
			fmt.Fprintf(o.ErrOut, "No resources found in %s namespace.\n", o.Namespace)
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
		RequestChunksOf(o.DescriberSettings.ChunkSize).
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

type DescribeOptions struct {
	CmdParent string
	Selector  string
	Namespace string

	Describer  func(*meta.RESTMapping) (describe.ResourceDescriber, error)
	NewBuilder func() *resource.Builder

	BuilderArgs []string

	EnforceNamespace bool
	AllNamespaces    bool

	DescriberSettings *describe.DescriberSettings
	FilenameOptions   *resource.FilenameOptions

	genericiooptions.IOStreams
}
