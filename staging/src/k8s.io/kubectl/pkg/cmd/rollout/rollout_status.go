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

package rollout

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	statusLong = templates.LongDesc(i18n.T(`
		Show the status of the rollout.

		By default 'rollout status' will watch the status of the latest rollout
		until it's done. If you don't want to wait for the rollout to finish then
		you can use --watch=false. Note that if a new rollout starts in-between, then
		'rollout status' will continue watching the latest revision. If you want to
		pin to a specific revision and abort if it is rolled over by another revision,
		use --revision=N where N is the revision you need to watch for.`))

	statusExample = templates.Examples(`
		# Watch the rollout status of a deployment
		kubectl rollout status deployment/nginx`)
)

// RolloutStatusOptions holds the command-line options for 'rollout status' sub command
type RolloutStatusOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	Namespace        string
	EnforceNamespace bool
	BuilderArgs      []string

	Watch    bool
	Revision int64
	Timeout  time.Duration

	StatusViewerFn func(*meta.RESTMapping) (polymorphichelpers.StatusViewer, error)
	Builder        func() *resource.Builder
	DynamicClient  dynamic.Interface

	FilenameOptions *resource.FilenameOptions
	genericclioptions.IOStreams
}

// NewRolloutStatusOptions returns an initialized RolloutStatusOptions instance
func NewRolloutStatusOptions(streams genericclioptions.IOStreams) *RolloutStatusOptions {
	return &RolloutStatusOptions{
		PrintFlags:      genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme),
		FilenameOptions: &resource.FilenameOptions{},
		IOStreams:       streams,
		Watch:           true,
		Timeout:         0,
	}
}

// NewCmdRolloutStatus returns a Command instance for the 'rollout status' sub command
func NewCmdRolloutStatus(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewRolloutStatusOptions(streams)

	validArgs := []string{"deployment", "daemonset", "statefulset"}

	cmd := &cobra.Command{
		Use:                   "status (TYPE NAME | TYPE/NAME) [flags]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Show the status of the rollout"),
		Long:                  statusLong,
		Example:               statusExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
		ValidArgs: validArgs,
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, o.FilenameOptions, usage)
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "Watch the status of the rollout until it's done.")
	cmd.Flags().Int64Var(&o.Revision, "revision", o.Revision, "Pin to a specific revision for showing its status. Defaults to 0 (last revision).")
	cmd.Flags().DurationVar(&o.Timeout, "timeout", o.Timeout, "The length of time to wait before ending watch, zero means never. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")

	return cmd
}

// Complete completes all the required options
func (o *RolloutStatusOptions) Complete(f cmdutil.Factory, args []string) error {
	o.Builder = f.NewBuilder

	var err error
	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.BuilderArgs = args
	o.StatusViewerFn = polymorphichelpers.StatusViewerFn

	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}

	o.DynamicClient, err = dynamic.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	return nil
}

// Validate makes sure all the provided values for command-line options are valid
func (o *RolloutStatusOptions) Validate() error {
	if len(o.BuilderArgs) == 0 && cmdutil.IsFilenameSliceEmpty(o.FilenameOptions.Filenames, o.FilenameOptions.Kustomize) {
		return fmt.Errorf("required resource not specified")
	}

	if o.Revision < 0 {
		return fmt.Errorf("revision must be a positive integer: %v", o.Revision)
	}

	return nil
}

// Run performs the execution of 'rollout status' sub command
func (o *RolloutStatusOptions) Run() error {
	r := o.Builder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, o.FilenameOptions).
		ResourceTypeOrNameArgs(true, o.BuilderArgs...).
		SingleResourceType().
		Latest().
		Do()
	err := r.Err()
	if err != nil {
		return err
	}

	infos, err := r.Infos()
	if err != nil {
		return err
	}
	if len(infos) != 1 {
		return fmt.Errorf("rollout status is only supported on individual resources and resource collections - %d resources were found", len(infos))
	}
	info := infos[0]
	mapping := info.ResourceMapping()

	statusViewer, err := o.StatusViewerFn(mapping)
	if err != nil {
		return err
	}

	fieldSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(context.TODO(), options)
		},
	}

	// if the rollout isn't done yet, keep watching deployment status
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), o.Timeout)
	intr := interrupt.New(nil, cancel)
	return intr.Run(func() error {
		_, err = watchtools.UntilWithSync(ctx, lw, &unstructured.Unstructured{}, nil, func(e watch.Event) (bool, error) {
			switch t := e.Type; t {
			case watch.Added, watch.Modified:
				status, done, err := statusViewer.Status(e.Object.(runtime.Unstructured), o.Revision)
				if err != nil {
					return false, err
				}
				fmt.Fprintf(o.Out, "%s", status)
				// Quit waiting if the rollout is done
				if done {
					return true, nil
				}

				shouldWatch := o.Watch
				if !shouldWatch {
					return true, nil
				}

				return false, nil

			case watch.Deleted:
				// We need to abort to avoid cases of recreation and not to silently watch the wrong (new) object
				return true, fmt.Errorf("object has been deleted")

			default:
				return true, fmt.Errorf("internal error: unexpected event %#v", e)
			}
		})
		return err
	})
}
