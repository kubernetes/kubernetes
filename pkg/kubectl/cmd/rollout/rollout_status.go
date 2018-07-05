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
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

var (
	status_long = templates.LongDesc(`
		Show the status of the rollout.

		By default 'rollout status' will watch the status of the latest rollout
		until it's done. If you don't want to wait for the rollout to finish then
		you can use --watch=false. Note that if a new rollout starts in-between, then
		'rollout status' will continue watching the latest revision. If you want to
		pin to a specific revision and abort if it is rolled over by another revision,
		use --revision=N where N is the revision you need to watch for.`)

	status_example = templates.Examples(`
		# Watch the rollout status of a deployment
		kubectl rollout status deployment/nginx`)
)

type RolloutStatusOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	Namespace        string
	EnforceNamespace bool
	BuilderArgs      []string

	Watch    bool
	Revision int64

	StatusViewer func(*meta.RESTMapping) (kubectl.StatusViewer, error)
	Builder      func() *resource.Builder

	FilenameOptions *resource.FilenameOptions
	genericclioptions.IOStreams
}

func NewRolloutStatusOptions(streams genericclioptions.IOStreams) *RolloutStatusOptions {
	return &RolloutStatusOptions{
		PrintFlags:      genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme),
		FilenameOptions: &resource.FilenameOptions{},
		IOStreams:       streams,
		Watch:           true,
	}
}

func NewCmdRolloutStatus(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewRolloutStatusOptions(streams)

	validArgs := []string{"deployment", "daemonset", "statefulset"}

	cmd := &cobra.Command{
		Use: "status (TYPE NAME | TYPE/NAME) [flags]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Show the status of the rollout"),
		Long:    status_long,
		Example: status_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, args))
			cmdutil.CheckErr(o.Validate(cmd, args))
			cmdutil.CheckErr(o.Run())
		},
		ValidArgs: validArgs,
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, o.FilenameOptions, usage)
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "Watch the status of the rollout until it's done.")
	cmd.Flags().Int64Var(&o.Revision, "revision", o.Revision, "Pin to a specific revision for showing its status. Defaults to 0 (last revision).")

	return cmd
}

func (o *RolloutStatusOptions) Complete(f cmdutil.Factory, args []string) error {
	o.Builder = f.NewBuilder

	var err error
	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.BuilderArgs = args
	o.StatusViewer = func(mapping *meta.RESTMapping) (kubectl.StatusViewer, error) {
		return polymorphichelpers.StatusViewerFn(f, mapping)
	}
	return nil
}

func (o *RolloutStatusOptions) Validate(cmd *cobra.Command, args []string) error {
	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(o.FilenameOptions.Filenames) {
		return cmdutil.UsageErrorf(cmd, "Required resource not specified.")
	}

	if o.Revision < 0 {
		return fmt.Errorf("revision must be a positive integer: %v", o.Revision)
	}

	return nil
}

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

	obj, err := r.Object()
	if err != nil {
		return err
	}
	rv, err := meta.NewAccessor().ResourceVersion(obj)
	if err != nil {
		return err
	}

	statusViewer, err := o.StatusViewer(mapping)
	if err != nil {
		return err
	}

	// check if deployment's has finished the rollout
	status, done, err := statusViewer.Status(info.Namespace, info.Name, o.Revision)
	if err != nil {
		return err
	}
	fmt.Fprintf(o.Out, "%s", status)
	if done {
		return nil
	}

	shouldWatch := o.Watch
	if !shouldWatch {
		return nil
	}

	// watch for changes to the deployment
	w, err := r.Watch(rv)
	if err != nil {
		return err
	}

	// if the rollout isn't done yet, keep watching deployment status
	intr := interrupt.New(nil, w.Stop)
	return intr.Run(func() error {
		_, err := watch.Until(0, w, func(e watch.Event) (bool, error) {
			// print deployment's status
			status, done, err := statusViewer.Status(info.Namespace, info.Name, o.Revision)
			if err != nil {
				return false, err
			}
			fmt.Fprintf(o.Out, "%s", status)
			// Quit waiting if the rollout is done
			if done {
				return true, nil
			}
			return false, nil
		})
		return err
	})
}
