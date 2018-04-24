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
	"time"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/scalejob"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
)

var (
	scaleLong = templates.LongDesc(i18n.T(`
		Set a new size for a Deployment, ReplicaSet, Replication Controller, or StatefulSet.

		Scale also allows users to specify one or more preconditions for the scale action.

		If --current-replicas or --resource-version is specified, it is validated before the
		scale is attempted, and it is guaranteed that the precondition holds true when the
		scale is sent to the server.`))

	scaleExample = templates.Examples(i18n.T(`
		# Scale a replicaset named 'foo' to 3.
		kubectl scale --replicas=3 rs/foo

		# Scale a resource identified by type and name specified in "foo.yaml" to 3.
		kubectl scale --replicas=3 -f foo.yaml

		# If the deployment named mysql's current size is 2, scale mysql to 3.
		kubectl scale --current-replicas=2 --replicas=3 deployment/mysql

		# Scale multiple replication controllers.
		kubectl scale --replicas=5 rc/foo rc/bar rc/baz

		# Scale statefulset named 'web' to 3.
		kubectl scale --replicas=3 statefulset/web`))
)

type ScaleOptions struct {
	FilenameOptions resource.FilenameOptions
	RecordFlags     *genericclioptions.RecordFlags
	PrintFlags      *printers.PrintFlags
	PrintObj        printers.ResourcePrinterFunc

	BuilderArgs      []string
	Namespace        string
	EnforceNamespace bool

	Builder   *resource.Builder
	ClientSet internalclientset.Interface
	Scaler    kubectl.Scaler

	All      bool
	Selector string

	CmdParent string

	ResourceVersion string
	CurrentReplicas int
	Replicas        int
	Duration        time.Duration

	ClientForMapping func(*meta.RESTMapping) (resource.RESTClient, error)

	Recorder genericclioptions.Recorder

	genericclioptions.IOStreams
}

func NewScaleOptions(ioStreams genericclioptions.IOStreams) *ScaleOptions {
	return &ScaleOptions{
		RecordFlags: genericclioptions.NewRecordFlags(),
		PrintFlags:  printers.NewPrintFlags("scaled"),

		CurrentReplicas: -1,

		Recorder:  genericclioptions.NoopRecorder{},
		IOStreams: ioStreams,
	}
}

// NewCmdScale returns a cobra command with the appropriate configuration and flags to run scale
func NewCmdScale(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewScaleOptions(streams)

	validArgs := []string{"deployment", "replicaset", "replicationcontroller", "statefulset"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use: "scale [--resource-version=version] [--current-replicas=count] --replicas=COUNT (-f FILENAME | TYPE NAME)",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job"),
		Long:    scaleLong,
		Example: scaleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			cmdutil.CheckErr(o.RunScale())
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.All, "all", o.All, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVar(&o.ResourceVersion, "resource-version", o.ResourceVersion, i18n.T("Precondition for resource version. Requires that the current resource version match this value in order to scale."))
	cmd.Flags().IntVar(&o.CurrentReplicas, "current-replicas", o.CurrentReplicas, "Precondition for current size. Requires that the current size of the resource match this value in order to scale.")
	cmd.Flags().IntVar(&o.Replicas, "replicas", o.Replicas, "The new desired number of replicas. Required.")
	cmd.MarkFlagRequired("replicas")
	cmd.Flags().DurationVar(&o.Duration, "timeout", o.Duration, "The length of time to wait before giving up on a scale operation, zero means don't wait. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")

	usage := "identifying the resource to set a new size"
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	return cmd
}

func (o *ScaleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Namespace, o.EnforceNamespace, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.CmdParent = cmd.Parent().Name()
	o.Builder = f.NewBuilder()

	o.ClientSet, err = f.ClientSet()
	if err != nil {
		return err
	}

	o.Scaler, err = f.Scaler()
	if err != nil {
		return err
	}

	o.BuilderArgs = args
	o.ClientForMapping = f.UnstructuredClientForMapping

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	o.RecordFlags.Complete(f.Command(cmd, false))
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	return nil
}

// RunScale executes the scaling
func (o *ScaleOptions) RunScale() error {

	if o.Replicas < 0 {
		return fmt.Errorf("The --replicas=COUNT flag is required, and COUNT must be greater than or equal to 0")
	}

	r := o.Builder.
		Unstructured().
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(o.All, o.BuilderArgs...).
		Flatten().
		LabelSelectorParam(o.Selector).
		Do()
	err := r.Err()
	if resource.IsUsageError(err) {
		return fmt.Errorf("%v", err)
	}
	if err != nil {
		return err
	}

	infos := []*resource.Info{}
	err = r.Visit(func(info *resource.Info, err error) error {
		if err == nil {
			infos = append(infos, info)
		}
		return nil
	})

	if len(o.ResourceVersion) != 0 && len(infos) > 1 {
		return fmt.Errorf("cannot use --resource-version with multiple resources")
	}

	currentSize := o.CurrentReplicas
	precondition := &kubectl.ScalePrecondition{Size: currentSize, ResourceVersion: o.ResourceVersion}
	retry := kubectl.NewRetryParams(kubectl.Interval, kubectl.Timeout)

	var waitForReplicas *kubectl.RetryParams
	if timeout := o.Duration; timeout != 0 {
		waitForReplicas = kubectl.NewRetryParams(kubectl.Interval, timeout)
	}

	counter := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		if mapping.Resource == "jobs" {
			// go down the legacy jobs path.  This can be removed in 3.14  For now, contain it.
			fmt.Fprintf(o.ErrOut, "%s scale job is DEPRECATED and will be removed in a future version.\n", o.CmdParent)

			if err := ScaleJob(info, o.ClientSet.Batch(), uint(o.Replicas), precondition, retry, waitForReplicas); err != nil {
				return err
			}

		} else {
			gvk := mapping.GroupVersionKind.GroupVersion().WithResource(mapping.Resource)
			if err := o.Scaler.Scale(info.Namespace, info.Name, uint(o.Replicas), precondition, retry, waitForReplicas, gvk.GroupResource()); err != nil {
				return err
			}
		}

		// if the recorder makes a change, compute and create another patch
		if mergePatch, err := o.Recorder.MakeRecordMergePatch(info.Object); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		} else if len(mergePatch) > 0 {
			client, err := o.ClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)
			if _, err := helper.Patch(info.Namespace, info.Name, types.MergePatchType, mergePatch); err != nil {
				glog.V(4).Infof("error recording reason: %v", err)
			}
		}

		counter++
		return o.PrintObj(info.Object, o.Out)
	})
	if err != nil {
		return err
	}
	if counter == 0 {
		return fmt.Errorf("no objects passed to scale")
	}
	return nil
}

func ScaleJob(info *resource.Info, jobsClient batchclient.JobsGetter, count uint, preconditions *kubectl.ScalePrecondition, retry, waitForReplicas *kubectl.RetryParams) error {
	scaler := scalejob.JobPsuedoScaler{
		JobsClient: jobsClient,
	}
	var jobPreconditions *scalejob.ScalePrecondition
	if preconditions != nil {
		jobPreconditions = &scalejob.ScalePrecondition{Size: preconditions.Size, ResourceVersion: preconditions.ResourceVersion}
	}
	var jobRetry *scalejob.RetryParams
	if retry != nil {
		jobRetry = &scalejob.RetryParams{Interval: retry.Interval, Timeout: retry.Timeout}
	}
	var jobWaitForReplicas *scalejob.RetryParams
	if waitForReplicas != nil {
		jobWaitForReplicas = &scalejob.RetryParams{Interval: waitForReplicas.Interval, Timeout: waitForReplicas.Timeout}
	}

	return scaler.Scale(info.Namespace, info.Name, count, jobPreconditions, jobRetry, jobWaitForReplicas)
}
