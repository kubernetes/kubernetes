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

package scale

import (
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes"
	batchclient "k8s.io/client-go/kubernetes/typed/batch/v1"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
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

const (
	timeout = 5 * time.Minute
)

type ScaleOptions struct {
	FilenameOptions resource.FilenameOptions
	RecordFlags     *genericclioptions.RecordFlags
	PrintFlags      *genericclioptions.PrintFlags
	PrintObj        printers.ResourcePrinterFunc

	Selector        string
	All             bool
	Replicas        int
	ResourceVersion string
	CurrentReplicas int
	Timeout         time.Duration

	Recorder                     genericclioptions.Recorder
	builder                      *resource.Builder
	namespace                    string
	enforceNamespace             bool
	args                         []string
	shortOutput                  bool
	clientSet                    kubernetes.Interface
	scaler                       kubectl.Scaler
	unstructuredClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	parent                       string

	genericclioptions.IOStreams
}

func NewScaleOptions(ioStreams genericclioptions.IOStreams) *ScaleOptions {
	return &ScaleOptions{
		PrintFlags:      genericclioptions.NewPrintFlags("scaled"),
		RecordFlags:     genericclioptions.NewRecordFlags(),
		CurrentReplicas: -1,
		Recorder:        genericclioptions.NoopRecorder{},
		IOStreams:       ioStreams,
	}
}

// NewCmdScale returns a cobra command with the appropriate configuration and flags to run scale
func NewCmdScale(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewScaleOptions(ioStreams)

	validArgs := []string{"deployment", "replicaset", "replicationcontroller", "statefulset"}

	cmd := &cobra.Command{
		Use:                   "scale [--resource-version=version] [--current-replicas=count] --replicas=COUNT (-f FILENAME | TYPE NAME)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job"),
		Long:                  scaleLong,
		Example:               scaleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate(cmd))
			cmdutil.CheckErr(o.RunScale())
		},
		ValidArgs: validArgs,
	}

	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.All, "all", o.All, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVar(&o.ResourceVersion, "resource-version", o.ResourceVersion, i18n.T("Precondition for resource version. Requires that the current resource version match this value in order to scale."))
	cmd.Flags().IntVar(&o.CurrentReplicas, "current-replicas", o.CurrentReplicas, "Precondition for current size. Requires that the current size of the resource match this value in order to scale.")
	cmd.Flags().IntVar(&o.Replicas, "replicas", o.Replicas, "The new desired number of replicas. Required.")
	cmd.MarkFlagRequired("replicas")
	cmd.Flags().DurationVar(&o.Timeout, "timeout", 0, "The length of time to wait before giving up on a scale operation, zero means don't wait. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, "identifying the resource to set a new size")
	return cmd
}

func (o *ScaleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	o.namespace, o.enforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	o.builder = f.NewBuilder()
	o.args = args
	o.shortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.clientSet, err = f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.scaler, err = scaler(f)
	if err != nil {
		return err
	}
	o.unstructuredClientForMapping = f.UnstructuredClientForMapping
	o.parent = cmd.Parent().Name()

	return nil
}

func (o *ScaleOptions) Validate(cmd *cobra.Command) error {
	if o.Replicas < 0 {
		return fmt.Errorf("The --replicas=COUNT flag is required, and COUNT must be greater than or equal to 0")
	}

	return nil
}

// RunScale executes the scaling
func (o *ScaleOptions) RunScale() error {
	r := o.builder.
		Unstructured().
		ContinueOnError().
		NamespaceParam(o.namespace).DefaultNamespace().
		FilenameParam(o.enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(o.All, o.args...).
		Flatten().
		LabelSelectorParam(o.Selector).
		Do()
	err := r.Err()
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

	precondition := &kubectl.ScalePrecondition{Size: o.CurrentReplicas, ResourceVersion: o.ResourceVersion}
	retry := kubectl.NewRetryParams(1*time.Second, 5*time.Minute)

	var waitForReplicas *kubectl.RetryParams
	if o.Timeout != 0 {
		waitForReplicas = kubectl.NewRetryParams(1*time.Second, timeout)
	}

	counter := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		if mapping.Resource.GroupResource() == (schema.GroupResource{Group: "batch", Resource: "jobs"}) {
			// go down the legacy jobs path.  This can be removed in 3.14  For now, contain it.
			fmt.Fprintf(o.ErrOut, "%s scale job is DEPRECATED and will be removed in a future version.\n", o.parent)

			if err := ScaleJob(info, o.clientSet.BatchV1(), uint(o.Replicas), precondition, retry, waitForReplicas); err != nil {
				return err
			}

		} else {
			if err := o.scaler.Scale(info.Namespace, info.Name, uint(o.Replicas), precondition, retry, waitForReplicas, mapping.Resource.GroupResource()); err != nil {
				return err
			}
		}

		// if the recorder makes a change, compute and create another patch
		if mergePatch, err := o.Recorder.MakeRecordMergePatch(info.Object); err != nil {
			klog.V(4).Infof("error recording current command: %v", err)
		} else if len(mergePatch) > 0 {
			client, err := o.unstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)
			if _, err := helper.Patch(info.Namespace, info.Name, types.MergePatchType, mergePatch, nil); err != nil {
				klog.V(4).Infof("error recording reason: %v", err)
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
	scaler := JobPsuedoScaler{
		JobsClient: jobsClient,
	}
	var jobPreconditions *ScalePrecondition
	if preconditions != nil {
		jobPreconditions = &ScalePrecondition{Size: preconditions.Size, ResourceVersion: preconditions.ResourceVersion}
	}
	var jobRetry *RetryParams
	if retry != nil {
		jobRetry = &RetryParams{Interval: retry.Interval, Timeout: retry.Timeout}
	}
	var jobWaitForReplicas *RetryParams
	if waitForReplicas != nil {
		jobWaitForReplicas = &RetryParams{Interval: waitForReplicas.Interval, Timeout: waitForReplicas.Timeout}
	}

	return scaler.Scale(info.Namespace, info.Name, count, jobPreconditions, jobRetry, jobWaitForReplicas)
}

func scaler(f cmdutil.Factory) (kubectl.Scaler, error) {
	scalesGetter, err := cmdutil.ScaleClientFn(f)
	if err != nil {
		return nil, err
	}

	return kubectl.NewScaler(scalesGetter), nil
}
