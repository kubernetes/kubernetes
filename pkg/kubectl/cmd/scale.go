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
	"io"

	"github.com/spf13/cobra"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	batchclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/scalejob"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
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

	Recorder genericclioptions.Recorder
}

func NewScaleOptions() *ScaleOptions {
	return &ScaleOptions{
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},
	}
}

// NewCmdScale returns a cobra command with the appropriate configuration and flags to run scale
func NewCmdScale(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	o := NewScaleOptions()

	validArgs := []string{"deployment", "replicaset", "replicationcontroller", "statefulset"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use: "scale [--resource-version=version] [--current-replicas=count] --replicas=COUNT (-f FILENAME | TYPE NAME)",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job"),
		Long:    scaleLong,
		Example: scaleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
			cmdutil.CheckErr(o.RunScale(f, out, errOut, cmd, args, shortOutput))
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	o.RecordFlags.AddFlags(cmd)

	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().Bool("all", false, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().String("resource-version", "", i18n.T("Precondition for resource version. Requires that the current resource version match this value in order to scale."))
	cmd.Flags().Int("current-replicas", -1, "Precondition for current size. Requires that the current size of the resource match this value in order to scale.")
	cmd.Flags().Int("replicas", -1, "The new desired number of replicas. Required.")
	cmd.MarkFlagRequired("replicas")
	cmd.Flags().Duration("timeout", 0, "The length of time to wait before giving up on a scale operation, zero means don't wait. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")
	cmdutil.AddOutputFlagsForMutation(cmd)

	usage := "identifying the resource to set a new size"
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	return cmd
}

func (o *ScaleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error

	o.RecordFlags.Complete(f.Command(cmd, false))
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	return err
}

// RunScale executes the scaling
func (o *ScaleOptions) RunScale(f cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, args []string, shortOutput bool) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	count := cmdutil.GetFlagInt(cmd, "replicas")
	if count < 0 {
		return cmdutil.UsageErrorf(cmd, "The --replicas=COUNT flag is required, and COUNT must be greater than or equal to 0")
	}

	selector := cmdutil.GetFlagString(cmd, "selector")
	all := cmdutil.GetFlagBool(cmd, "all")

	r := f.NewBuilder().
		Unstructured().
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(all, args...).
		Flatten().
		LabelSelectorParam(selector).
		Do()
	err = r.Err()
	if resource.IsUsageError(err) {
		return cmdutil.UsageErrorf(cmd, "%v", err)
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

	resourceVersion := cmdutil.GetFlagString(cmd, "resource-version")
	if len(resourceVersion) != 0 && len(infos) > 1 {
		return fmt.Errorf("cannot use --resource-version with multiple resources")
	}

	currentSize := cmdutil.GetFlagInt(cmd, "current-replicas")
	precondition := &kubectl.ScalePrecondition{Size: currentSize, ResourceVersion: resourceVersion}
	retry := kubectl.NewRetryParams(kubectl.Interval, kubectl.Timeout)

	var waitForReplicas *kubectl.RetryParams
	if timeout := cmdutil.GetFlagDuration(cmd, "timeout"); timeout != 0 {
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
			fmt.Fprintf(errOut, "%s scale job is DEPRECATED and will be removed in a future version.\n", cmd.Parent().Name())

			clientset, err := f.ClientSet()
			if err != nil {
				return err
			}
			if err := ScaleJob(info, clientset.Batch(), uint(count), precondition, retry, waitForReplicas); err != nil {
				return err
			}

		} else {
			scaler, err := f.Scaler()
			if err != nil {
				return err
			}

			gvk := mapping.GroupVersionKind.GroupVersion().WithResource(mapping.Resource)
			if err := scaler.Scale(info.Namespace, info.Name, uint(count), precondition, retry, waitForReplicas, gvk.GroupResource()); err != nil {
				return err
			}
		}

		// if the recorder makes a change, compute and create another patch
		if mergePatch, err := o.Recorder.MakeRecordMergePatch(info.Object); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		} else if len(mergePatch) > 0 {
			client, err := f.UnstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)
			if _, err := helper.Patch(info.Namespace, info.Name, types.MergePatchType, mergePatch); err != nil {
				glog.V(4).Infof("error recording reason: %v", err)
			}
		}

		counter++
		cmdutil.PrintSuccess(shortOutput, out, info.Object, false, "scaled")
		return nil
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
