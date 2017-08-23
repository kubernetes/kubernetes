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

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	scaleLong = templates.LongDesc(i18n.T(`
		Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job.

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

		# Scale job named 'cron' to 3.
		kubectl scale --replicas=3 job/cron`))
)

// NewCmdScale returns a cobra command with the appropriate configuration and flags to run scale
func NewCmdScale(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	validArgs := []string{"deployment", "replicaset", "replicationcontroller", "job", "statefulset"}
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "scale [--resource-version=version] [--current-replicas=count] --replicas=COUNT (-f FILENAME | TYPE NAME)",
		Short:   i18n.T("Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job"),
		Long:    scaleLong,
		Example: scaleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(cmdutil.ValidateOutputArgs(cmd))
			shortOutput := cmdutil.GetFlagString(cmd, "output") == "name"
			err := RunScale(f, out, cmd, args, shortOutput, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmd.Flags().String("resource-version", "", i18n.T("Precondition for resource version. Requires that the current resource version match this value in order to scale."))
	cmd.Flags().Int("current-replicas", -1, "Precondition for current size. Requires that the current size of the resource match this value in order to scale.")
	cmd.Flags().Int("replicas", -1, "The new desired number of replicas. Required.")
	cmd.MarkFlagRequired("replicas")
	cmd.Flags().Duration("timeout", 0, "The length of time to wait before giving up on a scale operation, zero means don't wait. Any other values should contain a corresponding time unit (e.g. 1s, 2m, 3h).")
	cmdutil.AddOutputFlagsForMutation(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	usage := "identifying the resource to set a new size"
	cmdutil.AddFilenameOptionFlags(cmd, options, usage)
	return cmd
}

// RunScale executes the scaling
func RunScale(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, shortOutput bool, options *resource.FilenameOptions) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, _ := f.Object()
	r := f.NewBuilder(true).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options).
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if resource.IsUsageError(err) {
		return cmdutil.UsageErrorf(cmd, "%v", err)
	}
	if err != nil {
		return err
	}

	count := cmdutil.GetFlagInt(cmd, "replicas")
	if count < 0 {
		return cmdutil.UsageErrorf(cmd, "The --replicas=COUNT flag is required, and COUNT must be greater than or equal to 0")
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

	counter := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		scaler, err := f.Scaler(mapping)
		if err != nil {
			return err
		}

		currentSize := cmdutil.GetFlagInt(cmd, "current-replicas")
		precondition := &kubectl.ScalePrecondition{Size: currentSize, ResourceVersion: resourceVersion}
		retry := kubectl.NewRetryParams(kubectl.Interval, kubectl.Timeout)

		var waitForReplicas *kubectl.RetryParams
		if timeout := cmdutil.GetFlagDuration(cmd, "timeout"); timeout != 0 {
			waitForReplicas = kubectl.NewRetryParams(kubectl.Interval, timeout)
		}

		if err := scaler.Scale(info.Namespace, info.Name, uint(count), precondition, retry, waitForReplicas); err != nil {
			return err
		}
		if cmdutil.ShouldRecord(cmd, info) {
			patchBytes, patchType, err := cmdutil.ChangeResourcePatch(info, f.Command(cmd, true))
			if err != nil {
				return err
			}
			mapping := info.ResourceMapping()
			client, err := f.ClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)
			_, err = helper.Patch(info.Namespace, info.Name, patchType, patchBytes)
			if err != nil {
				return err
			}
		}
		counter++
		cmdutil.PrintSuccess(mapper, shortOutput, out, info.Mapping.Resource, info.Name, false, "scaled")
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
