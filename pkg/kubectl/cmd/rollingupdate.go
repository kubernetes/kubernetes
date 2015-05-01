/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/spf13/cobra"
)

const (
	updatePeriod       = "1m0s"
	timeout            = "5m0s"
	pollInterval       = "3s"
	rollingUpdate_long = `Perform a rolling update of the given ReplicationController.

Replaces the specified controller with new controller, updating one pod at a time to use the
new PodTemplate. The new-controller.json must specify the same namespace as the
existing controller and overwrite at least one (common) label in its replicaSelector.`
	rollingUpdate_example = `// Update pods of frontend-v1 using new controller data in frontend-v2.json.
$ kubectl rolling-update frontend-v1 -f frontend-v2.json

// Update pods of frontend-v1 using JSON data passed into stdin.
$ cat frontend-v2.json | kubectl rolling-update frontend-v1 -f -

// Update the pods of frontend-v1 to frontend-v2 by just changing the image, and switching the
// name of the replication controller.
$ kubectl rolling-update frontend-v1 frontend-v2 --image=image:v2

// Update the pods of frontend by just changing the image, and keeping the old name
$ kubectl rolling-update frontend --image=image:v2
`
)

func NewCmdRollingUpdate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use: "rolling-update OLD_CONTROLLER_NAME ([NEW_CONTROLLER_NAME] --image=NEW_CONTAINER_IMAGE | -f NEW_CONTROLLER_SPEC)",
		// rollingupdate is deprecated.
		Aliases: []string{"rollingupdate"},
		Short:   "Perform a rolling update of the given ReplicationController.",
		Long:    rollingUpdate_long,
		Example: rollingUpdate_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunRollingUpdate(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().String("update-period", updatePeriod, `Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("poll-interval", pollInterval, `Time delay between polling controller status after update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().String("timeout", timeout, `Max time to wait for a controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to create the new controller.")
	cmd.Flags().String("image", "", "Image to upgrade the controller to.  Can not be used with --filename/-f")
	cmd.Flags().String("deployment-label-key", "deployment", "The key to use to differentiate between two different controllers, default 'deployment'.  Only relevant when --image is specified, ignored otherwise")
	cmd.Flags().Bool("dry-run", false, "If true, print out the changes that would be made, but don't actually make them.")
	cmd.Flags().Bool("rollback", false, "If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout")
	cmdutil.AddPrinterFlags(cmd)
	return cmd
}

func validateArguments(cmd *cobra.Command, args []string) (deploymentKey, filename, image, oldName string, err error) {
	deploymentKey = cmdutil.GetFlagString(cmd, "deployment-label-key")
	filename = cmdutil.GetFlagString(cmd, "filename")
	image = cmdutil.GetFlagString(cmd, "image")

	if len(deploymentKey) == 0 {
		return "", "", "", "", cmdutil.UsageError(cmd, "--deployment-label-key can not be empty")
	}
	if len(filename) == 0 && len(image) == 0 {
		return "", "", "", "", cmdutil.UsageError(cmd, "Must specify --filename or --image for new controller")
	}
	if len(filename) != 0 && len(image) != 0 {
		return "", "", "", "", cmdutil.UsageError(cmd, "--filename and --image can not both be specified")
	}
	if len(args) < 1 {
		return "", "", "", "", cmdutil.UsageError(cmd, "Must specify the controller to update")
	}

	return deploymentKey, filename, image, args[0], nil
}

func RunRollingUpdate(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	if os.Args[1] == "rollingupdate" {
		printDeprecationWarning("rolling-update", "rollingupdate")
	}
	deploymentKey, filename, image, oldName, err := validateArguments(cmd, args)
	if err != nil {
		return err
	}
	period := cmdutil.GetFlagDuration(cmd, "update-period")
	interval := cmdutil.GetFlagDuration(cmd, "poll-interval")
	timeout := cmdutil.GetFlagDuration(cmd, "timeout")
	dryrun := cmdutil.GetFlagBool(cmd, "dry-run")

	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	updaterClient := kubectl.NewRollingUpdaterClient(client)

	var newRc *api.ReplicationController
	// fetch rc
	oldRc, err := client.ReplicationControllers(cmdNamespace).Get(oldName)
	if err != nil {
		if !errors.IsNotFound(err) || len(image) == 0 || len(args) > 1 {
			return err
		}
		// We're in the middle of a rename, look for an RC with a source annotation of oldName
		newRc, err := kubectl.FindSourceController(updaterClient, cmdNamespace, oldName)
		if err != nil {
			return err
		}
		return kubectl.Rename(kubectl.NewRollingUpdaterClient(client), newRc, oldName)
	}

	var keepOldName bool

	mapper, typer := f.Object()

	if len(filename) != 0 {
		obj, err := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
			NamespaceParam(cmdNamespace).RequireNamespace().
			FilenameParam(filename).
			Do().
			Object()
		if err != nil {
			return err
		}
		var ok bool
		newRc, ok = obj.(*api.ReplicationController)
		if !ok {
			return cmdutil.UsageError(cmd, "%s does not specify a valid ReplicationController", filename)
		}
	}
	// If the --image option is specified, we need to create a new rc with at least one different selector
	// than the old rc. This selector is the hash of the rc, which will differ because the new rc has a
	// different image.
	if len(image) != 0 {
		keepOldName = len(args) == 1
		newName := findNewName(args, oldRc)
		if newRc, err = kubectl.LoadExistingNextReplicationController(client, cmdNamespace, newName); err != nil {
			return err
		}
		if newRc != nil {
			fmt.Fprintf(out, "Found existing update in progress (%s), resuming.\n", newRc.Name)
		} else {
			newRc, err = kubectl.CreateNewControllerFromCurrentController(client, cmdNamespace, oldName, newName, image, deploymentKey)
			if err != nil {
				return err
			}
		}
		// Update the existing replication controller with pointers to the 'next' controller
		// and adding the <deploymentKey> label if necessary to distinguish it from the 'next' controller.
		oldHash, err := api.HashObject(oldRc, client.Codec)
		if err != nil {
			return err
		}
		oldRc, err = kubectl.UpdateExistingReplicationController(client, oldRc, cmdNamespace, newRc.Name, deploymentKey, oldHash, out)
		if err != nil {
			return err
		}
	}
	if oldName == newRc.Name {
		return cmdutil.UsageError(cmd, "%s cannot have the same name as the existing ReplicationController %s",
			filename, oldName)
	}

	updater := kubectl.NewRollingUpdater(newRc.Namespace, updaterClient)

	// To successfully pull off a rolling update the new and old rc have to differ
	// by at least one selector. Every new pod should have the selector and every
	// old pod should not have the selector.
	var hasLabel bool
	for key, oldValue := range oldRc.Spec.Selector {
		if newValue, ok := newRc.Spec.Selector[key]; ok && newValue != oldValue {
			hasLabel = true
			break
		}
	}
	if !hasLabel {
		return cmdutil.UsageError(cmd, "%s must specify a matching key with non-equal value in Selector for %s",
			filename, oldName)
	}
	// TODO: handle resizes during rolling update
	if newRc.Spec.Replicas == 0 {
		newRc.Spec.Replicas = oldRc.Spec.Replicas
	}
	if dryrun {
		oldRcData := &bytes.Buffer{}
		if err := f.PrintObject(cmd, oldRc, oldRcData); err != nil {
			return err
		}
		newRcData := &bytes.Buffer{}
		if err := f.PrintObject(cmd, newRc, newRcData); err != nil {
			return err
		}
		fmt.Fprintf(out, "Rolling from:\n%s\nTo:\n%s\n", string(oldRcData.Bytes()), string(newRcData.Bytes()))
		return nil
	}
	updateCleanupPolicy := kubectl.DeleteRollingUpdateCleanupPolicy
	if keepOldName {
		updateCleanupPolicy = kubectl.RenameRollingUpdateCleanupPolicy
	}
	config := &kubectl.RollingUpdaterConfig{
		Out:           out,
		OldRc:         oldRc,
		NewRc:         newRc,
		UpdatePeriod:  period,
		Interval:      interval,
		Timeout:       timeout,
		CleanupPolicy: updateCleanupPolicy,
	}
	if cmdutil.GetFlagBool(cmd, "rollback") {
		kubectl.AbortRollingUpdate(config)
		client.ReplicationControllers(config.NewRc.Namespace).Update(config.NewRc)
	}
	err = updater.Update(config)
	if err != nil {
		return err
	}

	if keepOldName {
		fmt.Fprintf(out, "%s\n", oldName)
	} else {
		fmt.Fprintf(out, "%s\n", newRc.Name)
	}
	return nil
}

func findNewName(args []string, oldRc *api.ReplicationController) string {
	if len(args) >= 2 {
		return args[1]
	}
	if oldRc != nil {
		newName, _ := kubectl.GetNextControllerAnnotation(oldRc)
		return newName
	}
	return ""
}
