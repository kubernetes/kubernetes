/*
Copyright 2014 Google Inc. All rights reserved.

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
	"crypto/md5"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
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

	// fetch rc
	oldRc, err := client.ReplicationControllers(cmdNamespace).Get(oldName)
	if err != nil {
		return err
	}

	keepOldName := false

	mapper, typer := f.Object()
	var newRc *api.ReplicationController

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
		var newName string
		var err error

		if len(args) >= 2 {
			newName = args[1]
		} else {
			newName, _ = kubectl.GetNextControllerAnnotation(oldRc)
		}

		if len(newName) > 0 {
			newRc, err = client.ReplicationControllers(cmdNamespace).Get(newName)
			if err != nil && !apierrors.IsNotFound(err) {
				return err
			}
			fmt.Fprint(out, "Found existing update in progress (%s), resuming.\n", newName)
		}
		if newRc == nil {
			// load the old RC into the "new" RC
			if newRc, err = client.ReplicationControllers(cmdNamespace).Get(oldName); err != nil {
				return err
			}

			if len(newRc.Spec.Template.Spec.Containers) > 1 {
				// TODO: support multi-container image update.
				return errors.New("Image update is not supported for multi-container pods")
			}
			if len(newRc.Spec.Template.Spec.Containers) == 0 {
				return cmdutil.UsageError(cmd, "Pod has no containers! (%v)", newRc)
			}
			newRc.Spec.Template.Spec.Containers[0].Image = image

			newHash, err := hashObject(newRc, client.Codec)
			if err != nil {
				return err
			}

			if len(newName) == 0 {
				keepOldName = true
				newName = fmt.Sprintf("%s-%s", newRc.Name, newHash)
			}
			newRc.Name = newName

			newRc.Spec.Selector[deploymentKey] = newHash
			newRc.Spec.Template.Labels[deploymentKey] = newHash
			// Clear resource version after hashing so that identical updates get different hashes.
			newRc.ResourceVersion = ""

			kubectl.SetNextControllerAnnotation(oldRc, newName)
			if _, found := oldRc.Spec.Selector[deploymentKey]; !found {
				if oldRc, err = addDeploymentKeyToReplicationController(oldRc, client, deploymentKey, cmdNamespace, out); err != nil {
					return err
				}
			}
		}
	}
	newName := newRc.Name
	if oldName == newName {
		return cmdutil.UsageError(cmd, "%s cannot have the same name as the existing ReplicationController %s",
			filename, oldName)
	}

	updater := kubectl.NewRollingUpdater(newRc.Namespace, kubectl.NewRollingUpdaterClient(client))

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
	err = updater.Update(&kubectl.RollingUpdaterConfig{
		Out:           out,
		OldRc:         oldRc,
		NewRc:         newRc,
		UpdatePeriod:  period,
		Interval:      interval,
		Timeout:       timeout,
		CleanupPolicy: updateCleanupPolicy,
	})
	if err != nil {
		return err
	}

	if keepOldName {
		fmt.Fprintf(out, "%s\n", oldName)
	} else {
		fmt.Fprintf(out, "%s\n", newName)
	}
	return nil
}

func hashObject(obj runtime.Object, codec runtime.Codec) (string, error) {
	data, err := codec.Encode(obj)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", md5.Sum(data)), nil
}

const MaxRetries = 3

func addDeploymentKeyToReplicationController(oldRc *api.ReplicationController, client *client.Client, deploymentKey, namespace string, out io.Writer) (*api.ReplicationController, error) {
	oldHash, err := hashObject(oldRc, client.Codec)
	if err != nil {
		return nil, err
	}
	// First, update the template label.  This ensures that any newly created pods will have the new label
	if oldRc.Spec.Template.Labels == nil {
		oldRc.Spec.Template.Labels = map[string]string{}
	}
	oldRc.Spec.Template.Labels[deploymentKey] = oldHash
	if oldRc, err = client.ReplicationControllers(namespace).Update(oldRc); err != nil {
		return nil, err
	}

	// Update all pods managed by the rc to have the new hash label, so they are correctly adopted
	// TODO: extract the code from the label command and re-use it here.
	podList, err := client.Pods(namespace).List(labels.SelectorFromSet(oldRc.Spec.Selector), fields.Everything())
	if err != nil {
		return nil, err
	}
	for ix := range podList.Items {
		pod := &podList.Items[ix]
		if pod.Labels == nil {
			pod.Labels = map[string]string{
				deploymentKey: oldHash,
			}
		} else {
			pod.Labels[deploymentKey] = oldHash
		}
		err = nil
		delay := 3
		for i := 0; i < MaxRetries; i++ {
			_, err = client.Pods(namespace).Update(pod)
			if err != nil {
				fmt.Fprint(out, "Error updating pod (%v), retrying after %d seconds", err, delay)
				time.Sleep(time.Second * time.Duration(delay))
				delay *= delay
			} else {
				break
			}
		}
		if err != nil {
			return nil, err
		}
	}

	if oldRc.Spec.Selector == nil {
		oldRc.Spec.Selector = map[string]string{}
	}
	// Copy the old selector, so that we can scrub out any orphaned pods
	selectorCopy := map[string]string{}
	for k, v := range oldRc.Spec.Selector {
		selectorCopy[k] = v
	}
	oldRc.Spec.Selector[deploymentKey] = oldHash

	// Update the selector of the rc so it manages all the pods we updated above
	if oldRc, err = client.ReplicationControllers(namespace).Update(oldRc); err != nil {
		return nil, err
	}

	// Clean up any orphaned pods that don't have the new label, this can happen if the rc manager
	// doesn't see the update to its pod template and creates a new pod with the old labels after
	// we've finished re-adopting existing pods to the rc.
	podList, err = client.Pods(namespace).List(labels.SelectorFromSet(selectorCopy), fields.Everything())
	for ix := range podList.Items {
		pod := &podList.Items[ix]
		if value, found := pod.Labels[deploymentKey]; !found || value != oldHash {
			if err := client.Pods(namespace).Delete(pod.Name, nil); err != nil {
				return nil, err
			}
		}
	}

	return oldRc, nil
}
