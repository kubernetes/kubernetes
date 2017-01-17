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
	"bytes"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/golang/glog"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/intstr"
)

var (
	rollingUpdate_long = templates.LongDesc(`
		Perform a rolling update of the given ReplicationController.

		Replaces the specified replication controller with a new replication controller by updating one pod at a time to use the
		new PodTemplate. The new-controller.json must specify the same namespace as the
		existing replication controller and overwrite at least one (common) label in its replicaSelector.

		![Workflow](http://kubernetes.io/images/docs/kubectl_rollingupdate.svg)`)

	rollingUpdate_example = templates.Examples(`
		# Update pods of frontend-v1 using new replication controller data in frontend-v2.json.
		kubectl rolling-update frontend-v1 -f frontend-v2.json

		# Update pods of frontend-v1 using JSON data passed into stdin.
		cat frontend-v2.json | kubectl rolling-update frontend-v1 -f -

		# Update the pods of frontend-v1 to frontend-v2 by just changing the image, and switching the
		# name of the replication controller.
		kubectl rolling-update frontend-v1 frontend-v2 --image=image:v2

		# Update the pods of frontend by just changing the image, and keeping the old name.
		kubectl rolling-update frontend --image=image:v2

		# Abort and reverse an existing rollout in progress (from frontend-v1 to frontend-v2).
		kubectl rolling-update frontend-v1 frontend-v2 --rollback`)
)

var (
	updatePeriod, _ = time.ParseDuration("1m0s")
	timeout, _      = time.ParseDuration("5m0s")
	pollInterval, _ = time.ParseDuration("3s")
)

func NewCmdRollingUpdate(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &resource.FilenameOptions{}

	cmd := &cobra.Command{
		Use: "rolling-update OLD_CONTROLLER_NAME ([NEW_CONTROLLER_NAME] --image=NEW_CONTAINER_IMAGE | -f NEW_CONTROLLER_SPEC)",
		// rollingupdate is deprecated.
		Aliases: []string{"rollingupdate"},
		Short:   "Perform a rolling update of the given ReplicationController",
		Long:    rollingUpdate_long,
		Example: rollingUpdate_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunRollingUpdate(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().Duration("update-period", updatePeriod, `Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().Duration("poll-interval", pollInterval, `Time delay between polling for replication controller status after the update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().Duration("timeout", timeout, `Max time to wait for a replication controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	usage := "Filename or URL to file to use to create the new replication controller."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.MarkFlagRequired("filename")
	cmd.Flags().String("image", "", "Image to use for upgrading the replication controller. Must be distinct from the existing image (either new image or new image tag).  Can not be used with --filename/-f")
	cmd.MarkFlagRequired("image")
	cmd.Flags().String("deployment-label-key", "deployment", "The key to use to differentiate between two different controllers, default 'deployment'.  Only relevant when --image is specified, ignored otherwise")
	cmd.Flags().String("container", "", "Container name which will have its image upgraded. Only relevant when --image is specified, ignored otherwise. Required when using --image on a multi-container pod")
	cmd.Flags().String("image-pull-policy", "", "Explicit policy for when to pull container images. Required when --image is same as existing image, ignored otherwise.")
	cmd.Flags().Bool("rollback", false, "If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)

	return cmd
}

func validateArguments(cmd *cobra.Command, filenames, args []string) error {
	deploymentKey := cmdutil.GetFlagString(cmd, "deployment-label-key")
	image := cmdutil.GetFlagString(cmd, "image")
	rollback := cmdutil.GetFlagBool(cmd, "rollback")

	errors := []error{}
	if len(deploymentKey) == 0 {
		errors = append(errors, cmdutil.UsageError(cmd, "--deployment-label-key can not be empty"))
	}
	if len(filenames) > 1 {
		errors = append(errors, cmdutil.UsageError(cmd, "May only specify a single filename for new controller"))
	}

	if !rollback {
		if len(filenames) == 0 && len(image) == 0 {
			errors = append(errors, cmdutil.UsageError(cmd, "Must specify --filename or --image for new controller"))
		} else if len(filenames) != 0 && len(image) != 0 {
			errors = append(errors, cmdutil.UsageError(cmd, "--filename and --image can not both be specified"))
		}
	} else {
		if len(filenames) != 0 || len(image) != 0 {
			errors = append(errors, cmdutil.UsageError(cmd, "Don't specify --filename or --image on rollback"))
		}
	}

	if len(args) < 1 {
		errors = append(errors, cmdutil.UsageError(cmd, "Must specify the controller to update"))
	}

	return utilerrors.NewAggregate(errors)
}

func RunRollingUpdate(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *resource.FilenameOptions) error {
	if len(os.Args) > 1 && os.Args[1] == "rollingupdate" {
		printDeprecationWarning("rolling-update", "rollingupdate")
	}
	err := validateArguments(cmd, options.Filenames, args)
	if err != nil {
		return err
	}

	deploymentKey := cmdutil.GetFlagString(cmd, "deployment-label-key")
	filename := ""
	image := cmdutil.GetFlagString(cmd, "image")
	pullPolicy := cmdutil.GetFlagString(cmd, "image-pull-policy")
	oldName := args[0]
	rollback := cmdutil.GetFlagBool(cmd, "rollback")
	period := cmdutil.GetFlagDuration(cmd, "update-period")
	interval := cmdutil.GetFlagDuration(cmd, "poll-interval")
	timeout := cmdutil.GetFlagDuration(cmd, "timeout")
	dryrun := cmdutil.GetDryRunFlag(cmd)
	outputFormat := cmdutil.GetFlagString(cmd, "output")
	container := cmdutil.GetFlagString(cmd, "container")

	if len(options.Filenames) > 0 {
		filename = options.Filenames[0]
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}
	coreClient := clientset.Core()

	var newRc *api.ReplicationController
	// fetch rc
	oldRc, err := coreClient.ReplicationControllers(cmdNamespace).Get(oldName, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) || len(image) == 0 || len(args) > 1 {
			return err
		}
		// We're in the middle of a rename, look for an RC with a source annotation of oldName
		newRc, err := kubectl.FindSourceController(coreClient, cmdNamespace, oldName)
		if err != nil {
			return err
		}
		return kubectl.Rename(coreClient, newRc, oldName)
	}

	var keepOldName bool
	var replicasDefaulted bool

	mapper, typer := f.Object()

	if len(filename) != 0 {
		schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
		if err != nil {
			return err
		}

		request := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
			Schema(schema).
			NamespaceParam(cmdNamespace).DefaultNamespace().
			FilenameParam(enforceNamespace, &resource.FilenameOptions{Recursive: false, Filenames: []string{filename}}).
			Do()
		obj, err := request.Object()
		if err != nil {
			return err
		}
		var ok bool
		// Handle filename input from stdin. The resource builder always returns an api.List
		// when creating resource(s) from a stream.
		if list, ok := obj.(*api.List); ok {
			if len(list.Items) > 1 {
				return cmdutil.UsageError(cmd, "%s specifies multiple items", filename)
			}
			if len(list.Items) == 0 {
				return cmdutil.UsageError(cmd, "please make sure %s exists and is not empty", filename)
			}
			obj = list.Items[0]
		}
		newRc, ok = obj.(*api.ReplicationController)
		if !ok {
			if gvks, _, err := typer.ObjectKinds(obj); err == nil {
				return cmdutil.UsageError(cmd, "%s contains a %v not a ReplicationController", filename, gvks[0])
			}
			glog.V(4).Infof("Object %#v is not a ReplicationController", obj)
			return cmdutil.UsageError(cmd, "%s does not specify a valid ReplicationController", filename)
		}
		infos, err := request.Infos()
		if err != nil || len(infos) != 1 {
			glog.V(2).Infof("was not able to recover adequate information to discover if .spec.replicas was defaulted")
		} else {
			replicasDefaulted = isReplicasDefaulted(infos[0])
		}
	}
	// If the --image option is specified, we need to create a new rc with at least one different selector
	// than the old rc. This selector is the hash of the rc, with a suffix to provide uniqueness for
	// same-image updates.
	if len(image) != 0 {
		codec := api.Codecs.LegacyCodec(clientset.CoreClient.RESTClient().APIVersion())
		keepOldName = len(args) == 1
		newName := findNewName(args, oldRc)
		if newRc, err = kubectl.LoadExistingNextReplicationController(coreClient, cmdNamespace, newName); err != nil {
			return err
		}
		if newRc != nil {
			if inProgressImage := newRc.Spec.Template.Spec.Containers[0].Image; inProgressImage != image {
				return cmdutil.UsageError(cmd, "Found existing in-progress update to image (%s).\nEither continue in-progress update with --image=%s or rollback with --rollback", inProgressImage, inProgressImage)
			}
			fmt.Fprintf(out, "Found existing update in progress (%s), resuming.\n", newRc.Name)
		} else {
			config := &kubectl.NewControllerConfig{
				Namespace:     cmdNamespace,
				OldName:       oldName,
				NewName:       newName,
				Image:         image,
				Container:     container,
				DeploymentKey: deploymentKey,
			}
			if oldRc.Spec.Template.Spec.Containers[0].Image == image {
				if len(pullPolicy) == 0 {
					return cmdutil.UsageError(cmd, "--image-pull-policy (Always|Never|IfNotPresent) must be provided when --image is the same as existing container image")
				}
				config.PullPolicy = api.PullPolicy(pullPolicy)
			}
			newRc, err = kubectl.CreateNewControllerFromCurrentController(coreClient, codec, config)
			if err != nil {
				return err
			}
		}
		// Update the existing replication controller with pointers to the 'next' controller
		// and adding the <deploymentKey> label if necessary to distinguish it from the 'next' controller.
		oldHash, err := api.HashObject(oldRc, codec)
		if err != nil {
			return err
		}
		// If new image is same as old, the hash may not be distinct, so add a suffix.
		oldHash += "-orig"
		oldRc, err = kubectl.UpdateExistingReplicationController(coreClient, coreClient, oldRc, cmdNamespace, newRc.Name, deploymentKey, oldHash, out)
		if err != nil {
			return err
		}
	}

	if rollback {
		keepOldName = len(args) == 1
		newName := findNewName(args, oldRc)
		if newRc, err = kubectl.LoadExistingNextReplicationController(coreClient, cmdNamespace, newName); err != nil {
			return err
		}

		if newRc == nil {
			return cmdutil.UsageError(cmd, "Could not find %s to rollback.\n", newName)
		}
	}

	if oldName == newRc.Name {
		return cmdutil.UsageError(cmd, "%s cannot have the same name as the existing ReplicationController %s",
			filename, oldName)
	}

	updater := kubectl.NewRollingUpdater(newRc.Namespace, coreClient, coreClient)

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
	// TODO: handle scales during rolling update
	if replicasDefaulted {
		newRc.Spec.Replicas = oldRc.Spec.Replicas
	}
	if dryrun {
		oldRcData := &bytes.Buffer{}
		newRcData := &bytes.Buffer{}
		if outputFormat == "" {
			oldRcData.WriteString(oldRc.Name)
			newRcData.WriteString(newRc.Name)
		} else {
			if err := f.PrintObject(cmd, mapper, oldRc, oldRcData); err != nil {
				return err
			}
			if err := f.PrintObject(cmd, mapper, newRc, newRcData); err != nil {
				return err
			}
		}
		fmt.Fprintf(out, "Rolling from:\n%s\nTo:\n%s\n", string(oldRcData.Bytes()), string(newRcData.Bytes()))
		return nil
	}
	updateCleanupPolicy := kubectl.DeleteRollingUpdateCleanupPolicy
	if keepOldName {
		updateCleanupPolicy = kubectl.RenameRollingUpdateCleanupPolicy
	}
	config := &kubectl.RollingUpdaterConfig{
		Out:            out,
		OldRc:          oldRc,
		NewRc:          newRc,
		UpdatePeriod:   period,
		Interval:       interval,
		Timeout:        timeout,
		CleanupPolicy:  updateCleanupPolicy,
		MaxUnavailable: intstr.FromInt(0),
		MaxSurge:       intstr.FromInt(1),
	}
	if rollback {
		err = kubectl.AbortRollingUpdate(config)
		if err != nil {
			return err
		}
		coreClient.ReplicationControllers(config.NewRc.Namespace).Update(config.NewRc)
	}
	err = updater.Update(config)
	if err != nil {
		return err
	}

	message := "rolling updated"
	if keepOldName {
		newRc.Name = oldName
	} else {
		message = fmt.Sprintf("rolling updated to %q", newRc.Name)
	}
	newRc, err = coreClient.ReplicationControllers(cmdNamespace).Get(newRc.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if outputFormat != "" {
		return f.PrintObject(cmd, mapper, newRc, out)
	}
	kinds, _, err := api.Scheme.ObjectKinds(newRc)
	if err != nil {
		return err
	}
	_, res := meta.KindToResource(kinds[0])
	cmdutil.PrintSuccess(mapper, false, out, res.Resource, oldName, dryrun, message)
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

func isReplicasDefaulted(info *resource.Info) bool {
	if info == nil || info.VersionedObject == nil {
		// was unable to recover versioned info
		return false
	}
	switch t := info.VersionedObject.(type) {
	case *v1.ReplicationController:
		return t.Spec.Replicas == nil
	}
	return false
}
