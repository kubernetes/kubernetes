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

package rollingupdate

import (
	"bytes"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
	"k8s.io/kubernetes/pkg/kubectl/validation"
)

var (
	rollingUpdateLong = templates.LongDesc(i18n.T(`
		Perform a rolling update of the given ReplicationController.

		Replaces the specified replication controller with a new replication controller by updating one pod at a time to use the
		new PodTemplate. The new-controller.json must specify the same namespace as the
		existing replication controller and overwrite at least one (common) label in its replicaSelector.

		![Workflow](http://kubernetes.io/images/docs/kubectl_rollingupdate.svg)`))

	rollingUpdateExample = templates.Examples(i18n.T(`
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
		kubectl rolling-update frontend-v1 frontend-v2 --rollback`))
)

const (
	updatePeriod = 1 * time.Minute
	timeout      = 5 * time.Minute
	pollInterval = 3 * time.Second
)

type RollingUpdateOptions struct {
	FilenameOptions *resource.FilenameOptions

	OldName     string
	KeepOldName bool

	DeploymentKey    string
	Image            string
	Container        string
	PullPolicy       string
	Rollback         bool
	Period           time.Duration
	Timeout          time.Duration
	Interval         time.Duration
	DryRun           bool
	OutputFormat     string
	Namespace        string
	EnforceNamespace bool

	ScaleClient scaleclient.ScalesGetter
	ClientSet   kubernetes.Interface
	Builder     *resource.Builder

	ShouldValidate bool
	Validator      func(bool) (validation.Schema, error)

	FindNewName func(*corev1.ReplicationController) string

	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	genericclioptions.IOStreams
}

func NewRollingUpdateOptions(streams genericclioptions.IOStreams) *RollingUpdateOptions {
	return &RollingUpdateOptions{
		PrintFlags:      genericclioptions.NewPrintFlags("rolling updated").WithTypeSetter(scheme.Scheme),
		FilenameOptions: &resource.FilenameOptions{},
		DeploymentKey:   "deployment",
		Timeout:         timeout,
		Interval:        pollInterval,
		Period:          updatePeriod,

		IOStreams: streams,
	}
}

func NewCmdRollingUpdate(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewRollingUpdateOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "rolling-update OLD_CONTROLLER_NAME ([NEW_CONTROLLER_NAME] --image=NEW_CONTAINER_IMAGE | -f NEW_CONTROLLER_SPEC)",
		DisableFlagsInUseLine: true,
		Short:                 "Perform a rolling update. This command is deprecated, use rollout instead.",
		Long:                  rollingUpdateLong,
		Example:               rollingUpdateExample,
		Deprecated:            `use "rollout" instead`,
		Hidden:                true,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate(cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().DurationVar(&o.Period, "update-period", o.Period, `Time to wait between updating pods. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().DurationVar(&o.Interval, "poll-interval", o.Interval, `Time delay between polling for replication controller status after the update. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	cmd.Flags().DurationVar(&o.Timeout, "timeout", o.Timeout, `Max time to wait for a replication controller to update before giving up. Valid time units are "ns", "us" (or "µs"), "ms", "s", "m", "h".`)
	usage := "Filename or URL to file to use to create the new replication controller."
	cmdutil.AddJsonFilenameFlag(cmd.Flags(), &o.FilenameOptions.Filenames, usage)
	cmd.Flags().StringVar(&o.Image, "image", o.Image, i18n.T("Image to use for upgrading the replication controller. Must be distinct from the existing image (either new image or new image tag).  Can not be used with --filename/-f"))
	cmd.Flags().StringVar(&o.DeploymentKey, "deployment-label-key", o.DeploymentKey, i18n.T("The key to use to differentiate between two different controllers, default 'deployment'.  Only relevant when --image is specified, ignored otherwise"))
	cmd.Flags().StringVar(&o.Container, "container", o.Container, i18n.T("Container name which will have its image upgraded. Only relevant when --image is specified, ignored otherwise. Required when using --image on a multi-container pod"))
	cmd.Flags().StringVar(&o.PullPolicy, "image-pull-policy", o.PullPolicy, i18n.T("Explicit policy for when to pull container images. Required when --image is same as existing image, ignored otherwise."))
	cmd.Flags().BoolVar(&o.Rollback, "rollback", o.Rollback, "If true, this is a request to abort an existing rollout that is partially rolled out. It effectively reverses current and next and runs a rollout")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddValidateFlags(cmd)

	return cmd
}

func validateArguments(cmd *cobra.Command, filenames, args []string) error {
	deploymentKey := cmdutil.GetFlagString(cmd, "deployment-label-key")
	image := cmdutil.GetFlagString(cmd, "image")
	rollback := cmdutil.GetFlagBool(cmd, "rollback")

	errors := []error{}
	if len(deploymentKey) == 0 {
		errors = append(errors, cmdutil.UsageErrorf(cmd, "--deployment-label-key can not be empty"))
	}
	if len(filenames) > 1 {
		errors = append(errors, cmdutil.UsageErrorf(cmd, "May only specify a single filename for new controller"))
	}

	if !rollback {
		if len(filenames) == 0 && len(image) == 0 {
			errors = append(errors, cmdutil.UsageErrorf(cmd, "Must specify --filename or --image for new controller"))
		} else if len(filenames) != 0 && len(image) != 0 {
			errors = append(errors, cmdutil.UsageErrorf(cmd, "--filename and --image can not both be specified"))
		}
	} else {
		if len(filenames) != 0 || len(image) != 0 {
			errors = append(errors, cmdutil.UsageErrorf(cmd, "Don't specify --filename or --image on rollback"))
		}
	}

	if len(args) < 1 {
		errors = append(errors, cmdutil.UsageErrorf(cmd, "Must specify the controller to update"))
	}

	return utilerrors.NewAggregate(errors)
}

func (o *RollingUpdateOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) > 0 {
		o.OldName = args[0]
	}
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.OutputFormat = cmdutil.GetFlagString(cmd, "output")
	o.KeepOldName = len(args) == 1
	o.ShouldValidate = cmdutil.GetFlagBool(cmd, "validate")

	o.Validator = f.Validator
	o.FindNewName = func(obj *corev1.ReplicationController) string {
		return findNewName(args, obj)
	}

	var err error
	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.ScaleClient, err = cmdutil.ScaleClientFn(f)
	if err != nil {
		return err
	}

	o.ClientSet, err = f.KubernetesClientSet()
	if err != nil {
		return err
	}

	o.Builder = f.NewBuilder()

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		if o.DryRun {
			o.PrintFlags.Complete("%s (dry run)")
		}

		return o.PrintFlags.ToPrinter()
	}
	return nil
}

func (o *RollingUpdateOptions) Validate(cmd *cobra.Command, args []string) error {
	return validateArguments(cmd, o.FilenameOptions.Filenames, args)
}

func (o *RollingUpdateOptions) Run() error {
	filename := ""
	if len(o.FilenameOptions.Filenames) > 0 {
		filename = o.FilenameOptions.Filenames[0]
	}

	coreClient := o.ClientSet.CoreV1()

	var newRc *corev1.ReplicationController
	// fetch rc
	oldRc, err := coreClient.ReplicationControllers(o.Namespace).Get(o.OldName, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) || len(o.Image) == 0 || !o.KeepOldName {
			return err
		}
		// We're in the middle of a rename, look for an RC with a source annotation of oldName
		newRc, err := kubectl.FindSourceController(coreClient, o.Namespace, o.OldName)
		if err != nil {
			return err
		}
		return kubectl.Rename(coreClient, newRc, o.OldName)
	}

	var replicasDefaulted bool

	if len(filename) != 0 {
		schema, err := o.Validator(o.ShouldValidate)
		if err != nil {
			return err
		}

		request := o.Builder.
			Unstructured().
			Schema(schema).
			NamespaceParam(o.Namespace).DefaultNamespace().
			FilenameParam(o.EnforceNamespace, &resource.FilenameOptions{Recursive: false, Filenames: []string{filename}}).
			Flatten().
			Do()
		infos, err := request.Infos()
		if err != nil {
			return err
		}
		// Handle filename input from stdin.
		if len(infos) > 1 {
			return fmt.Errorf("%s specifies multiple items", filename)
		}
		if len(infos) == 0 {
			return fmt.Errorf("please make sure %s exists and is not empty", filename)
		}

		uncastVersionedObj, err := scheme.Scheme.ConvertToVersion(infos[0].Object, corev1.SchemeGroupVersion)
		if err != nil {
			klog.V(4).Infof("Object %T is not a ReplicationController", infos[0].Object)
			return fmt.Errorf("%s contains a %v not a ReplicationController", filename, infos[0].Object.GetObjectKind().GroupVersionKind())
		}
		switch t := uncastVersionedObj.(type) {
		case *v1.ReplicationController:
			replicasDefaulted = t.Spec.Replicas == nil
			newRc = t
		}
		if newRc == nil {
			klog.V(4).Infof("Object %T is not a ReplicationController", infos[0].Object)
			return fmt.Errorf("%s contains a %v not a ReplicationController", filename, infos[0].Object.GetObjectKind().GroupVersionKind())
		}
	}

	// If the --image option is specified, we need to create a new rc with at least one different selector
	// than the old rc. This selector is the hash of the rc, with a suffix to provide uniqueness for
	// same-image updates.
	if len(o.Image) != 0 {
		codec := scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
		newName := o.FindNewName(oldRc)
		if newRc, err = kubectl.LoadExistingNextReplicationController(coreClient, o.Namespace, newName); err != nil {
			return err
		}
		if newRc != nil {
			if inProgressImage := newRc.Spec.Template.Spec.Containers[0].Image; inProgressImage != o.Image {
				return fmt.Errorf("Found existing in-progress update to image (%s).\nEither continue in-progress update with --image=%s or rollback with --rollback", inProgressImage, inProgressImage)
			}
			fmt.Fprintf(o.Out, "Found existing update in progress (%s), resuming.\n", newRc.Name)
		} else {
			config := &kubectl.NewControllerConfig{
				Namespace:     o.Namespace,
				OldName:       o.OldName,
				NewName:       newName,
				Image:         o.Image,
				Container:     o.Container,
				DeploymentKey: o.DeploymentKey,
			}
			if oldRc.Spec.Template.Spec.Containers[0].Image == o.Image {
				if len(o.PullPolicy) == 0 {
					return fmt.Errorf("--image-pull-policy (Always|Never|IfNotPresent) must be provided when --image is the same as existing container image")
				}
				config.PullPolicy = corev1.PullPolicy(o.PullPolicy)
			}
			newRc, err = kubectl.CreateNewControllerFromCurrentController(coreClient, codec, config)
			if err != nil {
				return err
			}
		}
		// Update the existing replication controller with pointers to the 'next' controller
		// and adding the <deploymentKey> label if necessary to distinguish it from the 'next' controller.
		oldHash, err := util.HashObject(oldRc, codec)
		if err != nil {
			return err
		}
		// If new image is same as old, the hash may not be distinct, so add a suffix.
		oldHash += "-orig"
		oldRc, err = kubectl.UpdateExistingReplicationController(coreClient, coreClient, oldRc, o.Namespace, newRc.Name, o.DeploymentKey, oldHash, o.Out)
		if err != nil {
			return err
		}
	}

	if o.Rollback {
		newName := o.FindNewName(oldRc)
		if newRc, err = kubectl.LoadExistingNextReplicationController(coreClient, o.Namespace, newName); err != nil {
			return err
		}

		if newRc == nil {
			return fmt.Errorf("Could not find %s to rollback.\n", newName)
		}
	}

	if o.OldName == newRc.Name {
		return fmt.Errorf("%s cannot have the same name as the existing ReplicationController %s",
			filename, o.OldName)
	}

	updater := kubectl.NewRollingUpdater(newRc.Namespace, coreClient, coreClient, o.ScaleClient)

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
		return fmt.Errorf("%s must specify a matching key with non-equal value in Selector for %s",
			filename, o.OldName)
	}
	// TODO: handle scales during rolling update
	if replicasDefaulted {
		t := *oldRc.Spec.Replicas
		newRc.Spec.Replicas = &t
	}

	if o.DryRun {
		oldRcData := &bytes.Buffer{}
		newRcData := &bytes.Buffer{}
		if o.OutputFormat == "" {
			oldRcData.WriteString(oldRc.Name)
			newRcData.WriteString(newRc.Name)
		} else {
			printer, err := o.ToPrinter("rolling updated")
			if err != nil {
				return err
			}
			if err := printer.PrintObj(oldRc, oldRcData); err != nil {
				return err
			}
			if err := printer.PrintObj(newRc, newRcData); err != nil {
				return err
			}
		}
		fmt.Fprintf(o.Out, "Rolling from:\n%s\nTo:\n%s\n", string(oldRcData.Bytes()), string(newRcData.Bytes()))
		return nil
	}
	updateCleanupPolicy := kubectl.DeleteRollingUpdateCleanupPolicy
	if o.KeepOldName {
		updateCleanupPolicy = kubectl.RenameRollingUpdateCleanupPolicy
	}
	config := &kubectl.RollingUpdaterConfig{
		Out:            o.Out,
		OldRc:          oldRc,
		NewRc:          newRc,
		UpdatePeriod:   o.Period,
		Interval:       o.Interval,
		Timeout:        timeout,
		CleanupPolicy:  updateCleanupPolicy,
		MaxUnavailable: intstr.FromInt(0),
		MaxSurge:       intstr.FromInt(1),
	}
	if o.Rollback {
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
	if o.KeepOldName {
		newRc.Name = o.OldName
	} else {
		message = fmt.Sprintf("rolling updated to %q", newRc.Name)
	}
	newRc, err = coreClient.ReplicationControllers(o.Namespace).Get(newRc.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	printer, err := o.ToPrinter(message)
	if err != nil {
		return err
	}
	return printer.PrintObj(newRc, o.Out)
}

func findNewName(args []string, oldRc *corev1.ReplicationController) string {
	if len(args) >= 2 {
		return args[1]
	}
	if oldRc != nil {
		newName, _ := kubectl.GetNextControllerAnnotation(oldRc)
		return newName
	}
	return ""
}
