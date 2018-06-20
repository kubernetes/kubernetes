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

	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl/cmd/set"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/printers"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// PauseConfig is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type PauseConfig struct {
	resource.FilenameOptions
	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	Pauser polymorphichelpers.ObjectPauserFunc
	Infos  []*resource.Info

	genericclioptions.IOStreams
}

var (
	pause_long = templates.LongDesc(`
		Mark the provided resource as paused

		Paused resources will not be reconciled by a controller.
		Use "kubectl rollout resume" to resume a paused resource.
		Currently only deployments support being paused.`)

	pause_example = templates.Examples(`
		# Mark the nginx deployment as paused. Any current state of
		# the deployment will continue its function, new updates to the deployment will not
		# have an effect as long as the deployment is paused.
		kubectl rollout pause deployment/nginx`)
)

func NewCmdRolloutPause(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := &PauseConfig{
		PrintFlags: genericclioptions.NewPrintFlags("paused").WithTypeSetter(scheme.Scheme),
		IOStreams:  streams,
	}

	validArgs := []string{"deployment"}

	cmd := &cobra.Command{
		Use: "pause RESOURCE",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Mark the provided resource as paused"),
		Long:    pause_long,
		Example: pause_example,
		Run: func(cmd *cobra.Command, args []string) {
			allErrs := []error{}
			err := o.CompletePause(f, cmd, args)
			if err != nil {
				allErrs = append(allErrs, err)
			}
			err = o.RunPause()
			if err != nil {
				allErrs = append(allErrs, err)
			}
			cmdutil.CheckErr(utilerrors.Flatten(utilerrors.NewAggregate(allErrs)))
		},
		ValidArgs: validArgs,
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	return cmd
}

func (o *PauseConfig) CompletePause(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		return cmdutil.UsageErrorf(cmd, "%s", cmd.Use)
	}

	o.Pauser = polymorphichelpers.ObjectPauserFn

	cmdNamespace, enforceNamespace, err := f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	r := f.NewBuilder().
		WithScheme(legacyscheme.Scheme).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		return o.PrintFlags.ToPrinter()
	}

	o.Infos, err = r.Infos()
	if err != nil {
		return err
	}
	return nil
}

func (o PauseConfig) RunPause() error {
	allErrs := []error{}
	for _, patch := range set.CalculatePatches(o.Infos, cmdutil.InternalVersionJSONEncoder(), set.PatchFn(o.Pauser)) {
		info := patch.Info
		if patch.Err != nil {
			resourceString := info.Mapping.Resource.Resource
			if len(info.Mapping.Resource.Group) > 0 {
				resourceString = resourceString + "." + info.Mapping.Resource.Group
			}
			allErrs = append(allErrs, fmt.Errorf("error: %s %q %v", resourceString, info.Name, patch.Err))
			continue
		}

		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			printer, err := o.ToPrinter("already paused")
			if err != nil {
				allErrs = append(allErrs, err)
				continue
			}
			if err = printer.PrintObj(cmdutil.AsDefaultVersionedOrOriginal(info.Object, info.Mapping), o.Out); err != nil {
				allErrs = append(allErrs, err)
			}
			continue
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch: %v", err))
			continue
		}

		info.Refresh(obj, true)
		printer, err := o.ToPrinter("paused")
		if err != nil {
			allErrs = append(allErrs, err)
			continue
		}
		if err = printer.PrintObj(cmdutil.AsDefaultVersionedOrOriginal(info.Object, info.Mapping), o.Out); err != nil {
			allErrs = append(allErrs, err)
		}
	}

	return utilerrors.NewAggregate(allErrs)
}
