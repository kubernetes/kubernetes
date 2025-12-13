/*
Copyright 2019 The Kubernetes Authors.

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
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/kubectl/pkg/cmd/set"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// RestartOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type RestartOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	Resources []string

	Builder          func() *resource.Builder
	Restarter        polymorphichelpers.ObjectRestarterFunc
	Namespace        string
	EnforceNamespace bool
	LabelSelector    string

	resource.FilenameOptions
	genericiooptions.IOStreams

	fieldManager string

	All               bool
	AllNamespaces     bool
	ResourceTypesFlag string
}

var (
	restartLong = templates.LongDesc(i18n.T(`
		Restart a resource.

	        Resource rollout will be restarted.`))

	restartExample = templates.Examples(`
		# Restart all deployments in the test-namespace namespace
		kubectl rollout restart deployment -n test-namespace

		# Restart a deployment
		kubectl rollout restart deployment/nginx

		# Restart a daemon set
		kubectl rollout restart daemonset/abc

		# Restart all rollout-capable resources (Deployment, DaemonSet, StatefulSet) in the current namespace
		kubectl rollout restart --all
	
		# Restart all rollout-capable resources across all namespaces
		kubectl rollout restart --all --all-namespaces
	
		# Restart all deployments across all namespaces
		kubectl rollout restart deployment --all-namespaces

		# Restart deployments with the app=nginx label
		kubectl rollout restart deployment --selector=app=nginx`)
)

// NewRolloutRestartOptions returns an initialized RestartOptions instance
func NewRolloutRestartOptions(streams genericiooptions.IOStreams) *RestartOptions {
	return &RestartOptions{
		PrintFlags: genericclioptions.NewPrintFlags("restarted").WithTypeSetter(scheme.Scheme),
		IOStreams:  streams,
	}
}

// NewCmdRolloutRestart returns a Command instance for 'rollout restart' sub command
func NewCmdRolloutRestart(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	o := NewRolloutRestartOptions(streams)

	validArgs := []string{"deployment", "daemonset", "statefulset"}

	cmd := &cobra.Command{
		Use:                   "restart RESOURCE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Restart a resource"),
		Long:                  restartLong,
		Example:               restartExample,
		ValidArgsFunction:     completion.SpecifiedResourceTypeAndNameCompletionFunc(f, validArgs),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunRestart())
		},
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmdutil.AddFieldManagerFlagVar(cmd, &o.fieldManager, "kubectl-rollout")
	cmdutil.AddLabelSelectorFlagVar(cmd, &o.LabelSelector)
	cmd.Flags().BoolVar(&o.All, "all", false, "Restart all rollout-capable resources in the current namespace")
	cmd.Flags().BoolVar(&o.AllNamespaces, "all-namespaces", false, "If present, list the requested resource across all namespaces")
	o.PrintFlags.AddFlags(cmd)
	return cmd
}

// Complete completes all the required options
func (o *RestartOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Resources = args

	o.Restarter = polymorphichelpers.ObjectRestarterFn

	var err error
	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	if o.AllNamespaces {
		o.Namespace = ""
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		return o.PrintFlags.ToPrinter()
	}

	o.Builder = f.NewBuilder

	return nil
}

func (o *RestartOptions) Validate() error {
	if !o.All && len(o.Resources) == 0 && cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("required resource not specified")
	}
	if o.All && len(o.Resources) > 0 {
		return fmt.Errorf("`--all` cannot be used with resource arguments (%v)", o.Resources)
	}
	if o.All && !cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("`--all` cannot be used with `-f` or `--filename`")
	}
	if o.AllNamespaces && len(o.Namespace) != 0 {
		return fmt.Errorf("`--all-namespaces` cannot be used with `--namespace`")
	}
	return nil
}

// Supported rollout-capable resource types
var rolloutCapableResources = []string{"deployment", "daemonset", "statefulset"}

// RunRestart performs the execution of 'rollout restart' sub command
func (o RestartOptions) RunRestart() error {
	if len(o.Resources) > 0 || !cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return o.handleExplicitResources()
	}

	if o.All {
		allErrs, resourcesFound, notFoundResources := o.handleAllResources()
		o.printAllResourcesNotFoundMessage(resourcesFound, notFoundResources)

		if len(allErrs) > 0 {
			return utilerrors.NewAggregate(allErrs)
		}
		return nil
	}

	return fmt.Errorf("you must specify resources or use --all")
}

func (o RestartOptions) restartResources(r *resource.Result) error {
	allErrs := []error{}
	infos, err := r.Infos()
	if err != nil {
		allErrs = append(allErrs, err)
	}

	patches := set.CalculatePatches(infos, scheme.DefaultJSONEncoder(), set.PatchFn(o.Restarter))

	for _, patch := range patches {
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
			allErrs = append(allErrs, fmt.Errorf("failed to create patch for %v: if restart has already been triggered within the past second, please wait before attempting to trigger another", info.Name))
			continue
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).
			WithFieldManager(o.fieldManager).
			Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch, nil)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch: %v", err))
			continue
		}

		err = info.Refresh(obj, true)
		if err != nil {
			allErrs = append(allErrs, err)
			continue
		}
		printer, err := o.ToPrinter("restarted")
		if err != nil {
			allErrs = append(allErrs, err)
			continue
		}
		if err = printer.PrintObj(info.Object, o.Out); err != nil {
			allErrs = append(allErrs, err)
		}
	}

	return utilerrors.NewAggregate(allErrs)
}

func (o RestartOptions) handleExplicitResources() error {
	r := o.Builder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		LabelSelectorParam(o.LabelSelector).
		ResourceTypeOrNameArgs(true, o.Resources...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()

	if err := r.Err(); err != nil {
		return err
	}

	infos, err := r.Infos()
	if err != nil {
		return err
	}

	if len(infos) == 0 {
		o.printNoResourcesFoundMessage()
		return nil
	}

	return o.restartResources(r)
}

func (o RestartOptions) printNoResourcesFoundMessage() {
	var parts []string

	if len(o.Resources) > 0 {
		parts = append(parts, fmt.Sprintf("%v", o.Resources))
	}
	if o.LabelSelector != "" {
		parts = append(parts, fmt.Sprintf("label selector: %s", o.LabelSelector))
	}
	if !cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		parts = append(parts, "files")
	}

	var msg string
	switch len(parts) {
	case 0:
		msg = "No resources found matching the given criteria"
	default:
		msg = fmt.Sprintf("No resources found matching: %s", strings.Join(parts, ", "))
	}

	if o.Namespace != "" {
		msg += fmt.Sprintf(" in namespace %q", o.Namespace)
	} else {
		msg += " in the cluster"
	}

	_, _ = fmt.Fprintf(o.ErrOut, "%s.\n", msg)
}

func (o RestartOptions) handleAllResources() ([]error, bool, []string) {
	allErrs := []error{}
	resourcesFound := false
	notFoundResources := []string{}

	for _, rt := range rolloutCapableResources {
		r := o.Builder().
			WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
			NamespaceParam(o.Namespace).DefaultNamespace().
			ResourceTypeOrNameArgs(true, rt).
			LabelSelectorParam(o.LabelSelector).
			ContinueOnError().
			Latest().
			Flatten().
			Do()

		if err := r.Err(); err != nil {
			allErrs = append(allErrs, fmt.Errorf("error listing %s: %w", rt, err))
			continue
		}

		infos, _ := r.Infos()
		if len(infos) == 0 {
			notFoundResources = append(notFoundResources, rt)
			continue
		}

		resourcesFound = true

		if err := o.restartResources(r); err != nil {
			allErrs = append(allErrs, err)
		}
	}

	return allErrs, resourcesFound, notFoundResources
}

func (o RestartOptions) printAllResourcesNotFoundMessage(resourcesFound bool, notFoundResources []string) {
	if !resourcesFound {
		if o.Namespace != "" {
			_, _ = fmt.Fprintf(o.ErrOut, "No rollout-capable resources found in namespace %q.\n", o.Namespace)

		} else {
			_, _ = fmt.Fprintf(o.ErrOut, "No rollout-capable resources found in the cluster.\n")
		}
	}

	if len(notFoundResources) > 0 {
		msg := fmt.Sprintf("No resources found for: %s", strings.Join(notFoundResources, ", "))
		if o.LabelSelector != "" {
			msg += fmt.Sprintf(", label selector: %s", o.LabelSelector)
		}
		_, _ = fmt.Fprintf(o.ErrOut, "%s\n", msg)
	}
}
