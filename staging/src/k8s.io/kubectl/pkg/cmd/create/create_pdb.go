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

package create

import (
	"context"
	"fmt"
	"regexp"

	"github.com/spf13/cobra"

	policyv1 "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	resourcecli "k8s.io/cli-runtime/pkg/resource"
	policyv1client "k8s.io/client-go/kubernetes/typed/policy/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	pdbLong = templates.LongDesc(i18n.T(`
		Create a pod disruption budget with the specified name, selector, and desired minimum available pods.`))

	pdbExample = templates.Examples(i18n.T(`
		# Create a pod disruption budget named my-pdb that will select all pods with the app=rails label
		# and require at least one of them being available at any point in time
		kubectl create poddisruptionbudget my-pdb --selector=app=rails --min-available=1

		# Create a pod disruption budget named my-pdb that will select all pods with the app=nginx label
		# and require at least half of the pods selected to be available at any point in time
		kubectl create pdb my-pdb --selector=app=nginx --min-available=50%`))
)

// PodDisruptionBudgetOpts holds the command-line options for poddisruptionbudget sub command
type PodDisruptionBudgetOpts struct {
	// PrintFlags holds options necessary for obtaining a printer
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error
	// Name of resource being created
	Name string

	MinAvailable   string
	MaxUnavailable string

	// A label selector to use for this budget
	Selector         string
	CreateAnnotation bool
	FieldManager     string
	Namespace        string
	EnforceNamespace bool

	Client         *policyv1client.PolicyV1Client
	DryRunStrategy cmdutil.DryRunStrategy
	DryRunVerifier *resourcecli.DryRunVerifier

	genericclioptions.IOStreams
}

// NewPodDisruptionBudgetOpts creates a new *PodDisruptionBudgetOpts with sane defaults
func NewPodDisruptionBudgetOpts(ioStreams genericclioptions.IOStreams) *PodDisruptionBudgetOpts {
	return &PodDisruptionBudgetOpts{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreatePodDisruptionBudget is a macro command to create a new pod disruption budget.
func NewCmdCreatePodDisruptionBudget(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewPodDisruptionBudgetOpts(ioStreams)

	cmd := &cobra.Command{
		Use:                   "poddisruptionbudget NAME --selector=SELECTOR --min-available=N [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"pdb"},
		Short:                 i18n.T("Create a pod disruption budget with the specified name"),
		Long:                  pdbLong,
		Example:               pdbExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)

	cmd.Flags().StringVar(&o.MinAvailable, "min-available", o.MinAvailable, i18n.T("The minimum number or percentage of available pods this budget requires."))
	cmd.Flags().StringVar(&o.MaxUnavailable, "max-unavailable", o.MaxUnavailable, i18n.T("The maximum number or percentage of unavailable pods this budget requires."))
	cmd.Flags().StringVar(&o.Selector, "selector", o.Selector, i18n.T("A label selector to use for this budget. Only equality-based selector requirements are supported."))
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	return cmd
}

// Complete completes all the required options
func (o *PodDisruptionBudgetOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Name, err = NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	restConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = policyv1client.NewForConfig(restConfig)
	if err != nil {
		return err
	}

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	discoveryClient, err := f.ToDiscoveryClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resourcecli.NewDryRunVerifier(dynamicClient, discoveryClient)

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

// Validate checks to the PodDisruptionBudgetOpts to see if there is sufficient information run the command
func (o *PodDisruptionBudgetOpts) Validate() error {
	if len(o.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}

	if len(o.Selector) == 0 {
		return fmt.Errorf("a selector must be specified")
	}

	if len(o.MaxUnavailable) == 0 && len(o.MinAvailable) == 0 {
		return fmt.Errorf("one of min-available or max-unavailable must be specified")
	}

	if len(o.MaxUnavailable) > 0 && len(o.MinAvailable) > 0 {
		return fmt.Errorf("min-available and max-unavailable cannot be both specified")
	}

	// The following regex matches the following values:
	// 10, 20, 30%, 50% (number and percentage)
	// but not 10Gb, 20Mb
	re := regexp.MustCompile(`^[0-9]+%?$`)

	switch {
	case len(o.MinAvailable) > 0 && !re.MatchString(o.MinAvailable):
		return fmt.Errorf("invalid format specified for min-available")
	case len(o.MaxUnavailable) > 0 && !re.MatchString(o.MaxUnavailable):
		return fmt.Errorf("invalid format specified for max-unavailable")
	}

	return nil
}

// Run calls the CreateSubcommandOptions.Run in PodDisruptionBudgetOpts instance
func (o *PodDisruptionBudgetOpts) Run() error {
	podDisruptionBudget, err := o.createPodDisruptionBudgets()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, podDisruptionBudget, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(podDisruptionBudget.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		podDisruptionBudget, err = o.Client.PodDisruptionBudgets(o.Namespace).Create(context.TODO(), podDisruptionBudget, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create poddisruptionbudgets: %v", err)
		}
	}
	return o.PrintObj(podDisruptionBudget)
}

func (o *PodDisruptionBudgetOpts) createPodDisruptionBudgets() (*policyv1.PodDisruptionBudget, error) {
	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}

	podDisruptionBudget := &policyv1.PodDisruptionBudget{
		TypeMeta: metav1.TypeMeta{
			APIVersion: policyv1.SchemeGroupVersion.String(),
			Kind:       "PodDisruptionBudget",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Namespace: namespace,
		},
	}

	selector, err := metav1.ParseToLabelSelector(o.Selector)
	if err != nil {
		return nil, err
	}

	podDisruptionBudget.Spec.Selector = selector

	switch {
	case len(o.MinAvailable) > 0:
		minAvailable := intstr.Parse(o.MinAvailable)
		podDisruptionBudget.Spec.MinAvailable = &minAvailable
	case len(o.MaxUnavailable) > 0:
		maxUnavailable := intstr.Parse(o.MaxUnavailable)
		podDisruptionBudget.Spec.MaxUnavailable = &maxUnavailable
	}

	return podDisruptionBudget, nil
}
