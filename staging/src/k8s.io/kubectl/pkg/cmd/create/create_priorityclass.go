/*
Copyright 2017 The Kubernetes Authors.

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

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	schedulingv1client "k8s.io/client-go/kubernetes/typed/scheduling/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	pcLong = templates.LongDesc(i18n.T(`
		Create a priority class with the specified name, value, globalDefault and description.`))

	pcExample = templates.Examples(i18n.T(`
		# Create a priority class named high-priority
		kubectl create priorityclass high-priority --value=1000 --description="high priority"

		# Create a priority class named default-priority that is considered as the global default priority
		kubectl create priorityclass default-priority --value=1000 --global-default=true --description="default priority"

		# Create a priority class named high-priority that cannot preempt pods with lower priority
		kubectl create priorityclass high-priority --value=1000 --description="high priority" --preemption-policy="Never"`))
)

// PriorityClassOptions holds the options for 'create priorityclass' sub command
type PriorityClassOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error

	Name             string
	Value            int32
	GlobalDefault    bool
	Description      string
	PreemptionPolicy string
	FieldManager     string
	CreateAnnotation bool

	Client              *schedulingv1client.SchedulingV1Client
	DryRunStrategy      cmdutil.DryRunStrategy
	ValidationDirective string

	genericiooptions.IOStreams
}

// NewPriorityClassOptions returns an initialized PriorityClassOptions instance
func NewPriorityClassOptions(ioStreams genericiooptions.IOStreams) *PriorityClassOptions {
	return &PriorityClassOptions{
		Value:            0,
		PreemptionPolicy: "PreemptLowerPriority",
		PrintFlags:       genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:        ioStreams,
	}
}

// NewCmdCreatePriorityClass is a macro command to create a new priorityClass.
func NewCmdCreatePriorityClass(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewPriorityClassOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "priorityclass NAME --value=VALUE --global-default=BOOL [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"pc"},
		Short:                 i18n.T("Create a priority class with the specified name"),
		Long:                  pcLong,
		Example:               pcExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().Int32Var(&o.Value, "value", o.Value, i18n.T("the value of this priority class."))
	cmd.Flags().BoolVar(&o.GlobalDefault, "global-default", o.GlobalDefault, i18n.T("global-default specifies whether this PriorityClass should be considered as the default priority."))
	cmd.Flags().StringVar(&o.Description, "description", o.Description, i18n.T("description is an arbitrary string that usually provides guidelines on when this priority class should be used."))
	cmd.Flags().StringVar(&o.PreemptionPolicy, "preemption-policy", o.PreemptionPolicy, i18n.T("preemption-policy is the policy for preempting pods with lower priority."))
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	return cmd
}

// Complete completes all the required options
func (o *PriorityClassOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Name, err = NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	restConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = schedulingv1client.NewForConfig(restConfig)
	if err != nil {
		return err
	}

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
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

	o.ValidationDirective, err = cmdutil.GetValidationDirective(cmd)
	if err != nil {
		return err
	}

	return nil
}

// Run calls the CreateSubcommandOptions.Run in the PriorityClassOptions instance
func (o *PriorityClassOptions) Run() error {
	priorityClass, err := o.createPriorityClass()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, priorityClass, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		var err error
		priorityClass, err = o.Client.PriorityClasses().Create(context.TODO(), priorityClass, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create priorityclass: %v", err)
		}
	}

	return o.PrintObj(priorityClass)
}

func (o *PriorityClassOptions) createPriorityClass() (*schedulingv1.PriorityClass, error) {
	preemptionPolicy := corev1.PreemptionPolicy(o.PreemptionPolicy)
	return &schedulingv1.PriorityClass{
		// this is ok because we know exactly how we want to be serialized
		TypeMeta: metav1.TypeMeta{APIVersion: schedulingv1.SchemeGroupVersion.String(), Kind: "PriorityClass"},
		ObjectMeta: metav1.ObjectMeta{
			Name: o.Name,
		},
		Value:            o.Value,
		GlobalDefault:    o.GlobalDefault,
		Description:      o.Description,
		PreemptionPolicy: &preemptionPolicy,
	}, nil
}
