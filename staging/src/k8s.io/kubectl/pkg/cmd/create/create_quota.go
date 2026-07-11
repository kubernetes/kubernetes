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
	"strings"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	quotaLong = templates.LongDesc(i18n.T(`
		Create a resource quota with the specified name, hard limits, and optional scopes.`))

	quotaExample = templates.Examples(i18n.T(`
		# Create a new resource quota named my-quota
		kubectl create quota my-quota --hard=cpu=1,memory=1G,pods=2,services=3,replicationcontrollers=2,resourcequotas=1,secrets=5,persistentvolumeclaims=10

		# Create a new resource quota named best-effort
		kubectl create quota best-effort --hard=pods=100 --scopes=BestEffort`))
)

// QuotaOpts holds the command-line options for 'create quota' sub command
type QuotaOpts struct {
	// PrintFlags holds options necessary for obtaining a printer
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error
	// The name of a quota object.
	Name string
	// The hard resource limit string before parsing.
	Hard string
	// The scopes of a quota object before parsing.
	Scopes           string
	CreateAnnotation bool
	FieldManager     string
	Namespace        string
	EnforceNamespace bool

	Client              *coreclient.CoreV1Client
	DryRunStrategy      cmdutil.DryRunStrategy
	ValidationDirective string

	genericiooptions.IOStreams
}

// NewQuotaOpts creates a new *QuotaOpts with sane defaults
func NewQuotaOpts(ioStreams genericiooptions.IOStreams) *QuotaOpts {
	return &QuotaOpts{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreateQuota is a macro command to create a new quota
func NewCmdCreateQuota(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewQuotaOpts(ioStreams)

	cmd := &cobra.Command{
		Use:                   "quota NAME [--hard=key1=value1,key2=value2] [--scopes=Scope1,Scope2] [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"resourcequota"},
		Short:                 i18n.T("Create a quota with the specified name"),
		Long:                  quotaLong,
		Example:               quotaExample,
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
	cmd.Flags().StringVar(&o.Hard, "hard", o.Hard, i18n.T("A comma-delimited set of resource=quantity pairs that define a hard limit."))
	cmd.Flags().StringVar(&o.Scopes, "scopes", o.Scopes, i18n.T("A comma-delimited set of quota scopes that must all match each object tracked by the quota."))
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")
	return cmd
}

// Complete completes all the required options
func (o *QuotaOpts) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Name, err = NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	restConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = coreclient.NewForConfig(restConfig)
	if err != nil {
		return err
	}

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}

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

	o.ValidationDirective, err = cmdutil.GetValidationDirective(cmd)
	if err != nil {
		return err
	}

	return nil
}

// Validate checks to the QuotaOpts to see if there is sufficient information run the command.
func (o *QuotaOpts) Validate() error {
	if len(o.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	return nil
}

// Run does the work
func (o *QuotaOpts) Run() error {
	resourceQuota, err := o.createQuota()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, resourceQuota, scheme.DefaultJSONEncoder()); err != nil {
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
		resourceQuota, err = o.Client.ResourceQuotas(o.Namespace).Create(context.TODO(), resourceQuota, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create quota: %v", err)
		}
	}
	return o.PrintObj(resourceQuota)
}

func (o *QuotaOpts) createQuota() (*corev1.ResourceQuota, error) {
	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}
	resourceQuota := &corev1.ResourceQuota{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "ResourceQuota"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Namespace: namespace,
		},
	}

	resourceList, err := populateResourceListV1(o.Hard)
	if err != nil {
		return nil, err
	}

	scopes, err := parseScopes(o.Scopes)
	if err != nil {
		return nil, err
	}

	resourceQuota.Spec.Hard = resourceList
	resourceQuota.Spec.Scopes = scopes

	return resourceQuota, nil
}

// populateResourceListV1 takes strings of form <resourceName1>=<value1>,<resourceName1>=<value2>
// and returns ResourceList.
func populateResourceListV1(spec string) (corev1.ResourceList, error) {
	// empty input gets a nil response to preserve generator test expected behaviors
	if spec == "" {
		return nil, nil
	}

	result := corev1.ResourceList{}
	resourceStatements := strings.Split(spec, ",")
	for _, resourceStatement := range resourceStatements {
		parts := strings.Split(resourceStatement, "=")
		if len(parts) != 2 {
			return nil, fmt.Errorf("Invalid argument syntax %v, expected <resource>=<value>", resourceStatement)
		}
		resourceName := corev1.ResourceName(parts[0])
		resourceQuantity, err := resourceapi.ParseQuantity(parts[1])
		if err != nil {
			return nil, err
		}
		result[resourceName] = resourceQuantity
	}
	return result, nil
}

func parseScopes(spec string) ([]corev1.ResourceQuotaScope, error) {
	// empty input gets a nil response to preserve test expected behaviors
	if spec == "" {
		return nil, nil
	}

	scopes := strings.Split(spec, ",")
	result := make([]corev1.ResourceQuotaScope, 0, len(scopes))
	for _, scope := range scopes {
		// intentionally do not verify the scope against the valid scope list. This is done by the apiserver anyway.

		if scope == "" {
			return nil, fmt.Errorf("invalid resource quota scope \"\"")
		}

		result = append(result, corev1.ResourceQuotaScope(scope))
	}
	return result, nil
}
