/*
Copyright 2020 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	res "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	volumeAccessClaimModes = []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
		v1.ReadWriteOncePod,
	}

	persistentVolumeClaimLong = templates.LongDesc(i18n.T(`
	Create a persistent volume claim with the specified name.`))

	persistentVolumeClaimExample = templates.Examples(i18n.T(`
		# Create a persistent volume claim.
		kubectl create persistentvolumeclaim simple --storage 1Gi --namespace default 

		# Create a persistent volume claim and define a storageClassName.
		kubectl create persistentvolumeclaim pvHostPath --storage 1Gi --storage-class manual

		# Create a persistent volume claim includes ReadOnlyMany and ReadWriteMany access modes.
		kubectl create persistentvolumeclaim mypv --storage 1Gi --access-mode ReadOnlyMany --access-mode ReadWriteMany 
	`))
)

// CreatePersistentVolumeClaimOptions is returned by NewCmdCreatePersistentVolumeClaim
type CreatePersistentVolumeClaimOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	PrintObj func(obj runtime.Object) error

	Name         string
	AccessModes  []string
	Storage      string
	StorageClass string

	Namespace        string
	EnforceNamespace bool

	Client              *coreclient.CoreV1Client
	DryRunStrategy      cmdutil.DryRunStrategy
	DryRunVerifier      *resource.QueryParamVerifier
	ValidationDirective string

	FieldManager     string
	CreateAnnotation bool

	genericclioptions.IOStreams
}

// NewCreatePersistentVolumeClaimOptions creates the CreatePersistentVolumeClaimOptions to be used later
func NewCreatePersistentVolumeClaimOptions(ioStreams genericclioptions.IOStreams) *CreatePersistentVolumeClaimOptions {
	return &CreatePersistentVolumeClaimOptions{
		PrintFlags:  genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:   ioStreams,
		AccessModes: []string{"ReadWriteOnce"},
	}
}

// NewCmdCreatePersistentVolumeClaim is a macro command to create a new persistent volume.
// This command is better known to users as `kubectl create persistentvolumeclaim`.
func NewCmdCreatePersistentVolumeClaim(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreatePersistentVolumeClaimOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "persistentvolumeclaim NAME --storage=VOLUME-SIZE [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"pvc"},
		Short:                 i18n.T("Create a persistent volume"),
		Long:                  persistentVolumeClaimLong,
		Example:               persistentVolumeClaimExample,
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
	cmd.Flags().StringVarP(&o.Storage, "storage", "r", o.Storage, "Request specific sizes for storage by the user.")
	cmd.Flags().StringVarP(&o.StorageClass, "storage-class", "c", o.StorageClass, "A claim can request a particular class by specifying the name of a StorageClass.")
	cmd.Flags().StringArrayVarP(&o.AccessModes, "access-mode", "m", o.AccessModes, "Claims can request specific access modes (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany, ReadWriteMany, or ReadWriteOncePod).")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")

	cmd.MarkFlagRequired("storage")

	return cmd
}

// Validate checks to the CreatePersistentVolumeClaimOptions to see if there is sufficient information run the command
func (o *CreatePersistentVolumeClaimOptions) Validate() error {
	_, err := res.ParseQuantity(o.Storage)
	if err != nil {
		return fmt.Errorf("%s is not a valid storage resource value", o.Storage)
	}

	_, err = o.getAccessModes()
	if err != nil {
		return err
	}

	return nil
}

// Complete completes all the options
func (o *CreatePersistentVolumeClaimOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewQueryParamVerifier(dynamicClient, f.OpenAPIGetter(), resource.QueryParamDryRun)

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

// Run performs the execution of 'create persistentvolumeclaim' sub command
func (o *CreatePersistentVolumeClaimOptions) Run() error {
	persistentvolumeclaim, err := o.createPersistentVolumeClaim()
	if err != nil {
		return err
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, persistentvolumeclaim, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(persistentvolumeclaim.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		_, err = o.Client.PersistentVolumeClaims(o.Namespace).Create(context.TODO(), persistentvolumeclaim, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create PersistentVolumeClaims: %v", err)
		}
	}

	return o.PrintObj(persistentvolumeclaim)
}

func (o *CreatePersistentVolumeClaimOptions) createPersistentVolumeClaim() (*v1.PersistentVolumeClaim, error) {
	spec, err := o.buildPersistentVolumeClaimSpec()
	if err != nil {
		return nil, err
	}

	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}

	persistentVolumeClaim := &v1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Namespace: namespace,
		},
		Spec: *spec,
	}

	return persistentVolumeClaim, nil
}

// buildPersistentVolumeClaimSpec builds the .spec from the diverse arguments passed to kubectl
func (o *CreatePersistentVolumeClaimOptions) buildPersistentVolumeClaimSpec() (*v1.PersistentVolumeClaimSpec, error) {
	var err error
	spec := v1.PersistentVolumeClaimSpec{}

	resources := v1.ResourceList{}
	resources[v1.ResourceStorage] = res.MustParse(o.Storage)
	spec.Resources.Requests = resources

	if o.StorageClass != "" {
		spec.StorageClassName = &o.StorageClass
	}

	spec.AccessModes, err = o.getAccessModes()
	if err != nil {
		return nil, err
	}

	return &spec, nil
}

func (o *CreatePersistentVolumeClaimOptions) getAccessModes() ([]v1.PersistentVolumeAccessMode, error) {
	accessModes := []v1.PersistentVolumeAccessMode{}
out:
	for _, accessMode := range o.AccessModes {
		for _, volumeAccessMode := range volumeAccessClaimModes {
			if strings.EqualFold(accessMode, string(volumeAccessMode)) {
				accessModes = append(accessModes, volumeAccessMode)
				continue out
			}
		}
		return nil, fmt.Errorf(`invalid access-mode value (%v)`, accessMode)
	}
	return accessModes, nil
}
