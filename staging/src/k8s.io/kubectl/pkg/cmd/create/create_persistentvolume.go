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
	volumeAccessModes = []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
		v1.ReadWriteOncePod,
	}

	persistentVolumeLong = templates.LongDesc(i18n.T(`
	Create a persistent volume with the specified name.`))

	persistentVolumeExample = templates.Examples(i18n.T(`
		# Create a persistent volume and set the default hostPath to the root directory.
		kubectl create persistentvolume simple --storage 1Gi

		# Create a persistent volume and define a custom hostPath location.
		kubectl create persistentvolume pvHostPath --storage 1Gi --host-path /Volumes/Data

		# Create a persistent volume and define a storageClassName.
		kubectl create persistentvolume pvHostPath --storage 1Gi --host-path /Volumes/Data --storage-class manual

		# Create a persistent volume includes ReadOnlyMany and ReadWriteMany access modes.
		kubectl create persistentvolume mypv --storage 1Gi -p /Volumes/Data --access-mode ReadOnlyMany --access-mode ReadWriteMany 

		# Create a persistent volume and persistent volume claim.
		kubectl create persistentvolume pvHostPath --storage 1Gi -p /Volumes/Data --namespace default --claim
		`))
)

// CreatePersistentVolumeOptions is returned by NewCmdCreatePersistentVolume
type CreatePersistentVolumeOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	PrintObj func(obj runtime.Object) error

	Name         string
	AccessModes  []string
	HostPath     string
	Storage      string
	StorageClass string
	Claim        bool

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

// NewCreatePersistentVolumeOptions creates the CreatePersistentVolumeOptions to be used later
func NewCreatePersistentVolumeOptions(ioStreams genericclioptions.IOStreams) *CreatePersistentVolumeOptions {
	return &CreatePersistentVolumeOptions{
		PrintFlags:  genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:   ioStreams,
		AccessModes: []string{"ReadWriteOnce"},
		HostPath:    "/",
		Claim:       false,
	}
}

// NewCmdCreatePersistentVolume is a macro command to create a new persistent volume.
// This command is better known to users as `kubectl create persistentvolume`.
func NewCmdCreatePersistentVolume(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreatePersistentVolumeOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "persistentvolume NAME --storage=VOLUME-SIZE [--dry-run=server|client|none]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"persistentvolumes", "pv"},
		Short:                 i18n.T("Create a persistent volume"),
		Long:                  persistentVolumeLong,
		Example:               persistentVolumeExample,
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
	cmd.Flags().StringArrayVarP(&o.AccessModes, "access-mode", "m", o.AccessModes, "Set permission to access the volume (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany, ReadWriteMany, or ReadWriteOncePod).")
	cmd.Flags().StringVarP(&o.HostPath, "host-path", "p", o.HostPath, "A hostPath uses a file or directory on the Node to emulate network-attached storage.")
	cmd.Flags().StringVarP(&o.Storage, "storage", "r", o.Storage, "An administrator provides the storage resource to the volume size.")
	cmd.Flags().StringVarP(&o.StorageClass, "storage-class", "c", o.StorageClass, "A StorageClass provides a way for administrators to set up dynamic provisioning.")
	cmd.Flags().BoolVar(&o.Claim, "claim", o.Claim, "If true, create a PersistentVolumeClaim associated with the PersistentVolume.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")

	cmd.MarkFlagRequired("storage")

	return cmd
}

// Validate checks to the CreatePersistentVolumeOptions to see if there is sufficient information run the command
func (o *CreatePersistentVolumeOptions) Validate() error {
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
func (o *CreatePersistentVolumeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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

// Run performs the execution of 'create persistentvolume' sub command
func (o *CreatePersistentVolumeOptions) Run() error {
	var runtimeObject runtime.Object
	var persistentvolumeClaim *v1.PersistentVolumeClaim
	persistentvolume, err := o.createPersistentVolume()
	if err != nil {
		return err
	}
	runtimeObject = persistentvolume

	if o.Claim {
		persistentvolumeClaim, err = o.createPersistentVolumeClaim()
		if err != nil {
			return err
		}
		o.PrintObj(runtimeObject) //print PersistentVolume
		runtimeObject = persistentvolumeClaim
	}

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, persistentvolume, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		createOptions.FieldValidation = o.ValidationDirective
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(persistentvolume.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		_, err = o.Client.PersistentVolumes().Create(context.TODO(), persistentvolume, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create PersistentVolumes: %v", err)
		}
		if persistentvolumeClaim != nil {
			_, err = o.Client.PersistentVolumeClaims(o.Namespace).Create(context.TODO(), persistentvolumeClaim, createOptions)
			if err != nil {
				return fmt.Errorf("failed to create PersistentVolumeClaims: %v", err)
			}
		}
	}

	return o.PrintObj(runtimeObject)
}

func (o *CreatePersistentVolumeOptions) createPersistentVolume() (*v1.PersistentVolume, error) {
	spec, err := o.buildPersistentVolumeSpec()
	if err != nil {
		return nil, err
	}

	persistentVolume := &v1.PersistentVolume{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolume",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: o.Name,
		},
		Spec: *spec,
	}

	return persistentVolume, nil
}

// buildPersistentVolumeSpec builds the .spec from the diverse arguments passed to kubectl
func (o *CreatePersistentVolumeOptions) buildPersistentVolumeSpec() (*v1.PersistentVolumeSpec, error) {
	var err error
	spec := v1.PersistentVolumeSpec{
		PersistentVolumeSource: v1.PersistentVolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: o.HostPath},
		},
	}

	resources := v1.ResourceList{}
	resources[v1.ResourceStorage] = res.MustParse(o.Storage)
	spec.Capacity = resources

	if o.StorageClass != "" {
		spec.StorageClassName = o.StorageClass
	}

	spec.AccessModes, err = o.getAccessModes()
	if err != nil {
		return nil, err
	}

	return &spec, nil
}

func (o *CreatePersistentVolumeOptions) createPersistentVolumeClaim() (*v1.PersistentVolumeClaim, error) {
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
func (o *CreatePersistentVolumeOptions) buildPersistentVolumeClaimSpec() (*v1.PersistentVolumeClaimSpec, error) {
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

func (o *CreatePersistentVolumeOptions) getAccessModes() ([]v1.PersistentVolumeAccessMode, error) {
	accessModes := []v1.PersistentVolumeAccessMode{}
out:
	for _, accessMode := range o.AccessModes {
		for _, volumeAccessMode := range volumeAccessModes {
			if strings.EqualFold(accessMode, string(volumeAccessMode)) {
				accessModes = append(accessModes, volumeAccessMode)
				continue out
			}
		}
		return nil, fmt.Errorf(`invalid access-mode value (%v)`, accessMode)
	}
	return accessModes, nil
}
