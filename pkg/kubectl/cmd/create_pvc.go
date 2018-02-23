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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
	internalversionstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	pvcLong = templates.LongDesc(i18n.T(`
		Create a persistent volume claim using specified subcommand`))

	pvcExample = templates.Examples(i18n.T(`
	    # Create a persistent volume claim named "myclaim" with "ReadWriteOnce" access mode and "1Gi" capacity.
		kubectl create persistentvolumeclaim myclaim --access-mode=ReadWriteOnce --capacity=1Gi

		# Create a persistent volume claim named "myclaim" with "ReadWriteOnce" access mode, "1Gi" capacity and a storage class named "highquality".
		kubectl create persistentvolumeclaim myclaim --access-mode=ReadWriteOnce --capacity=1Gi --storage-class=highquality

		# Create a persistent volume claim named "myclaim" with multiple access modes and "1Gi" capacity.
		kubectl create persistentvolumeclaim myclaim --access-mode=RWO,RWX --capacity=1Gi`))
)

type CreatePVCOptions struct {
	Name         string
	AccessModes  []api.PersistentVolumeAccessMode
	Capacity     *resource.Quantity
	StorageClass *string

	DryRun       bool
	OutputFormat string
	Namespace    string
	Client       internalversionstorage.PersistentVolumeClaimInterface
	Mapper       meta.RESTMapper
	Out          io.Writer
	PrintObject  func(obj runtime.Object) error
	PrintSuccess func(mapper meta.RESTMapper, shortOutput bool, out io.Writer, resource, name string, dryRun bool, operation string)
}

// NewCmdCreatePersistentVolumeClaim is a command to ease creating PersistentVolumeClaims.
func NewCmdCreatePersistentVolumeClaim(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreatePVCOptions{
		Out: cmdOut,
	}
	cmd := &cobra.Command{
		Use:     "persistentvolumeclaim NAME --access-mode=accessmode --capacity=capacity [--storage-class=storageclass] [--dry-run]",
		Aliases: []string{"pvc"},
		Short:   pvcLong,
		Long:    pvcLong,
		Example: pvcExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(f, cmd, args))
			cmdutil.CheckErr(c.Validate())
			cmdutil.CheckErr(c.RunCreatePVC())
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)

	cmd.Flags().StringSlice("access-mode", []string{}, "AccessMode contains a comma-separated list of required access modes that the desired persistent volume must have.")
	cmd.Flags().String("capacity", "", "Capacity represents the minimum storage capacity required.")
	cmd.Flags().String("storage-class", "", "Name of the StorageClass required by the claim.")

	return cmd
}

func (c *CreatePVCOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	c.Name = name

	// Transform access mode abbreviation to complete access mode.
	accessModes := []api.PersistentVolumeAccessMode{}
	modes := cmdutil.GetFlagStringSlice(cmd, "access-mode")
	for _, m := range modes {
		var accessMode api.PersistentVolumeAccessMode
		switch m {
		case "RWX":
			accessMode = api.ReadWriteMany
		case "RWO":
			accessMode = api.ReadWriteOnce
		case "ROX":
			accessMode = api.ReadOnlyMany
		default:
			accessMode = api.PersistentVolumeAccessMode(m)
		}
		accessModes = append(accessModes, accessMode)
	}
	c.AccessModes = accessModes

	// If user does not provide storage-class option, then StorageClass will be set as nil.
	// Otherwise, StorageClass will be the value set by user, including empty string.
	if cmd.Flags().Changed("storage-class") {
		storageClass := cmdutil.GetFlagString(cmd, "storage-class")
		c.StorageClass = &storageClass
	}

	capacityString := cmdutil.GetFlagString(cmd, "capacity")
	if len(capacityString) != 0 {
		quantity, err := resource.ParseQuantity(capacityString)
		if err != nil {
			return err
		}

		c.Capacity = &quantity
	}

	// Complete other options for Run.
	c.Mapper, _ = f.Object()

	c.DryRun = cmdutil.GetDryRunFlag(cmd)
	c.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	c.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	c.PrintObject = func(obj runtime.Object) error {
		return f.PrintObject(cmd, false, c.Mapper, obj, c.Out)
	}
	c.PrintSuccess = f.PrintSuccess

	clientSet, err := f.ClientSet()
	if err != nil {
		return err
	}
	c.Client = clientSet.Core().PersistentVolumeClaims(c.Namespace)

	return nil
}

func (c *CreatePVCOptions) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("name must be specified")
	}

	if len(c.AccessModes) == 0 {
		return fmt.Errorf("at least one access mode must be specified")
	}

	if c.Capacity == nil {
		return fmt.Errorf("capacity of the storage must be specified")
	}

	return nil
}

func (c *CreatePVCOptions) RunCreatePVC() error {
	pvc := &api.PersistentVolumeClaim{}
	pvc.Name = c.Name
	pvc.Spec = api.PersistentVolumeClaimSpec{}
	pvc.Spec.AccessModes = c.AccessModes
	pvc.Spec.StorageClassName = c.StorageClass
	pvc.Spec.Resources = api.ResourceRequirements{
		Requests: api.ResourceList{
			"storage": *c.Capacity,
		},
	}

	// Create pvc.
	if !c.DryRun {
		_, err := c.Client.Create(pvc)
		if err != nil {
			return err
		}
	}

	if useShortOutput := c.OutputFormat == "name"; useShortOutput || len(c.OutputFormat) == 0 {
		c.PrintSuccess(c.Mapper, useShortOutput, c.Out, "persistentvolumeclaims", c.Name, c.DryRun, "created")
		return nil
	}

	return c.PrintObject(pvc)
}
