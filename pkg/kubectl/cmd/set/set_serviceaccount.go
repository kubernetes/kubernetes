/*
Copyright YEAR The Kubernetes Authors.

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

package set

import (
	"io"

	"fmt"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	serviceaccount_resources = `
	replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs)`

	serviceaccount_long = templates.LongDesc(`
	Update ServiceAccount of resources.

	Possible resources (case insensitive) can be : 
	` + serviceaccount_resources)

	serviceaccount_example = templates.Examples(`
	#Set ReplicationController nginx-controller's ServiceAccount to serviceaccount1
	kubectl set serviceaccount replicationcontroller nginx-controller serviceaccount1

	#Short form of the above command
	kubectl set sa rc nginx-controller serviceaccount1
	`)
)

type ServiceAccountConfig struct {
	fileNameOptions resource.FilenameOptions

	Mapper             meta.RESTMapper
	Typer              runtime.ObjectTyper
	Infos              []*resource.Info
	Encoder            runtime.Encoder
	Out                io.Writer
	Err                io.Writer
	DryRun             bool
	ShortOutput        bool
	All                bool
	Record             bool
	Output             string
	ChangeCause        string
	Local              bool
	Cmd                *cobra.Command
	PatchErrors        []error
	Resources          []string
	ServiceAccountName string

	PrintObject func(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	PatchFunc   func(info *resource.Info) ([]byte, error)
}

func NewCmdServiceAccount(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	saConfig := &ServiceAccountConfig{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "(serviceaccount | sa) (-f FILENAME | TYPE NAME) SERVICE_ACCOUNT",
		Aliases: []string{"sa"},
		Short:   i18n.T("Update ServiceAccount of a resource"),
		Long:    serviceaccount_long,
		Example: serviceaccount_example,
		PreRun: func(cmd *cobra.Command, args []string) {
			if len(args) == 0 {
				cmd.Help()
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(saConfig.Complete(f, cmd, args))
			cmdutil.CheckErr(saConfig.Validate())
			cmdutil.CheckErr(saConfig.Run())
		},
	}
	cmdutil.AddPrinterFlags(cmd)

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &saConfig.fileNameOptions, usage)
	cmd.Flags().BoolVar(&saConfig.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&saConfig.Local, "local", false, "If true, set image will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

func (saConfig *ServiceAccountConfig) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	saConfig.Mapper, saConfig.Typer = f.Object()
	saConfig.Encoder = f.JSONEncoder()
	saConfig.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	saConfig.Record = cmdutil.GetRecordFlag(cmd)
	saConfig.ChangeCause = f.Command(cmd, false)
	saConfig.PrintObject = f.PrintObject
	saConfig.Local = cmdutil.GetFlagBool(cmd, "local")
	saConfig.DryRun = cmdutil.GetDryRunFlag(cmd)
	saConfig.Output = cmdutil.GetFlagString(cmd, "output")
	saConfig.Cmd = cmd
	saConfig.ServiceAccountName = args[len(args)-1]
	saConfig.Resources = args[:len(args)-1]
	saConfig.PatchErrors = []error{}
	saConfig.PatchFunc = func(info *resource.Info) ([]byte, error) {
		f.UpdatePodSpecForObject(info.Object, func(podSpec *api.PodSpec) error {
			podSpec.ServiceAccountName = saConfig.ServiceAccountName
			return nil
		})
		return runtime.Encode(saConfig.Encoder, info.Object)
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	builder := resource.NewBuilder(saConfig.Mapper, f.CategoryExpander(), saConfig.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &saConfig.fileNameOptions).
		Flatten()
	if !saConfig.Local {
		builder = builder.
			ResourceTypeOrNameArgs(saConfig.All, saConfig.Resources...).
			Latest()
	}
	saConfig.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (saConfig *ServiceAccountConfig) Validate() error {
	if len(saConfig.Resources) < 1 && cmdutil.IsFilenameEmpty(saConfig.fileNameOptions.Filenames) {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	return nil
}

func (saConfig *ServiceAccountConfig) Run() error {

	patches := CalculatePatches(saConfig.Infos, saConfig.Encoder, saConfig.PatchFunc)
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			saConfig.PatchErrors = append(saConfig.PatchErrors, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}
		if saConfig.Local || saConfig.DryRun {
			return saConfig.PrintObject(saConfig.Cmd, saConfig.Mapper, patch.Info.Object, saConfig.Out)
		}
		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			saConfig.PatchErrors = append(saConfig.PatchErrors, fmt.Errorf("failed to patch ServiceAccountName %v", err))
			continue
		}
		info.Refresh(patched, true)
		if saConfig.Record || cmdutil.ContainsChangeCause(info) {
			if patch, patchType, err := cmdutil.ChangeResourcePatch(info, saConfig.ChangeCause); err == nil {
				if patched, err = resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, patchType, patch); err != nil {
					fmt.Fprintf(saConfig.Err, "WARNING: changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err)
				}
			}
		}
		info.Refresh(patched, true)
		if len(saConfig.Output) > 0 {
			return saConfig.PrintObject(saConfig.Cmd, saConfig.Mapper, patched, saConfig.Out)
		}
		cmdutil.PrintSuccess(saConfig.Mapper, saConfig.ShortOutput, saConfig.Out, info.Mapping.Resource, info.Name, saConfig.DryRun, "serviceaccount updated")

	}
	return utilerrors.NewAggregate(saConfig.PatchErrors)
}
