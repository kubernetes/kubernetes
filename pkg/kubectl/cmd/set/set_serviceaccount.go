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
	replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs), statefulset`

	serviceaccount_long = templates.LongDesc(`
	Update ServiceAccount of pod template resources.

	Possible resources (case insensitive) can be : 
	` + serviceaccount_resources)

	serviceaccount_example = templates.Examples(`
	# Set ReplicationController nginx-controller's ServiceAccount to serviceaccount1
	kubectl set serviceaccount replicationcontroller nginx-controller serviceaccount1

	# Short form of the above command
	kubectl set sa rc nginx-controller serviceaccount1

	# Print result in yaml format of updated nginx controller from local file, without hitting apiserver
	kubectl set sa -f nginx-rc.yaml serviceaccount1 --local --dry-run -o yaml
	`)
)

type ServiceAccountConfig struct {
	fileNameOptions resource.FilenameOptions

	mapper                 meta.RESTMapper
	encoder                runtime.Encoder
	decoder                runtime.Decoder
	out                    io.Writer
	err                    io.Writer
	dryRun                 bool
	shortOutput            bool
	all                    bool
	record                 bool
	output                 string
	changeCause            string
	local                  bool
	categoryExpander       resource.CategoryExpander
	clientMapper           resource.ClientMapper
	typer                  runtime.ObjectTyper
	namespace              string
	enforceNamespace       bool
	args                   []string
	print                  func(obj runtime.Object) error
	updatePodSpecForObject func(runtime.Object, func(*api.PodSpec) error) (bool, error)
}

func NewCmdServiceaccount(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	saConfig := &ServiceAccountConfig{
		out: out,
		err: err,
	}

	cmd := &cobra.Command{
		Use:     "serviceaccount (-f FILENAME | TYPE NAME) SERVICE_ACCOUNT",
		Aliases: []string{"sa"},
		Short:   i18n.T("Update ServiceAccount of a resource"),
		Long:    serviceaccount_long,
		Example: serviceaccount_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(saConfig.Complete(f, cmd, args))
			if err := saConfig.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			if err := saConfig.Run(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
		},
	}
	cmdutil.AddPrinterFlags(cmd)

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &saConfig.fileNameOptions, usage)
	cmd.Flags().BoolVar(&saConfig.all, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&saConfig.local, "local", false, "If true, set image will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

func (saConfig *ServiceAccountConfig) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	saConfig.mapper, saConfig.typer = f.Object()
	saConfig.encoder = f.JSONEncoder()
	saConfig.shortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	saConfig.record = cmdutil.GetRecordFlag(cmd)
	saConfig.changeCause = f.Command(cmd, false)
	saConfig.local = cmdutil.GetFlagBool(cmd, "local")
	saConfig.dryRun = cmdutil.GetDryRunFlag(cmd)
	saConfig.output = cmdutil.GetFlagString(cmd, "output")
	saConfig.args = args

	saConfig.decoder = f.Decoder(true)

	saConfig.updatePodSpecForObject = f.UpdatePodSpecForObject
	saConfig.clientMapper = resource.ClientMapperFunc(f.ClientForMapping)
	saConfig.categoryExpander = f.CategoryExpander()
	saConfig.print = func(obj runtime.Object) error {
		return f.PrintObject(cmd, saConfig.mapper, obj, saConfig.out)
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	saConfig.namespace, saConfig.enforceNamespace = cmdNamespace, enforceNamespace
	return nil
}

func (saConfig *ServiceAccountConfig) Validate() error {

	if len(saConfig.args) == 0 {
		return fmt.Errorf("serviceaccount is required")
	}
	return nil
}

func (saConfig *ServiceAccountConfig) Run() error {

	serviceAccountName := saConfig.args[len(saConfig.args)-1]
	resources := saConfig.args[:len(saConfig.args)-1]
	builder := resource.NewBuilder(saConfig.mapper, saConfig.categoryExpander, saConfig.typer, saConfig.clientMapper, saConfig.decoder).
		ContinueOnError().
		NamespaceParam(saConfig.namespace).DefaultNamespace().
		FilenameParam(saConfig.enforceNamespace, &saConfig.fileNameOptions).
		Flatten()
	if !saConfig.local {
		builder.
			ResourceTypeOrNameArgs(saConfig.all, resources...).
			Latest()
	}

	infos, err := builder.Do().Infos()
	if err != nil {
		return err
	}

	patchErrs := []error{}
	patchFn := func(info *resource.Info) ([]byte, error) {
		saConfig.updatePodSpecForObject(info.Object, func(podSpec *api.PodSpec) error {
			podSpec.ServiceAccountName = serviceAccountName
			return nil
		})
		return runtime.Encode(saConfig.encoder, info.Object)
	}

	patches := CalculatePatches(infos, saConfig.encoder, patchFn)
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}
		if saConfig.local || saConfig.dryRun {
			return saConfig.print(patch.Info.Object)
		}
		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("failed to patch ServiceAccountName %v", err))
			continue
		}
		info.Refresh(patched, true)
		if saConfig.record || cmdutil.ContainsChangeCause(info) {
			if patch, patchType, err := cmdutil.ChangeResourcePatch(info, saConfig.changeCause); err == nil {
				if patched, err = resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, patchType, patch); err != nil {
					fmt.Fprintf(saConfig.err, "WARNING: changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err)
				}
			}
		}
		info.Refresh(patched, true)
		if len(saConfig.output) > 0 {
			return saConfig.print(patched)
		}
		cmdutil.PrintSuccess(saConfig.mapper, saConfig.shortOutput, saConfig.out, info.Mapping.Resource, info.Name, saConfig.dryRun, "serviceaccount updated")

	}
	return utilerrors.NewAggregate(patchErrs)
}
