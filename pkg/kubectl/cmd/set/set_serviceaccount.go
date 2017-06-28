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
	"errors"
	"fmt"
	"io"

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
	serviceaccountResources = `
	replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs), statefulset`

	serviceaccountLong = templates.LongDesc(`
	Update ServiceAccount of pod template resources.

	Possible resources (case insensitive) can be: 
	` + serviceaccountResources)

	serviceaccountExample = templates.Examples(`
	# Set Deployment nginx-deployment's ServiceAccount to serviceaccount1
	$kubectl set service-account deployment nginx-deployment serviceaccount1

	# Short form of the above command
	$kubectl set sa deploy nginx-deployment serviceaccount1

	# Print result in yaml format of updated nginx deployment from local file, without hitting apiserver
	$kubectl set sa -f nginx-deployment.yaml serviceaccount1 --local --dry-run -o yaml
	`)
)

// serviceAccountConfig encapsulates the data required to perform the operation.
type serviceAccountConfig struct {
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
	_print                 func(obj runtime.Object) error
	updatePodSpecForObject func(runtime.Object, func(*api.PodSpec) error) (bool, error)
}

// NewCmdServiceAccount returns the "set service-account" command.
func NewCmdServiceAccount(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	saConfig := &serviceAccountConfig{
		out: out,
		err: err,
	}

	cmd := &cobra.Command{
		Use:     "service-account (-f FILENAME | TYPE NAME) SERVICE_ACCOUNT",
		Aliases: []string{"sa"},
		Short:   i18n.T("Update ServiceAccount of a resource"),
		Long:    serviceaccountLong,
		Example: serviceaccountExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(saConfig.Complete(f, cmd, args))
			cmdutil.CheckErr(saConfig.Validate())
			cmdutil.CheckErr(saConfig.Run())
		},
	}
	cmdutil.AddPrinterFlags(cmd)

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &saConfig.fileNameOptions, usage)
	cmd.Flags().BoolVar(&saConfig.all, "all", false, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&saConfig.local, "local", false, "If true, set image will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

// Complete configures serviceAccountConfig from command line args.
func (saConfig *serviceAccountConfig) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
	saConfig._print = func(obj runtime.Object) error {
		return f.PrintObject(cmd, saConfig.mapper, obj, saConfig.out)
	}
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	saConfig.namespace, saConfig.enforceNamespace = cmdNamespace, enforceNamespace
	return nil
}

// Validate does basic validation on serviceAccountConfig
func (saConfig *serviceAccountConfig) Validate() error {
	if len(saConfig.args) == 0 {
		return errors.New("serviceaccount is required")
	}
	return nil
}

// Run creates and applies the patch either locally or calling apiserver.
func (saConfig *serviceAccountConfig) Run() error {
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
			saConfig._print(patch.Info.Object)
			continue
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
		if len(saConfig.output) > 0 {
			saConfig._print(patched)
		}
		cmdutil.PrintSuccess(saConfig.mapper, saConfig.shortOutput, saConfig.out, info.Mapping.Resource, info.Name, saConfig.dryRun, "serviceaccount updated")
	}
	return utilerrors.NewAggregate(patchErrs)
}
