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
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	serviceaccountResources = `
	replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs), statefulset`

	serviceaccountLong = templates.LongDesc(i18n.T(`
	Update ServiceAccount of pod template resources.

	Possible resources (case insensitive) can be: 
	` + serviceaccountResources))

	serviceaccountExample = templates.Examples(i18n.T(`
	# Set Deployment nginx-deployment's ServiceAccount to serviceaccount1
	kubectl set serviceaccount deployment nginx-deployment serviceaccount1

	# Print the result (in yaml format) of updated nginx deployment with serviceaccount from local file, without hitting apiserver
	kubectl set sa -f nginx-deployment.yaml serviceaccount1 --local --dry-run -o yaml
	`))
)

// serviceAccountConfig encapsulates the data required to perform the operation.
type serviceAccountConfig struct {
	fileNameOptions        resource.FilenameOptions
	mapper                 meta.RESTMapper
	encoder                runtime.Encoder
	out                    io.Writer
	err                    io.Writer
	dryRun                 bool
	shortOutput            bool
	all                    bool
	record                 bool
	output                 string
	changeCause            string
	local                  bool
	saPrint                func(obj runtime.Object) error
	updatePodSpecForObject func(runtime.Object, func(*api.PodSpec) error) (bool, error)
	infos                  []*resource.Info
	serviceAccountName     string
}

// NewCmdServiceAccount returns the "set serviceaccount" command.
func NewCmdServiceAccount(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	saConfig := &serviceAccountConfig{
		out: out,
		err: err,
	}

	cmd := &cobra.Command{
		Use:     "serviceaccount (-f FILENAME | TYPE NAME) SERVICE_ACCOUNT",
		Aliases: []string{"sa"},
		Short:   i18n.T("Update ServiceAccount of a resource"),
		Long:    serviceaccountLong,
		Example: serviceaccountExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(saConfig.Complete(f, cmd, args))
			cmdutil.CheckErr(saConfig.Run())
		},
	}
	cmdutil.AddPrinterFlags(cmd)

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &saConfig.fileNameOptions, usage)
	cmd.Flags().BoolVar(&saConfig.all, "all", false, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&saConfig.local, "local", false, "If true, set serviceaccount will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

// Complete configures serviceAccountConfig from command line args.
func (saConfig *serviceAccountConfig) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	saConfig.mapper, _ = f.Object()
	saConfig.encoder = f.JSONEncoder()
	saConfig.shortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	saConfig.record = cmdutil.GetRecordFlag(cmd)
	saConfig.changeCause = f.Command(cmd, false)
	saConfig.dryRun = cmdutil.GetDryRunFlag(cmd)
	saConfig.output = cmdutil.GetFlagString(cmd, "output")
	saConfig.updatePodSpecForObject = f.UpdatePodSpecForObject
	saConfig.saPrint = func(obj runtime.Object) error {
		return f.PrintObject(cmd, saConfig.local, saConfig.mapper, obj, saConfig.out)
	}
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	if len(args) == 0 {
		return errors.New("serviceaccount is required")
	}
	saConfig.serviceAccountName = args[len(args)-1]
	resources := args[:len(args)-1]
	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &saConfig.fileNameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()
	if !saConfig.local {
		builder.ResourceTypeOrNameArgs(saConfig.all, resources...).
			Latest()
	} else {
		builder = builder.Local(f.ClientForMapping)
	}
	saConfig.infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}
	return nil
}

// Run creates and applies the patch either locally or calling apiserver.
func (saConfig *serviceAccountConfig) Run() error {
	patchErrs := []error{}
	patchFn := func(info *resource.Info) ([]byte, error) {
		saConfig.updatePodSpecForObject(info.Object, func(podSpec *api.PodSpec) error {
			podSpec.ServiceAccountName = saConfig.serviceAccountName
			return nil
		})
		return runtime.Encode(saConfig.encoder, info.Object)
	}
	patches := CalculatePatches(saConfig.infos, saConfig.encoder, patchFn)
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}
		if saConfig.local || saConfig.dryRun {
			saConfig.saPrint(patch.Info.Object)
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
			saConfig.saPrint(patched)
		}
		cmdutil.PrintSuccess(saConfig.mapper, saConfig.shortOutput, saConfig.out, info.Mapping.Resource, info.Name, saConfig.dryRun, "serviceaccount updated")
	}
	return utilerrors.NewAggregate(patchErrs)
}
