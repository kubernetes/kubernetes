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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/printers"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
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
type SetServiceAccountOptions struct {
	PrintFlags  *genericclioptions.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	fileNameOptions        resource.FilenameOptions
	dryRun                 bool
	shortOutput            bool
	all                    bool
	output                 string
	local                  bool
	updatePodSpecForObject polymorphichelpers.UpdatePodSpecForObjectFunc
	infos                  []*resource.Info
	serviceAccountName     string

	PrintObj printers.ResourcePrinterFunc
	Recorder genericclioptions.Recorder

	genericclioptions.IOStreams
}

func NewSetServiceAccountOptions(streams genericclioptions.IOStreams) *SetServiceAccountOptions {
	return &SetServiceAccountOptions{
		PrintFlags:  genericclioptions.NewPrintFlags("serviceaccount updated").WithTypeSetter(scheme.Scheme),
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},

		IOStreams: streams,
	}
}

// NewCmdServiceAccount returns the "set serviceaccount" command.
func NewCmdServiceAccount(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewSetServiceAccountOptions(streams)

	cmd := &cobra.Command{
		Use: "serviceaccount (-f FILENAME | TYPE NAME) SERVICE_ACCOUNT",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"sa"},
		Short:                 i18n.T("Update ServiceAccount of a resource"),
		Long:                  serviceaccountLong,
		Example:               serviceaccountExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)
	o.RecordFlags.AddFlags(cmd)

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.fileNameOptions, usage)
	cmd.Flags().BoolVar(&o.all, "all", o.all, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&o.local, "local", o.local, "If true, set serviceaccount will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

// Complete configures serviceAccountConfig from command line args.
func (o *SetServiceAccountOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.shortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.dryRun = cmdutil.GetDryRunFlag(cmd)
	o.output = cmdutil.GetFlagString(cmd, "output")
	o.updatePodSpecForObject = polymorphichelpers.UpdatePodSpecForObjectFn

	if o.dryRun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	cmdNamespace, enforceNamespace, err := f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	if len(args) == 0 {
		return errors.New("serviceaccount is required")
	}
	o.serviceAccountName = args[len(args)-1]
	resources := args[:len(args)-1]
	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		LocalParam(o.local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.fileNameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()
	if !o.local {
		builder.ResourceTypeOrNameArgs(o.all, resources...).
			Latest()
	}
	o.infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}
	return nil
}

// Run creates and applies the patch either locally or calling apiserver.
func (o *SetServiceAccountOptions) Run() error {
	patchErrs := []error{}
	patchFn := func(obj runtime.Object) ([]byte, error) {
		_, err := o.updatePodSpecForObject(obj, func(podSpec *v1.PodSpec) error {
			podSpec.ServiceAccountName = o.serviceAccountName
			return nil
		})
		if err != nil {
			return nil, err
		}
		// record this change (for rollout history)
		if err := o.Recorder.Record(obj); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		}

		return runtime.Encode(scheme.DefaultJSONEncoder(), obj)
	}

	patches := CalculatePatches(o.infos, scheme.DefaultJSONEncoder(), patchFn)
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}
		if o.local || o.dryRun {
			if err := o.PrintObj(info.Object, o.Out); err != nil {
				patchErrs = append(patchErrs, err)
			}
			continue
		}
		actual, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch, nil)
		if err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("failed to patch ServiceAccountName %v", err))
			continue
		}

		if err := o.PrintObj(actual, o.Out); err != nil {
			patchErrs = append(patchErrs, err)
		}
	}
	return utilerrors.NewAggregate(patchErrs)
}
