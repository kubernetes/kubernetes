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

	"k8s.io/kubernetes/pkg/printers"

	"github.com/spf13/cobra"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
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
type SetServiceAccountOptions struct {
	PrintFlags  *printers.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	fileNameOptions        resource.FilenameOptions
	out                    io.Writer
	err                    io.Writer
	dryRun                 bool
	shortOutput            bool
	all                    bool
	output                 string
	local                  bool
	updatePodSpecForObject func(runtime.Object, func(*v1.PodSpec) error) (bool, error)
	infos                  []*resource.Info
	serviceAccountName     string

	PrintObj printers.ResourcePrinterFunc
	Recorder genericclioptions.Recorder
}

func NewSetServiceAccountOptions(out, errOut io.Writer) *SetServiceAccountOptions {
	return &SetServiceAccountOptions{
		PrintFlags:  printers.NewPrintFlags("serviceaccount updated"),
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},

		out: out,
		err: errOut,
	}
}

// NewCmdServiceAccount returns the "set serviceaccount" command.
func NewCmdServiceAccount(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	o := NewSetServiceAccountOptions(out, errOut)

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

	o.RecordFlags.Complete(f.Command(cmd, false))
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.shortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.dryRun = cmdutil.GetDryRunFlag(cmd)
	o.output = cmdutil.GetFlagString(cmd, "output")
	o.updatePodSpecForObject = f.UpdatePodSpecForObject

	if o.dryRun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
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
		Internal().
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
	patchFn := func(info *resource.Info) ([]byte, error) {
		info.Object = info.AsVersioned()
		_, err := o.updatePodSpecForObject(info.Object, func(podSpec *v1.PodSpec) error {
			podSpec.ServiceAccountName = o.serviceAccountName
			return nil
		})
		if err != nil {
			return nil, err
		}
		// record this change (for rollout history)
		if err := o.Recorder.Record(info.Object); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		}

		return runtime.Encode(cmdutil.InternalVersionJSONEncoder(), info.Object)
	}

	patches := CalculatePatches(o.infos, cmdutil.InternalVersionJSONEncoder(), patchFn)
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}
		if o.local || o.dryRun {
			if err := o.PrintObj(patch.Info.AsVersioned(), o.out); err != nil {
				return err
			}
			continue
		}
		patched, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("failed to patch ServiceAccountName %v", err))
			continue
		}
		info.Refresh(patched, true)

		if err := o.PrintObj(info.AsVersioned(), o.out); err != nil {
			return err
		}
	}
	return utilerrors.NewAggregate(patchErrs)
}
