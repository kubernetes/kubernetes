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

	"github.com/spf13/cobra"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	serviceaccountResources = i18n.T(`replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs), statefulset`)

	serviceaccountLong = templates.LongDesc(i18n.T(`
	Update the service account of pod template resources.

	Possible resources (case insensitive) can be:

	`) + serviceaccountResources)

	serviceaccountExample = templates.Examples(i18n.T(`
	# Set deployment nginx-deployment's service account to serviceaccount1
	kubectl set serviceaccount deployment nginx-deployment serviceaccount1

	# Print the result (in YAML format) of updated nginx deployment with the service account from local file, without hitting the API server
	kubectl set sa -f nginx-deployment.yaml serviceaccount1 --local --dry-run=client -o yaml
	`))
)

// SetServiceAccountOptions encapsulates the data required to perform the operation.
type SetServiceAccountOptions struct {
	PrintFlags  *genericclioptions.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	fileNameOptions        resource.FilenameOptions
	dryRunStrategy         cmdutil.DryRunStrategy
	dryRunVerifier         *resource.DryRunVerifier
	shortOutput            bool
	all                    bool
	output                 string
	local                  bool
	updatePodSpecForObject polymorphichelpers.UpdatePodSpecForObjectFunc
	infos                  []*resource.Info
	serviceAccountName     string
	fieldManager           string

	PrintObj printers.ResourcePrinterFunc
	Recorder genericclioptions.Recorder

	genericclioptions.IOStreams
}

// NewSetServiceAccountOptions returns an initialized SetServiceAccountOptions instance
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
		Use:                   "serviceaccount (-f FILENAME | TYPE NAME) SERVICE_ACCOUNT",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"sa"},
		Short:                 i18n.T("Update the service account of a resource"),
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
	cmd.Flags().BoolVar(&o.all, "all", o.all, "Select all resources, in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&o.local, "local", o.local, "If true, set serviceaccount will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddFieldManagerFlagVar(cmd, &o.fieldManager, "kubectl-set")
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
	o.dryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	if o.local && o.dryRunStrategy == cmdutil.DryRunServer {
		return fmt.Errorf("cannot specify --local and --dry-run=server - did you mean --dry-run=client?")
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	o.dryRunVerifier = resource.NewDryRunVerifier(dynamicClient, f.OpenAPIGetter())
	o.output = cmdutil.GetFlagString(cmd, "output")
	o.updatePodSpecForObject = polymorphichelpers.UpdatePodSpecForObjectFn

	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.dryRunStrategy)
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
	builder := f.NewBuilder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		LocalParam(o.local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.fileNameOptions).
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
			klog.V(4).Infof("error recording current command: %v", err)
		}

		return runtime.Encode(scheme.DefaultJSONEncoder(), obj)
	}

	patches := CalculatePatches(o.infos, scheme.DefaultJSONEncoder(), patchFn)
	for _, patch := range patches {
		info := patch.Info
		name := info.ObjectName()
		if patch.Err != nil {
			patchErrs = append(patchErrs, fmt.Errorf("error: %s %v\n", name, patch.Err))
			continue
		}
		if o.local || o.dryRunStrategy == cmdutil.DryRunClient {
			if err := o.PrintObj(info.Object, o.Out); err != nil {
				patchErrs = append(patchErrs, err)
			}
			continue
		}
		if o.dryRunStrategy == cmdutil.DryRunServer {
			if err := o.dryRunVerifier.HasSupport(info.Mapping.GroupVersionKind); err != nil {
				patchErrs = append(patchErrs, err)
				continue
			}
		}
		actual, err := resource.
			NewHelper(info.Client, info.Mapping).
			DryRun(o.dryRunStrategy == cmdutil.DryRunServer).
			WithFieldManager(o.fieldManager).
			Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch, nil)
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
