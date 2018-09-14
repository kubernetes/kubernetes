/*
Copyright 2015 The Kubernetes Authors.

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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/printers"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	autoscalingv1client "k8s.io/client-go/kubernetes/typed/autoscaling/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	autoscaleLong = templates.LongDesc(i18n.T(`
		Creates an autoscaler that automatically chooses and sets the number of pods that run in a kubernetes cluster.

		Looks up a Deployment, ReplicaSet, or ReplicationController by name and creates an autoscaler that uses the given resource as a reference.
		An autoscaler can automatically increase or decrease number of pods deployed within the system as needed.`))

	autoscaleExample = templates.Examples(i18n.T(`
		# Auto scale a deployment "foo", with the number of pods between 2 and 10, no target CPU utilization specified so a default autoscaling policy will be used:
		kubectl autoscale deployment foo --min=2 --max=10

		# Auto scale a replication controller "foo", with the number of pods between 1 and 5, target CPU utilization at 80%:
		kubectl autoscale rc foo --max=5 --cpu-percent=80`))
)

type AutoscaleOptions struct {
	FilenameOptions *resource.FilenameOptions

	RecordFlags *genericclioptions.RecordFlags
	Recorder    genericclioptions.Recorder

	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinter, error)

	Name       string
	Generator  string
	Min        int32
	Max        int32
	CpuPercent int32

	createAnnotation bool
	args             []string
	enforceNamespace bool
	namespace        string
	dryRun           bool
	builder          *resource.Builder
	canBeAutoscaled  polymorphichelpers.CanBeAutoscaledFunc
	generatorFunc    func(string, *meta.RESTMapping) (kubectl.StructuredGenerator, error)

	HPAClient autoscalingv1client.HorizontalPodAutoscalersGetter

	genericclioptions.IOStreams
}

func NewAutoscaleOptions(ioStreams genericclioptions.IOStreams) *AutoscaleOptions {
	return &AutoscaleOptions{
		PrintFlags:      genericclioptions.NewPrintFlags("autoscaled").WithTypeSetter(scheme.Scheme),
		FilenameOptions: &resource.FilenameOptions{},
		RecordFlags:     genericclioptions.NewRecordFlags(),
		Recorder:        genericclioptions.NoopRecorder{},

		IOStreams: ioStreams,
	}
}

func NewCmdAutoscale(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewAutoscaleOptions(ioStreams)

	validArgs := []string{"deployment", "replicaset", "replicationcontroller"}

	cmd := &cobra.Command{
		Use: "autoscale (-f FILENAME | TYPE NAME | TYPE/NAME) [--min=MINPODS] --max=MAXPODS [--cpu-percent=CPU]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Auto-scale a Deployment, ReplicaSet, or ReplicationController"),
		Long:    autoscaleLong,
		Example: autoscaleExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
		ValidArgs: validArgs,
	}

	// bind flag structs
	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().StringVar(&o.Generator, "generator", cmdutil.HorizontalPodAutoscalerV1GeneratorName, i18n.T("The name of the API generator to use. Currently there is only 1 generator."))
	cmd.Flags().Int32Var(&o.Min, "min", -1, "The lower limit for the number of pods that can be set by the autoscaler. If it's not specified or negative, the server will apply a default value.")
	cmd.Flags().Int32Var(&o.Max, "max", -1, "The upper limit for the number of pods that can be set by the autoscaler. Required.")
	cmd.MarkFlagRequired("max")
	cmd.Flags().Int32Var(&o.CpuPercent, "cpu-percent", -1, fmt.Sprintf("The target average CPU utilization (represented as a percent of requested CPU) over all the pods. If it's not specified or negative, a default autoscaling policy will be used."))
	cmd.Flags().StringVar(&o.Name, "name", "", i18n.T("The name for the newly created object. If not specified, the name of the input resource will be used."))
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddFilenameOptionFlags(cmd, o.FilenameOptions, "identifying the resource to autoscale.")
	cmdutil.AddApplyAnnotationFlags(cmd)
	return cmd
}

func (o *AutoscaleOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.dryRun = cmdutil.GetFlagBool(cmd, "dry-run")
	o.createAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)
	o.builder = f.NewBuilder()
	o.canBeAutoscaled = polymorphichelpers.CanBeAutoscaledFn
	o.args = args
	o.RecordFlags.Complete(cmd)

	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	kubeClient, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.HPAClient = kubeClient.AutoscalingV1()

	// get the generator
	o.generatorFunc = func(name string, mapping *meta.RESTMapping) (kubectl.StructuredGenerator, error) {
		switch o.Generator {
		case cmdutil.HorizontalPodAutoscalerV1GeneratorName:
			return &kubectl.HorizontalPodAutoscalerGeneratorV1{
				Name:               name,
				MinReplicas:        o.Min,
				MaxReplicas:        o.Max,
				CPUPercent:         o.CpuPercent,
				ScaleRefName:       name,
				ScaleRefKind:       mapping.GroupVersionKind.Kind,
				ScaleRefApiVersion: mapping.GroupVersionKind.GroupVersion().String(),
			}, nil
		default:
			return nil, cmdutil.UsageErrorf(cmd, "Generator %s not supported. ", o.Generator)
		}
	}

	o.namespace, o.enforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinter, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		if o.dryRun {
			o.PrintFlags.Complete("%s (dry run)")
		}

		return o.PrintFlags.ToPrinter()
	}

	return nil
}

func (o *AutoscaleOptions) Validate() error {
	if o.Max < 1 {
		return fmt.Errorf("--max=MAXPODS is required and must be at least 1, max: %d", o.Max)
	}
	if o.Max < o.Min {
		return fmt.Errorf("--max=MAXPODS must be larger or equal to --min=MINPODS, max: %d, min: %d", o.Max, o.Min)
	}

	return nil
}

func (o *AutoscaleOptions) Run() error {
	r := o.builder.
		WithScheme(legacyscheme.Scheme).
		ContinueOnError().
		NamespaceParam(o.namespace).DefaultNamespace().
		FilenameParam(o.enforceNamespace, o.FilenameOptions).
		ResourceTypeOrNameArgs(false, o.args...).
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	count := 0
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		if err := o.canBeAutoscaled(mapping.GroupVersionKind.GroupKind()); err != nil {
			return err
		}

		generator, err := o.generatorFunc(info.Name, mapping)
		if err != nil {
			return err
		}

		// Generate new object
		object, err := generator.StructuredGenerate()
		if err != nil {
			return err
		}
		hpa, ok := object.(*autoscalingv1.HorizontalPodAutoscaler)
		if !ok {
			return fmt.Errorf("generator made %T, not autoscalingv1.HorizontalPodAutoscaler", object)
		}

		if err := o.Recorder.Record(hpa); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		}

		if o.dryRun {
			count++

			printer, err := o.ToPrinter("created")
			if err != nil {
				return err
			}
			return printer.PrintObj(hpa, o.Out)
		}

		if err := kubectl.CreateOrUpdateAnnotation(o.createAnnotation, hpa, cmdutil.InternalVersionJSONEncoder()); err != nil {
			return err
		}

		actualHPA, err := o.HPAClient.HorizontalPodAutoscalers(o.namespace).Create(hpa)
		if err != nil {
			return err
		}

		count++
		printer, err := o.ToPrinter("autoscaled")
		if err != nil {
			return err
		}
		return printer.PrintObj(actualHPA, o.Out)
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to autoscale")
	}
	return nil
}
