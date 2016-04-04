/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/errors"

	"github.com/spf13/cobra"
)

// AutoscaleOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type AutoscaleOptions struct {
	Filenames []string
	Recursive bool
}

const (
	autoscaleLong = `Creates an autoscaler that automatically chooses and sets the number of pods that run in a kubernetes cluster.

Looks up a Deployment, ReplicaSet, or ReplicationController by name and creates an autoscaler that uses the given resource as a reference.
An autoscaler can automatically increase or decrease number of pods deployed within the system as needed.`

	autoscaleExample = `# Auto scale a deployment "foo", with the number of pods between 2 to 10, target CPU utilization at a default value that server applies:
kubectl autoscale deployment foo --min=2 --max=10

# Auto scale a replication controller "foo", with the number of pods between 1 to 5, target CPU utilization at 80%:
kubectl autoscale rc foo --max=5 --cpu-percent=80`
)

func NewCmdAutoscale(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &AutoscaleOptions{}

	cmd := &cobra.Command{
		Use:     "autoscale (-f FILENAME | TYPE NAME | TYPE/NAME) [--min=MINPODS] --max=MAXPODS [--cpu-percent=CPU] [flags]",
		Short:   "Auto-scale a Deployment, ReplicaSet, or ReplicationController",
		Long:    autoscaleLong,
		Example: autoscaleExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunAutoscale(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "horizontalpodautoscaler/v1beta1", "The name of the API generator to use. Currently there is only 1 generator.")
	cmd.Flags().Int("min", -1, "The lower limit for the number of pods that can be set by the autoscaler. If it's not specified or negative, the server will apply a default value.")
	cmd.Flags().Int("max", -1, "The upper limit for the number of pods that can be set by the autoscaler. Required.")
	cmd.MarkFlagRequired("max")
	cmd.Flags().Int("cpu-percent", -1, fmt.Sprintf("The target average CPU utilization (represented as a percent of requested CPU) over all the pods. If it's not specified or negative, the server will apply a default value."))
	cmd.Flags().String("name", "", "The name for the newly created object. If not specified, the name of the input resource will be used.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without creating it.")
	usage := "Filename, directory, or URL to a file identifying the resource to autoscale."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func RunAutoscale(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *AutoscaleOptions) error {
	namespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	// validate flags
	if err := validateFlags(cmd); err != nil {
		return err
	}

	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	infos, err := r.Infos()
	if err != nil {
		return err
	}
	if len(infos) > 1 {
		return fmt.Errorf("multiple resources provided: %v", args)
	}
	info := infos[0]
	mapping := info.ResourceMapping()
	if err := f.CanBeAutoscaled(mapping.GroupVersionKind.GroupKind()); err != nil {
		return err
	}

	// Get the generator, setup and validate all required parameters
	generatorName := cmdutil.GetFlagString(cmd, "generator")
	generators := f.Generators("autoscale")
	generator, found := generators[generatorName]
	if !found {
		return cmdutil.UsageError(cmd, fmt.Sprintf("generator %q not found.", generatorName))
	}
	names := generator.ParamNames()
	params := kubectl.MakeParams(cmd, names)
	name := info.Name
	params["default-name"] = name

	params["scaleRef-kind"] = mapping.GroupVersionKind.Kind
	params["scaleRef-name"] = name
	params["scaleRef-apiVersion"] = mapping.GroupVersionKind.GroupVersion().String()

	if err = kubectl.ValidateParams(names, params); err != nil {
		return err
	}
	// Check for invalid flags used against the present generator.
	if err := kubectl.EnsureFlagsValid(cmd, generators, generatorName); err != nil {
		return err
	}

	// Generate new object
	object, err := generator.Generate(params)
	if err != nil {
		return err
	}

	resourceMapper := &resource.Mapper{
		ObjectTyper:  typer,
		RESTMapper:   mapper,
		ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),
		Decoder:      f.Decoder(true),
	}
	hpa, err := resourceMapper.InfoForObject(object, nil)
	if err != nil {
		return err
	}
	if cmdutil.ShouldRecord(cmd, hpa) {
		if err := cmdutil.RecordChangeCause(hpa.Object, f.Command()); err != nil {
			return err
		}
		object = hpa.Object
	}
	// TODO: extract this flag to a central location, when such a location exists.
	if cmdutil.GetFlagBool(cmd, "dry-run") {
		return f.PrintObject(cmd, mapper, object, out)
	}

	if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), hpa, f.JSONEncoder()); err != nil {
		return err
	}

	object, err = resource.NewHelper(hpa.Client, hpa.Mapping).Create(namespace, false, object)
	if err != nil {
		return err
	}

	if len(cmdutil.GetFlagString(cmd, "output")) > 0 {
		return f.PrintObject(cmd, mapper, object, out)
	}
	cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, "autoscaled")
	return nil
}

func validateFlags(cmd *cobra.Command) error {
	errs := []error{}
	max, min, cpu := cmdutil.GetFlagInt(cmd, "max"), cmdutil.GetFlagInt(cmd, "min"), cmdutil.GetFlagInt(cmd, "cpu-percent")
	if max < 1 || max < min {
		errs = append(errs, fmt.Errorf("--max=MAXPODS is required, and must be at least 1 and --min=MINPODS"))
	}
	if cpu > 100 {
		errs = append(errs, fmt.Errorf("CPU utilization (%%) cannot exceed 100"))
	}
	return errors.NewAggregate(errs)
}
