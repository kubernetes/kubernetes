/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	resources_long = templates.LongDesc(`
		Specify compute resource requirements (cpu, memory) for any resource that defines a pod template.  If a pod is successfully scheduled, it is guaranteed the amount of resource requested, but may burst up to its specified limits.

		for each compute resource, if a limit is specified and a request is omitted, the request will default to the limit.

		Possible resources include (case insensitive): %s.`)

	resources_example = templates.Examples(`
		# Set a deployments nginx container cpu limits to "200m" and memory to "512Mi"
		kubectl set resources deployment nginx -c=nginx --limits=cpu=200m,memory=512Mi

		# Set the resource request and limits for all containers in nginx
		kubectl set resources deployment nginx --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi

		# Remove the resource requests for resources on containers in nginx
		kubectl set resources deployment nginx --limits=cpu=0,memory=0 --requests=cpu=0,memory=0

		# Print the result (in yaml format) of updating nginx container limits from a local, without hitting the server
		kubectl set resources -f path/to/file.yaml --limits=cpu=200m,memory=512Mi --local -o yaml`)
)

// ResourcesOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags
type ResourcesOptions struct {
	resource.FilenameOptions

	Mapper            meta.RESTMapper
	Typer             runtime.ObjectTyper
	Infos             []*resource.Info
	Encoder           runtime.Encoder
	Out               io.Writer
	Err               io.Writer
	Selector          string
	ContainerSelector string
	ShortOutput       bool
	All               bool
	Record            bool
	ChangeCause       string
	Local             bool
	Cmd               *cobra.Command

	Limits               string
	Requests             string
	ResourceRequirements api.ResourceRequirements

	PrintObject            func(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
	Resources              []string
}

func NewCmdResources(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &ResourcesOptions{
		Out: out,
		Err: errOut,
	}

	resourceTypesWithPodTemplate := []string{}
	for _, resource := range f.SuggestedPodTemplateResources() {
		resourceTypesWithPodTemplate = append(resourceTypesWithPodTemplate, resource.Resource)
	}

	cmd := &cobra.Command{
		Use:     "resources (-f FILENAME | TYPE NAME)  ([--limits=LIMITS & --requests=REQUESTS]",
		Short:   i18n.T("Update resource requests/limits on objects with pod templates"),
		Long:    fmt.Sprintf(resources_long, strings.Join(resourceTypesWithPodTemplate, ", ")),
		Example: resources_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	//usage := "Filename, directory, or URL to a file identifying the resource to get from the server"
	//kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().StringVarP(&options.ContainerSelector, "containers", "c", "*", "The names of containers in the selected pod templates to change, all containers are selected by default - may use wildcards")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set resources will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmd.Flags().StringVar(&options.Limits, "limits", options.Limits, "The resource requirement requests for this container.  For example, 'cpu=100m,memory=256Mi'.  Note that server side components may assign requests depending on the server configuration, such as limit ranges.")
	cmd.Flags().StringVar(&options.Requests, "requests", options.Requests, "The resource requirement requests for this container.  For example, 'cpu=100m,memory=256Mi'.  Note that server side components may assign requests depending on the server configuration, such as limit ranges.")
	return cmd
}

func (o *ResourcesOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, o.Typer = f.Object()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.ChangeCause = f.Command(cmd, false)
	o.PrintObject = f.PrintObject
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := resource.NewBuilder(o.Mapper, f.CategoryExpander(), o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		//FilenameParam(enforceNamespace, o.Filenames...).
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		Flatten()
	if !o.Local {
		builder = builder.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}
	return nil
}

func (o *ResourcesOptions) Validate() error {
	var err error
	if len(o.Limits) == 0 && len(o.Requests) == 0 {
		return fmt.Errorf("you must specify an update to requests or limits (in the form of --requests/--limits)")
	}

	o.ResourceRequirements, err = kubectl.HandleResourceRequirements(map[string]string{"limits": o.Limits, "requests": o.Requests})
	if err != nil {
		return err
	}

	return nil
}

func (o *ResourcesOptions) Run() error {
	allErrs := []error{}
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		transformed := false
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
			containers, _ := selectContainers(spec.Containers, o.ContainerSelector)
			if len(containers) != 0 {
				for i := range containers {
					if len(o.Limits) != 0 && len(containers[i].Resources.Limits) == 0 {
						containers[i].Resources.Limits = make(api.ResourceList)
					}
					for key, value := range o.ResourceRequirements.Limits {
						containers[i].Resources.Limits[key] = value
					}

					if len(o.Requests) != 0 && len(containers[i].Resources.Requests) == 0 {
						containers[i].Resources.Requests = make(api.ResourceList)
					}
					for key, value := range o.ResourceRequirements.Requests {
						containers[i].Resources.Requests[key] = value
					}
					transformed = true
				}
			} else {
				allErrs = append(allErrs, fmt.Errorf("error: unable to find container named %s", o.ContainerSelector))
			}
			return nil
		})
		if transformed && err == nil {
			return runtime.Encode(o.Encoder, info.Object)
		}
		return nil, err
	})

	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		//no changes
		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			allErrs = append(allErrs, fmt.Errorf("info: %s %q was not changed\n", info.Mapping.Resource, info.Name))
			continue
		}

		if o.Local || cmdutil.GetDryRunFlag(o.Cmd) {
			return o.PrintObject(o.Cmd, o.Mapper, info.Object, o.Out)
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch limit update to pod template %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		//record this change (for rollout history)
		if o.Record || cmdutil.ContainsChangeCause(info) {
			if err := cmdutil.RecordChangeCause(obj, o.ChangeCause); err == nil {
				if obj, err = resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, false, obj); err != nil {
					allErrs = append(allErrs, fmt.Errorf("changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err))
				}
			}
		}
		info.Refresh(obj, true)
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, false, "resource requirements updated")
	}
	return utilerrors.NewAggregate(allErrs)
}
