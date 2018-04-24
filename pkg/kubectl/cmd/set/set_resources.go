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

	"k8s.io/kubernetes/pkg/printers"

	"github.com/spf13/cobra"
	"k8s.io/api/core/v1"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
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
type SetResourcesOptions struct {
	resource.FilenameOptions

	PrintFlags  *printers.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	Infos             []*resource.Info
	Out               io.Writer
	Err               io.Writer
	Selector          string
	ContainerSelector string
	Output            string
	All               bool
	Local             bool

	DryRun bool

	PrintObj printers.ResourcePrinterFunc
	Recorder genericclioptions.Recorder

	Limits               string
	Requests             string
	ResourceRequirements v1.ResourceRequirements

	UpdatePodSpecForObject func(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error)
	Resources              []string
}

// NewResourcesOptions returns a ResourcesOptions indicating all containers in the selected
// pod templates are selected by default.
func NewResourcesOptions(out io.Writer, errOut io.Writer) *SetResourcesOptions {
	return &SetResourcesOptions{
		PrintFlags:  printers.NewPrintFlags("resource requirements updated"),
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},

		Out:               out,
		Err:               errOut,
		ContainerSelector: "*",
	}
}

func NewCmdResources(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	o := NewResourcesOptions(out, errOut)

	resourceTypesWithPodTemplate := []string{}
	for _, resource := range f.SuggestedPodTemplateResources() {
		resourceTypesWithPodTemplate = append(resourceTypesWithPodTemplate, resource.Resource)
	}

	cmd := &cobra.Command{
		Use: "resources (-f FILENAME | TYPE NAME)  ([--limits=LIMITS & --requests=REQUESTS]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Update resource requests/limits on objects with pod templates"),
		Long:    fmt.Sprintf(resources_long, strings.Join(resourceTypesWithPodTemplate, ", ")),
		Example: resources_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)
	o.RecordFlags.AddFlags(cmd)

	//usage := "Filename, directory, or URL to a file identifying the resource to get from the server"
	//kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmd.Flags().BoolVar(&o.All, "all", o.All, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, not including uninitialized ones,supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().StringVarP(&o.ContainerSelector, "containers", "c", o.ContainerSelector, "The names of containers in the selected pod templates to change, all containers are selected by default - may use wildcards")
	cmd.Flags().BoolVar(&o.Local, "local", o.Local, "If true, set resources will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	cmd.Flags().StringVar(&o.Limits, "limits", o.Limits, "The resource requirement requests for this container.  For example, 'cpu=100m,memory=256Mi'.  Note that server side components may assign requests depending on the server configuration, such as limit ranges.")
	cmd.Flags().StringVar(&o.Requests, "requests", o.Requests, "The resource requirement requests for this container.  For example, 'cpu=100m,memory=256Mi'.  Note that server side components may assign requests depending on the server configuration, such as limit ranges.")
	return cmd
}

func (o *SetResourcesOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.RecordFlags.Complete(f.Command(cmd, false))
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.DryRun = cmdutil.GetDryRunFlag(cmd)

	if o.DryRun {
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

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().
		Internal().
		LocalParam(o.Local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.Local {
		builder.LabelSelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	} else {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		// TODO: this should be in the builder - if someone specifies tuples, fail when
		// local is true
		if len(args) > 0 {
			return resource.LocalResourceError
		}
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}
	return nil
}

func (o *SetResourcesOptions) Validate() error {
	var err error
	if o.All && len(o.Selector) > 0 {
		return fmt.Errorf("cannot set --all and --selector at the same time")
	}
	if len(o.Limits) == 0 && len(o.Requests) == 0 {
		return fmt.Errorf("you must specify an update to requests or limits (in the form of --requests/--limits)")
	}

	o.ResourceRequirements, err = kubectl.HandleResourceRequirementsV1(map[string]string{"limits": o.Limits, "requests": o.Requests})
	if err != nil {
		return err
	}

	return nil
}

func (o *SetResourcesOptions) Run() error {
	allErrs := []error{}
	patches := CalculatePatches(o.Infos, cmdutil.InternalVersionJSONEncoder(), func(info *resource.Info) ([]byte, error) {
		transformed := false
		info.Object = info.AsVersioned()
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *v1.PodSpec) error {
			containers, _ := selectContainers(spec.Containers, o.ContainerSelector)
			if len(containers) != 0 {
				for i := range containers {
					if len(o.Limits) != 0 && len(containers[i].Resources.Limits) == 0 {
						containers[i].Resources.Limits = make(v1.ResourceList)
					}
					for key, value := range o.ResourceRequirements.Limits {
						containers[i].Resources.Limits[key] = value
					}

					if len(o.Requests) != 0 && len(containers[i].Resources.Requests) == 0 {
						containers[i].Resources.Requests = make(v1.ResourceList)
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
		if err != nil {
			return nil, err
		}
		if !transformed {
			return nil, nil
		}
		// record this change (for rollout history)
		if err := o.Recorder.Record(info.Object); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		}

		return runtime.Encode(cmdutil.InternalVersionJSONEncoder(), info.Object)
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

		if o.Local || o.DryRun {
			if err := o.PrintObj(patch.Info.AsVersioned(), o.Out); err != nil {
				return err
			}
			continue
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch limit update to pod template %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		if err := o.PrintObj(info.AsVersioned(), o.Out); err != nil {
			return err
		}
	}
	return utilerrors.NewAggregate(allErrs)
}
