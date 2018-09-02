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

// ImageOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type SetImageOptions struct {
	resource.FilenameOptions

	PrintFlags  *genericclioptions.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	Infos        []*resource.Info
	Selector     string
	DryRun       bool
	All          bool
	Output       string
	Local        bool
	ResolveImage ImageResolver

	PrintObj printers.ResourcePrinterFunc
	Recorder genericclioptions.Recorder

	UpdatePodSpecForObject polymorphichelpers.UpdatePodSpecForObjectFunc
	Resources              []string
	ContainerImages        map[string]string

	genericclioptions.IOStreams
}

var (
	image_resources = `
  	pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), replicaset (rs)`

	image_long = templates.LongDesc(`
		Update existing container image(s) of resources.

		Possible resources include (case insensitive):
		` + image_resources)

	image_example = templates.Examples(`
		# Set a deployment's nginx container image to 'nginx:1.9.1', and its busybox container image to 'busybox'.
		kubectl set image deployment/nginx busybox=busybox nginx=nginx:1.9.1

		# Update all deployments' and rc's nginx container's image to 'nginx:1.9.1'
		kubectl set image deployments,rc nginx=nginx:1.9.1 --all

		# Update image of all containers of daemonset abc to 'nginx:1.9.1'
		kubectl set image daemonset abc *=nginx:1.9.1

		# Print result (in yaml format) of updating nginx container image from local file, without hitting the server
		kubectl set image -f path/to/file.yaml nginx=nginx:1.9.1 --local -o yaml`)
)

func NewImageOptions(streams genericclioptions.IOStreams) *SetImageOptions {
	return &SetImageOptions{
		PrintFlags:  genericclioptions.NewPrintFlags("image updated").WithTypeSetter(scheme.Scheme),
		RecordFlags: genericclioptions.NewRecordFlags(),

		Recorder: genericclioptions.NoopRecorder{},

		IOStreams: streams,
	}
}

func NewCmdImage(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewImageOptions(streams)

	cmd := &cobra.Command{
		Use: "image (-f FILENAME | TYPE NAME) CONTAINER_NAME_1=CONTAINER_IMAGE_1 ... CONTAINER_NAME_N=CONTAINER_IMAGE_N",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Update image of a pod template"),
		Long:    image_long,
		Example: image_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)
	o.RecordFlags.AddFlags(cmd)

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmd.Flags().BoolVar(&o.All, "all", o.All, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.Local, "local", o.Local, "If true, set image will NOT contact api-server but run locally.")
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

func (o *SetImageOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.UpdatePodSpecForObject = polymorphichelpers.UpdatePodSpecForObjectFn
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.ResolveImage = resolveImageFunc

	if o.DryRun {
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

	o.Resources, o.ContainerImages, err = getResourcesAndImages(args)
	if err != nil {
		return err
	}

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		LocalParam(o.Local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.Local {
		builder.LabelSelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, o.Resources...).
			Latest()
	} else {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		if len(o.Resources) > 0 {
			return resource.LocalResourceError
		}
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (o *SetImageOptions) Validate() error {
	errors := []error{}
	if o.All && len(o.Selector) > 0 {
		errors = append(errors, fmt.Errorf("cannot set --all and --selector at the same time"))
	}
	if len(o.Resources) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		errors = append(errors, fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>"))
	}
	if len(o.ContainerImages) < 1 {
		errors = append(errors, fmt.Errorf("at least one image update is required"))
	} else if len(o.ContainerImages) > 1 && hasWildcardKey(o.ContainerImages) {
		errors = append(errors, fmt.Errorf("all containers are already specified by *, but saw more than one container_name=container_image pairs"))
	}
	return utilerrors.NewAggregate(errors)
}

func (o *SetImageOptions) Run() error {
	allErrs := []error{}

	patches := CalculatePatches(o.Infos, scheme.DefaultJSONEncoder(), func(obj runtime.Object) ([]byte, error) {
		transformed := false
		_, err := o.UpdatePodSpecForObject(obj, func(spec *v1.PodSpec) error {
			for name, image := range o.ContainerImages {
				var (
					containerFound bool
					err            error
					resolved       string
				)
				// Find the container to update, and update its image
				for i, c := range spec.Containers {
					if c.Name == name || name == "*" {
						containerFound = true
						if len(resolved) == 0 {
							if resolved, err = o.ResolveImage(image); err != nil {
								allErrs = append(allErrs, fmt.Errorf("error: unable to resolve image %q for container %q: %v", image, name, err))
								// Do not loop again if the image resolving failed for wildcard case as we
								// will report the same error again for the next container.
								if name == "*" {
									break
								}
								continue
							}
						}
						if spec.Containers[i].Image != resolved {
							spec.Containers[i].Image = resolved
							// Perform updates
							transformed = true
						}
					}
				}
				// Add a new container if not found
				if !containerFound {
					allErrs = append(allErrs, fmt.Errorf("error: unable to find container named %q", name))
				}
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
		if err := o.Recorder.Record(obj); err != nil {
			glog.V(4).Infof("error recording current command: %v", err)
		}

		return runtime.Encode(scheme.DefaultJSONEncoder(), obj)
	})

	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		// no changes
		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			continue
		}

		if o.Local || o.DryRun {
			if err := o.PrintObj(info.Object, o.Out); err != nil {
				allErrs = append(allErrs, err)
			}
			continue
		}

		// patch the change
		actual, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch, nil)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch image update to pod template: %v\n", err))
			continue
		}

		if err := o.PrintObj(actual, o.Out); err != nil {
			allErrs = append(allErrs, err)
		}
	}
	return utilerrors.NewAggregate(allErrs)
}

// getResourcesAndImages retrieves resources and container name:images pair from given args
func getResourcesAndImages(args []string) (resources []string, containerImages map[string]string, err error) {
	pairType := "image"
	resources, imageArgs, err := cmdutil.GetResourcesAndPairs(args, pairType)
	if err != nil {
		return
	}
	containerImages, _, err = cmdutil.ParsePairs(imageArgs, pairType, false)
	return
}

func hasWildcardKey(containerImages map[string]string) bool {
	_, ok := containerImages["*"]
	return ok
}

// ImageResolver is a func that receives an image name, and
// resolves it to an appropriate / compatible image name.
// Adds flexibility for future image resolving methods.
type ImageResolver func(in string) (string, error)

// implements ImageResolver
func resolveImageFunc(in string) (string, error) {
	return in, nil
}
