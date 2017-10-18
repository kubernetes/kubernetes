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

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// ImageOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ImageOptions struct {
	resource.FilenameOptions

	Mapper       meta.RESTMapper
	Typer        runtime.ObjectTyper
	Infos        []*resource.Info
	Encoder      runtime.Encoder
	Selector     string
	Out          io.Writer
	Err          io.Writer
	DryRun       bool
	ShortOutput  bool
	All          bool
	Record       bool
	Output       string
	ChangeCause  string
	Local        bool
	Cmd          *cobra.Command
	ResolveImage func(in string) (string, error)

	PrintObject            func(cmd *cobra.Command, isLocal bool, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
	Resources              []string
	ContainerImages        map[string]string
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

func NewCmdImage(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &ImageOptions{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "image (-f FILENAME | TYPE NAME) CONTAINER_NAME_1=CONTAINER_IMAGE_1 ... CONTAINER_NAME_N=CONTAINER_IMAGE_N",
		Short:   i18n.T("Update image of a pod template"),
		Long:    image_long,
		Example: image_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set image will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

func (o *ImageOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, o.Typer = f.Object()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.ChangeCause = f.Command(cmd, false)
	o.PrintObject = f.PrintObject
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.ResolveImage = f.ResolveImage
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.Resources, o.ContainerImages, err = getResourcesAndImages(args)
	if err != nil {
		return err
	}

	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.Local {
		builder = builder.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, o.Resources...).
			Latest()
	} else {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		if len(o.Resources) > 0 {
			return resource.LocalResourceError
		}

		builder = builder.Local(f.ClientForMapping)
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (o *ImageOptions) Validate() error {
	errors := []error{}
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

func (o *ImageOptions) Run() error {
	allErrs := []error{}

	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		transformed := false
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
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
		if transformed && err == nil {
			// TODO: switch UpdatePodSpecForObject to work on v1.PodSpec, use info.VersionedObject, and avoid conversion completely
			versionedEncoder := legacyscheme.Codecs.EncoderForVersion(o.Encoder, info.Mapping.GroupVersionKind.GroupVersion())
			return runtime.Encode(versionedEncoder, info.Object)
		}
		return nil, err
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

		if o.PrintObject != nil && (o.Local || o.DryRun) {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, info.Object, o.Out); err != nil {
				return err
			}
			continue
		}

		// patch the change
		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch image update to pod template: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		// record this change (for rollout history)
		if o.Record || cmdutil.ContainsChangeCause(info) {
			if patch, patchType, err := cmdutil.ChangeResourcePatch(info, o.ChangeCause); err == nil {
				if obj, err = resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, patchType, patch); err != nil {
					fmt.Fprintf(o.Err, "WARNING: changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err)
				}
			}
		}

		info.Refresh(obj, true)

		if len(o.Output) > 0 {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, obj, o.Out); err != nil {
				return err
			}
			continue
		}
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, o.DryRun, "image updated")
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
