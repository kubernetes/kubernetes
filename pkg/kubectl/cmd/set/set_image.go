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

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

// ImageOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ImageOptions struct {
	Mapper      meta.RESTMapper
	Typer       runtime.ObjectTyper
	Infos       []*resource.Info
	Encoder     runtime.Encoder
	Selector    string
	Out         io.Writer
	Err         io.Writer
	Filenames   []string
	Recursive   bool
	ShortOutput bool
	All         bool
	Record      bool
	ChangeCause string
	Local       bool
	Cmd         *cobra.Command

	PrintObject            func(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
	Resources              []string
	ContainerImages        map[string]string
}

var (
	image_resources = `
  pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs)`

	image_long = dedent.Dedent(`
		Update existing container image(s) of resources.

		Possible resources include (case insensitive):`) + image_resources

	image_example = dedent.Dedent(`
		# Set a deployment's nginx container image to 'nginx:1.9.1', and its busybox container image to 'busybox'.
		kubectl set image deployment/nginx busybox=busybox nginx=nginx:1.9.1

		# Update all deployments' and rc's nginx container's image to 'nginx:1.9.1'
		kubectl set image deployments,rc nginx=nginx:1.9.1 --all

		# Update image of all containers of daemonset abc to 'nginx:1.9.1'
		kubectl set image daemonset abc *=nginx:1.9.1

		# Print result (in yaml format) of updating nginx container image from local file, without hitting the server 
		kubectl set image -f path/to/file.yaml nginx=nginx:1.9.1 --local -o yaml`)
)

func NewCmdImage(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ImageOptions{
		Out: out,
	}

	cmd := &cobra.Command{
		Use:     "image (-f FILENAME | TYPE NAME) CONTAINER_NAME_1=CONTAINER_IMAGE_1 ... CONTAINER_NAME_N=CONTAINER_IMAGE_N",
		Short:   "Update image of a pod template",
		Long:    image_long,
		Example: image_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.Flags().BoolVar(&options.All, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set image will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	return cmd
}

func (o *ImageOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, o.Typer = f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.ChangeCause = f.Command()
	o.PrintObject = f.PrintObject
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.Resources, o.ContainerImages, err = getResourcesAndImages(args)
	if err != nil {
		return err
	}

	builder := resource.NewBuilder(o.Mapper, o.Typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, o.Recursive, o.Filenames...).
		Flatten()
	if !o.Local {
		builder = builder.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, o.Resources...).
			Latest()
	}
	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	return nil
}

func (o *ImageOptions) Validate() error {
	if len(o.Resources) < 1 && len(o.Filenames) == 0 {
		return fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}
	if len(o.ContainerImages) < 1 {
		return fmt.Errorf("at least one image update is required")
	} else if len(o.ContainerImages) > 1 && hasWildcardKey(o.ContainerImages) {
		return fmt.Errorf("all containers are already specified by *, but saw more than one container_name=container_image pairs")
	}
	return nil
}

func (o *ImageOptions) Run() error {
	allErrs := []error{}

	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) (bool, error) {
		transformed := false
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
			for name, image := range o.ContainerImages {
				containerFound := false
				// Find the container to update, and update its image
				for i, c := range spec.Containers {
					if c.Name == name || name == "*" {
						spec.Containers[i].Image = image
						containerFound = true
						// Perform updates
						transformed = true
					}
				}
				// Add a new container if not found
				if !containerFound {
					allErrs = append(allErrs, fmt.Errorf("error: unable to find container named %q", name))
				}
			}
			return nil
		})
		return transformed, err
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

		if o.Local {
			fmt.Fprintln(o.Out, "running in local mode...")
			return o.PrintObject(o.Cmd, o.Mapper, info.Object, o.Out)
		}

		// patch the change
		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch image update to pod template: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		// record this change (for rollout history)
		if o.Record || cmdutil.ContainsChangeCause(info) {
			if patch, err := cmdutil.ChangeResourcePatch(info, o.ChangeCause); err == nil {
				if obj, err = resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch); err != nil {
					fmt.Fprintf(o.Err, "WARNING: changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err)
				}
			}
		}

		info.Refresh(obj, true)
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, "image updated")
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
