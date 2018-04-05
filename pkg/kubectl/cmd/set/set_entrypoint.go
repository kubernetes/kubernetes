/*
Copyright 2018 The Kubernetes Authors.

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

	"github.com/google/shlex"
	"github.com/spf13/cobra"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// EntrypointOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type EntrypointOptions struct {
	resource.FilenameOptions

	Mapper      meta.RESTMapper
	Infos       []*resource.Info
	Selector    string
	Out         io.Writer
	Err         io.Writer
	DryRun      bool
	ShortOutput bool
	All         bool
	Record      bool
	Output      string
	ChangeCause string
	Local       bool
	Cmd         *cobra.Command

	UpdatePodSpecForObject func(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error)
	Resources              []string
	Container              string
	Command                string
	Args                   string
	CommandChanged         bool
	ArgsChanged            bool
	CommandSlice           []string
	ArgSlice               []string
}

var (
	entrypoint_long = templates.LongDesc(`
		Update existing container entrypoint.`)

	entrypoint_example = templates.Examples(`
		# Update nginx-deployment container entrypoint command to '/bin/sh'
		kubectl set entrypoint deployment nginx-deployment nginx --command="'/bin/sh' 'ls -la'" --args="test"

		# Update nginx-deployment container entrypoint by removing arguments'
		kubectl set entrypoint deployment nginx-deployment nginx --args=""`)
)

func NewCmdEntrypoint(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &EntrypointOptions{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "entrypoint (-f FILENAME | TYPE NAME) CONTAINER_NAME --command=\"'CMD_1' ... 'CMD_N'\" --args=\"'ARG_1' ... 'ARG_N'\"",
		Short:   i18n.T("Update entrypoint of a pod template"),
		Long:    entrypoint_long,
		Example: entrypoint_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().StringVar(&options.Command, "command", options.Command, "The command to be set for this container.  For example, \"/bin/sh 'ls -la'\".")
	cmd.Flags().StringVar(&options.Args, "args", options.Args, "The command arguments for this container.  For example, 'test'.")
	cmd.Flags().BoolVar(&options.All, "all", false, "Select all resources, including uninitialized ones, in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, not including uninitialized ones, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set command will NOT contact api-server but run locally.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	return cmd
}

func (o *EntrypointOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, _ = f.Object()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Command = cmdutil.GetFlagString(cmd, "command")
	o.Args = cmdutil.GetFlagString(cmd, "args")
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.ChangeCause = f.Command(cmd, false)
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.CommandChanged = cmd.Flag("command").Changed
	o.ArgsChanged = cmd.Flag("args").Changed
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.Resources, o.Container, err = getResourcesAndContainer(args)
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

func (o *EntrypointOptions) Validate() error {
	errors := []error{}
	if len(o.Resources) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		errors = append(errors, fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>"))
	}
	if o.Container == "" {
		errors = append(errors, fmt.Errorf("name of the container to be update is required"))
	}
	if o.CommandChanged == false && o.ArgsChanged == false {
		errors = append(errors, fmt.Errorf("at least one command or argument update is required"))
	}

	var err error
	o.CommandSlice, err = shlex.Split(o.Command)
	if err != nil {
		errors = append(errors, fmt.Errorf("error: %v\n", err))
	}
	o.ArgSlice, err = shlex.Split(o.Args)
	if err != nil {
		errors = append(errors, fmt.Errorf("error: %v\n", err))
	}

	return utilerrors.NewAggregate(errors)
}

func (o *EntrypointOptions) Run() error {
	allErrs := []error{}

	patches := CalculatePatches(o.Infos, cmdutil.InternalVersionJSONEncoder(), func(info *resource.Info) ([]byte, error) {
		transformed := false
		info.Object = info.AsVersioned()
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *v1.PodSpec) error {
			containerFound := false
			// Find the container to update, and update its entrypoint
			for i, c := range spec.Containers {
				if c.Name == o.Container {
					containerFound = true
					if o.CommandChanged == true {
						transformed = true
						if o.Command == "" {
							spec.Containers[i].Command = nil
						} else {
							spec.Containers[i].Command = o.CommandSlice
						}
					}
					if o.ArgsChanged == true {
						transformed = true
						if o.Args == "" {
							spec.Containers[i].Args = nil
						} else {
							spec.Containers[i].Args = o.ArgSlice
						}
					}
				}
			}

			// Add a new container if not found
			if !containerFound {
				allErrs = append(allErrs, fmt.Errorf("error: unable to find container named %q", o.Container))
			}

			return nil
		})
		if transformed && err == nil {
			return runtime.Encode(cmdutil.InternalVersionJSONEncoder(), info.Object)
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

		if o.Local || o.DryRun {
			if err := cmdutil.PrintObject(o.Cmd, patch.Info.AsVersioned(), o.Out); err != nil {
				return err
			}
			continue
		}

		// patch the change
		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch entrypoint update to pod template: %v\n", err))
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
			if err := cmdutil.PrintObject(o.Cmd, info.AsVersioned(), o.Out); err != nil {
				return err
			}
			continue
		}
		cmdutil.PrintSuccess(o.ShortOutput, o.Out, info.Object, o.DryRun, "Entrypoint updated")
	}
	return utilerrors.NewAggregate(allErrs)
}

func getResourcesAndContainer(args []string) (resources []string, containerName string, err error) {

	if len(args) < 2 {
		err = fmt.Errorf("Resource or container name missing")
		return
	}

	resources = args[:len(args)-1]
	containerName = args[len(args)-1]
	return
}
