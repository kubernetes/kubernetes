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
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

// PortOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags()
type PortOptions struct {
	resource.FilenameOptions

	Mapper            meta.RESTMapper
	Typer             runtime.ObjectTyper
	Infos             []*resource.Info
	Encoder           runtime.Encoder
	Selector          string
	Out               io.Writer
	Err               io.Writer
	DryRun            bool
	ShortOutput       bool
	All               bool
	Record            bool
	Output            string
	ChangeCause       string
	Local             bool
	Cmd               *cobra.Command
	port              int
	protocol          string
	overwrite         bool
	containerSelector string
	portsForObject    func(object runtime.Object) ([]string, error)

	PrintObject            func(cmd *cobra.Command, isLocal bool, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
}

var (
	portResources = `
  	pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs)`

	portLong = templates.LongDesc(`
		Update existing container port(s) of resources.

		Possible resources include (case insensitive):
		` + portResources)

	portExample = templates.Examples(`
		# Set a deployment's nginx container port to 80.
		kubectl set port deployment/nginx --port=80 -c=nginx

		# Update all deployments' and rc's nginx container's image to 'nginx:1.9.1'
		kubectl set port deployments,rc --port=80 --all

		# Update port of all containers of daemonset abc to 'nginx:1.9.1'
		kubectl set port daemonset abc --port=80

		# Print result (in yaml format) of updating container port from local file, without hitting the server
		kubectl set port -f path/to/file.yaml --port=80 --local -o yaml`)
)

func NewCmdPort(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	options := &PortOptions{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "port (-f FILENAME | TYPE NAME) [--port=PORT]",
		Short:   i18n.T("Update existing container port(s) of resources"),
		Long:    portLong,
		Example: portExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().StringVarP(&options.containerSelector, "containers", "c", "*", "The names of containers in the selected pod templates to change - may use wildcards")
	cmd.Flags().BoolVar(&options.All, "all", false, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set port will NOT contact api-server but run locally.")
	cmd.Flags().Int("port", 0, i18n.T("Name or number for the port on the container that the service should direct traffic to. Optional."))
	cmd.Flags().String("protocol", "tcp", i18n.T("The network protocol for the service to be created. Default is 'TCP'."))
	cmd.Flags().BoolVar(&options.overwrite, "overwrite", true, "If true, allow environment to be overwritten, otherwise reject updates that overwrite existing environment.")
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

func (o *PortOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Mapper, o.Typer = f.Object()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.portsForObject = f.PortsForObject
	o.Encoder = f.JSONEncoder()
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.ChangeCause = f.Command(cmd, false)
	o.PrintObject = f.PrintObject
	o.Local = cmdutil.GetFlagBool(cmd, "local")
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.port = cmdutil.GetFlagInt(cmd, "port")
	o.protocol = strings.ToUpper(cmdutil.GetFlagString(cmd, "protocol"))
	o.containerSelector = cmdutil.GetFlagString(cmd, "containers")
	o.overwrite = cmdutil.GetFlagBool(cmd, "overwrite")
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := f.NewBuilder(!o.Local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
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

func (o *PortOptions) Validate() error {
	errors := []error{}
	if o.protocol != "TCP" && o.protocol != "UDP" {
		errors = append(errors, fmt.Errorf("protocol must tcp or udp"))
	}
	if o.port <= 0 {
		errors = append(errors, fmt.Errorf("must special port and port must greater than 0"))
	}
	return utilerrors.NewAggregate(errors)
}

func (o *PortOptions) Run() error {
	allErrs := []error{}

	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		transformed := false
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
			containers, _ := selectContainers(spec.Containers, o.containerSelector)
			if len(containers) == 0 {
				fmt.Println(info.Mapping.Resource, info.Name, o.containerSelector)
				if _, err := fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.containerSelector); err != nil {
					return err
				}
				return nil
			}

			ports, err := o.portsForObject(info.Object)
			if err != nil {
				return err
			}
			if !o.overwrite {
				for _, port := range ports {
					if port == strconv.Itoa(o.port) {
						return fmt.Errorf("'%s' already has a port %v, and --overwrite is false", c.Name, port)
					}
				}
			}
			for _, c := range containers {
				for _, port := range c.Ports {
					if port.ContainerPort != int32(o.port) || string(port.Protocol) != o.protocol {
						transformed = true
					}
				}
				containerPort := api.ContainerPort{
					ContainerPort: int32(o.port),
					Protocol:      api.Protocol(o.protocol),
				}
				c.Ports = append(c.Ports, containerPort)
			}
			if err != nil {
				return err
			}
			return nil
		})
		if !transformed {
			if _, err := fmt.Fprintln(o.Out, "no resources changed"); err != nil {
				return nil, nil
			}
			return nil, err
		}
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
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, o.DryRun, "port updated")
	}
	return utilerrors.NewAggregate(allErrs)
}
