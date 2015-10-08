/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/util/validation"
)

// ExposeOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ExposeOptions struct {
	Filenames []string
}

const (
	expose_long = `Take a replication controller, service or pod and expose it as a new Kubernetes Service.

Looks up a replication controller, service or pod by name and uses the selector for that resource as the
selector for a new Service on the specified port. If no labels are specified, the new service will
re-use the labels from the resource it exposes.`

	expose_example = `# Create a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose rc nginx --port=80 --target-port=8000

# Create a service for a replication controller identified by type and name specified in "nginx-controller.yaml", which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose -f nginx-controller.yaml --port=80 --target-port=8000

# Create a service for a pod valid-pod, which serves on port 444 with the name "frontend"
$ kubectl expose pod valid-pod --port=444 --name=frontend

# Create a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
$ kubectl expose service nginx --port=443 --target-port=8443 --name=nginx-https

# Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
$ kubectl expose rc streamer --port=4100 --protocol=udp --name=video-stream`
)

func NewCmdExposeService(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ExposeOptions{}

	cmd := &cobra.Command{
		Use:     "expose (-f FILENAME | TYPE NAME) [--port=port] [--protocol=TCP|UDP] [--target-port=number-or-name] [--name=name] [----external-ip=external-ip-of-service] [--type=type]",
		Short:   "Take a replication controller, service or pod and expose it as a new Kubernetes Service",
		Long:    expose_long,
		Example: expose_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExpose(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "service/v2", "The name of the API generator to use. There are 2 generators: 'service/v1' and 'service/v2'. The only difference between them is that service port in v1 is named 'default', while it is left unnamed in v2. Default is 'service/v2'.")
	cmd.Flags().String("protocol", "TCP", "The network protocol for the service to be created. Default is 'tcp'.")
	cmd.Flags().Int("port", -1, "The port that the service should serve on. Copied from the resource being exposed, if unspecified")
	cmd.Flags().String("type", "", "Type for this service: ClusterIP, NodePort, or LoadBalancer. Default is 'ClusterIP'.")
	// TODO: remove create-external-load-balancer in code on or after Aug 25, 2016.
	cmd.Flags().Bool("create-external-load-balancer", false, "If true, create an external load balancer for this service (trumped by --type). Implementation is cloud provider dependent. Default is 'false'.")
	cmd.Flags().MarkDeprecated("create-external-load-balancer", "use --type=\"LoadBalancer\" instead")
	cmd.Flags().String("load-balancer-ip", "", "IP to assign to to the Load Balancer. If empty, an ephemeral IP will be created and used(cloud-provider specific).")
	cmd.Flags().String("selector", "", "A label selector to use for this service. If empty (the default) infer the selector from the replication controller.")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the service created by this call.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without creating it.")
	cmd.Flags().String("container-port", "", "Synonym for --target-port")
	cmd.Flags().String("target-port", "", "Name or number for the port on the container that the service should direct traffic to. Optional.")
	cmd.Flags().String("external-ip", "", "External IP address to set for the service. The service can be accessed by this IP in addition to its generated service IP.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().String("name", "", "The name for the newly created object.")
	cmd.Flags().String("session-affinity", "", "If non-empty, set the session affinity for the service to this; legal values: 'None', 'ClientIP'")

	usage := "Filename, directory, or URL to a file identifying the resource to expose a service"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	return cmd
}

func RunExpose(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *ExposeOptions) error {
	namespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Filenames...).
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
	if err := f.CanBeExposed(mapping.Kind); err != nil {
		return err
	}
	// Get the input object
	inputObject, err := r.Object()
	if err != nil {
		return err
	}

	// Get the generator, setup and validate all required parameters
	generatorName := cmdutil.GetFlagString(cmd, "generator")
	generator, found := f.Generator(generatorName)
	if !found {
		return cmdutil.UsageError(cmd, fmt.Sprintf("generator %q not found.", generatorName))
	}
	names := generator.ParamNames()
	params := kubectl.MakeParams(cmd, names)
	name := info.Name
	if len(name) > validation.DNS952LabelMaxLength {
		name = name[:validation.DNS952LabelMaxLength]
	}
	params["default-name"] = name

	// For objects that need a pod selector, derive it from the exposed object in case a user
	// didn't explicitly specify one via --selector
	if s, found := params["selector"]; found && kubectl.IsZero(s) {
		s, err := f.PodSelectorForObject(inputObject)
		if err != nil {
			return cmdutil.UsageError(cmd, fmt.Sprintf("couldn't find selectors via --selector flag or introspection: %s", err))
		}
		params["selector"] = s
	}

	if cmdutil.GetFlagInt(cmd, "port") < 1 {
		noPorts := true
		for _, param := range names {
			if param.Name == "port" {
				noPorts = false
				break
			}
		}
		if cmdutil.GetFlagInt(cmd, "port") < 0 && !noPorts {
			ports, err := f.PortsForObject(inputObject)
			if err != nil {
				return cmdutil.UsageError(cmd, fmt.Sprintf("couldn't find port via --port flag or introspection: %s", err))
			}
			switch len(ports) {
			case 0:
				return cmdutil.UsageError(cmd, "couldn't find port via --port flag or introspection")
			case 1:
				params["port"] = ports[0]
			default:
				return cmdutil.UsageError(cmd, fmt.Sprintf("multiple ports to choose from: %v, please explicitly specify a port using the --port flag.", ports))
			}
		}
	}
	if cmdutil.GetFlagBool(cmd, "create-external-load-balancer") {
		params["create-external-load-balancer"] = "true"
	}
	if kubectl.IsZero(params["labels"]) {
		labels, err := f.LabelsForObject(inputObject)
		if err != nil {
			return err
		}
		params["labels"] = kubectl.MakeLabels(labels)
	}
	if v := cmdutil.GetFlagString(cmd, "type"); v != "" {
		params["type"] = v
	}
	err = kubectl.ValidateParams(names, params)
	if err != nil {
		return err
	}

	// Expose new object
	object, err := generator.Generate(params)
	if err != nil {
		return err
	}

	inline := cmdutil.GetFlagString(cmd, "overrides")
	if len(inline) > 0 {
		object, err = cmdutil.Merge(object, inline, mapping.Kind)
		if err != nil {
			return err
		}
	}

	resourceMapper := &resource.Mapper{ObjectTyper: typer, RESTMapper: mapper, ClientMapper: f.ClientMapperForCommand()}
	info, err = resourceMapper.InfoForObject(object)
	if err != nil {
		return err
	}
	// TODO: extract this flag to a central location, when such a location exists.
	if cmdutil.GetFlagBool(cmd, "dry-run") {
		fmt.Fprintln(out, "running in dry-run mode...")
	} else {
		// Serialize the configuration into an annotation.
		if err := kubectl.UpdateApplyAnnotation(info); err != nil {
			return err
		}

		// Serialize the object with the annotation applied.
		data, err := info.Mapping.Codec.Encode(info.Object)
		if err != nil {
			return err
		}

		object, err = resource.NewHelper(info.Client, info.Mapping).Create(namespace, false, data)
		if err != nil {
			return err
		}
	}
	outputFormat := cmdutil.GetFlagString(cmd, "output")
	if outputFormat != "" {
		return f.PrintObject(cmd, object, out)
	}
	cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, "exposed")
	return nil
}
