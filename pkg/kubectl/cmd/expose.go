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

	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

const (
	expose_long = `Take a replicated application and expose it as Kubernetes Service.

Looks up a replication controller or service by name and uses the selector for that resource as the
selector for a new Service on the specified port. If no labels are specified, the new service will
re-use the labels from the resource it exposes.`

	expose_example = `// Creates a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose rc nginx --port=80 --target-port=8000

// Creates a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
$ kubectl expose service nginx --port=443 --target-port=8443 --name=nginx-https

// Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
$ kubectl expose rc streamer --port=4100 --protocol=udp --name=video-stream`
)

func NewCmdExposeService(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "expose TYPE NAME --port=port [--protocol=TCP|UDP] [--target-port=number-or-name] [--name=name] [--public-ip=ip] [--type=type]",
		Short:   "Take a replicated application and expose it as Kubernetes Service",
		Long:    expose_long,
		Example: expose_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExpose(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "service/v2", "The name of the API generator to use. There are 2 generators: 'service/v1' and 'service/v2'. The only difference between them is that service port in v1 is named 'default', while it is left unnamed in v2. Default is 'service/v2'.")
	cmd.Flags().String("protocol", "TCP", "The network protocol for the service to be created. Default is 'tcp'.")
	cmd.Flags().Int("port", -1, "The port that the service should serve on. Copied from the resource being exposed, if unspecified")
	cmd.MarkFlagRequired("port")
	cmd.Flags().String("type", "", "Type for this service: ClusterIP, NodePort, or LoadBalancer. Default is 'ClusterIP' unless --create-external-load-balancer is specified.")
	cmd.Flags().Bool("create-external-load-balancer", false, "If true, create an external load balancer for this service (trumped by --type). Implementation is cloud provider dependent. Default is 'false'.")
	cmd.Flags().String("selector", "", "A label selector to use for this service. If empty (the default) infer the selector from the replication controller.")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the service created by this call.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without creating it.")
	cmd.Flags().String("container-port", "", "Synonym for --target-port")
	cmd.Flags().String("target-port", "", "Name or number for the port on the container that the service should direct traffic to. Optional.")
	cmd.Flags().String("public-ip", "", "Name of a public IP address to set for the service. The service will be assigned this IP in addition to its generated service IP.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().String("name", "", "The name for the newly created object.")
	cmd.Flags().String("session-affinity", "", "If non-empty, set the session affinity for the service to this; legal values: 'None', 'ClientIP'")
	return cmd
}

func RunExpose(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object()
	r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	mapping, err := r.ResourceMapping()
	if err != nil {
		return err
	}
	infos, err := r.Infos()
	if err != nil {
		return err
	}
	if len(infos) > 1 {
		return fmt.Errorf("multiple resources provided: %v", args)
	}
	info := infos[0]

	// Get the input object
	client, err := f.RESTClient(mapping)
	if err != nil {
		return err
	}
	inputObject, err := resource.NewHelper(client, mapping).Get(info.Namespace, info.Name)
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
	params["default-name"] = info.Name
	if s, found := params["selector"]; !found || len(s) == 0 || cmdutil.GetFlagInt(cmd, "port") < 1 {
		if len(s) == 0 {
			s, err := f.PodSelectorForObject(inputObject)
			if err != nil {
				return cmdutil.UsageError(cmd, fmt.Sprintf("couldn't find selectors via --selector flag or introspection: %s", err))
			}
			params["selector"] = s
		}
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
	if len(params["labels"]) == 0 {
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

	// TODO: extract this flag to a central location, when such a location exists.
	if !cmdutil.GetFlagBool(cmd, "dry-run") {
		resourceMapper := &resource.Mapper{typer, mapper, f.ClientMapperForCommand()}
		info, err := resourceMapper.InfoForObject(object)
		if err != nil {
			return err
		}
		data, err := info.Mapping.Codec.Encode(object)
		if err != nil {
			return err
		}
		_, err = resource.NewHelper(info.Client, info.Mapping).Create(namespace, false, data)
		if err != nil {
			return err
		}
	}

	return f.PrintObject(cmd, object, out)
}
