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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

const (
	expose_long = `Take a replicated application and expose it as Kubernetes Service.

Looks up a replication controller or service by name and uses the selector for that resource as the
selector for a new Service on the specified port.`

	expose_example = `// Creates a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose rc nginx --port=80 --target-port=8000

// Creates a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
$ kubectl expose service nginx --port=443 --target-port=8443 --service-name=nginx-https

// Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
$ kubectl expose rc streamer --port=4100 --protocol=udp --service-name=video-stream`
)

func NewCmdExposeService(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "expose RESOURCE NAME --port=port [--protocol=TCP|UDP] [--target-port=number-or-name] [--service-name=name] [--public-ip=ip] [--create-external-load-balancer=bool]",
		Short:   "Take a replicated application and expose it as Kubernetes Service",
		Long:    expose_long,
		Example: expose_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExpose(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "service/v1", "The name of the API generator to use.  Default is 'service/v1'.")
	cmd.Flags().String("protocol", "TCP", "The network protocol for the service to be created. Default is 'tcp'.")
	cmd.Flags().Int("port", -1, "The port that the service should serve on. Required.")
	cmd.MarkFlagRequired("port")
	cmd.Flags().Bool("create-external-load-balancer", false, "If true, create an external load balancer for this service. Implementation is cloud provider dependent. Default is 'false'.")
	cmd.Flags().String("selector", "", "A label selector to use for this service. If empty (the default) infer the selector from the replication controller.")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the service created by this call.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without creating it.")
	cmd.Flags().String("container-port", "", "Synonym for --target-port")
	cmd.Flags().String("target-port", "", "Name or number for the port on the container that the service should direct traffic to. Optional.")
	cmd.Flags().String("public-ip", "", "Name of a public IP address to set for the service. The service will be assigned this IP in addition to its generated service IP.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().String("service-name", "", "The name for the newly created service.")
	return cmd
}

func RunExpose(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	var name, res string
	switch l := len(args); {
	case l == 2:
		res, name = args[0], args[1]
	default:
		return cmdutil.UsageError(cmd, "the type and name of a resource to expose are required arguments")
	}

	namespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	// Get the input object
	var mapping *meta.RESTMapping
	mapper, typer := f.Object()
	v, k, err := mapper.VersionAndKindForResource(res)
	if err != nil {
		return err
	}
	mapping, err = mapper.RESTMapping(k, v)
	if err != nil {
		return err
	}
	client, err := f.RESTClient(mapping)
	if err != nil {
		return err
	}
	inputObject, err := resource.NewHelper(client, mapping).Get(namespace, name)
	if err != nil {
		return err
	}

	// Get the generator, setup and validate all required parameters
	generatorName := cmdutil.GetFlagString(cmd, "generator")
	generator, found := f.Generator(generatorName)
	if !found {
		return cmdutil.UsageError(cmd, fmt.Sprintf("generator %q not found.", generator))
	}
	names := generator.ParamNames()
	params := kubectl.MakeParams(cmd, names)
	if len(cmdutil.GetFlagString(cmd, "service-name")) == 0 {
		params["name"] = name
	} else {
		params["name"] = cmdutil.GetFlagString(cmd, "service-name")
	}
	if s, found := params["selector"]; !found || len(s) == 0 || cmdutil.GetFlagInt(cmd, "port") < 1 {
		if len(s) == 0 {
			s, err := f.PodSelectorForObject(inputObject)
			if err != nil {
				return err
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
				return err
			}
			if len(ports) == 0 {
				return cmdutil.UsageError(cmd, "couldn't find a suitable port via --port flag or introspection")
			}
			if len(ports) > 1 {
				return cmdutil.UsageError(cmd, "more than one port to choose from, please explicitly specify a port using the --port flag.")
			}
			params["port"] = ports[0]
		}
	}
	if cmdutil.GetFlagBool(cmd, "create-external-load-balancer") {
		params["create-external-load-balancer"] = "true"
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
