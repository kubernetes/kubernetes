/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/spf13/cobra"
)

const (
	expose_long = `Take a replicated application and expose it as Kubernetes Service.

Looks up a ReplicationController by name, and uses the selector for that replication controller
as the selector for a new Service on the specified port.`

	expose_example = `// Creates a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
$ kubectl expose nginx --port=80 --container-port=8000

// Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
$ kubectl expose streamer --port=4100 --protocol=udp --service-name=video-stream`
)

func (f *Factory) NewCmdExposeService(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "expose NAME --port=port [--protocol=TCP|UDP] [--container-port=number-or-name] [--service-name=name] [--public-ip=ip] [--create-external-load-balancer=bool]",
		Short:   "Take a replicated application and expose it as Kubernetes Service",
		Long:    expose_long,
		Example: expose_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExpose(f, out, cmd, args)
			util.CheckErr(err)
		},
	}
	util.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "service/v1", "The name of the API generator to use.  Default is 'service/v1'.")
	cmd.Flags().String("protocol", "TCP", "The network protocol for the service to be created. Default is 'tcp'.")
	cmd.Flags().Int("port", -1, "The port that the service should serve on. Required.")
	cmd.Flags().Bool("create-external-load-balancer", false, "If true, create an external load balancer for this service. Implementation is cloud provider dependent. Default is 'false'.")
	cmd.Flags().String("selector", "", "A label selector to use for this service. If empty (the default) infer the selector from the replication controller.")
	cmd.Flags().Bool("dry-run", false, "If true, only print the object that would be sent, without creating it.")
	cmd.Flags().String("container-port", "", "Name or number for the port on the container that the service should direct traffic to. Optional.")
	cmd.Flags().String("public-ip", "", "Name of a public IP address to set for the service. The service will be assigned this IP in addition to its generated service IP.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().String("service-name", "", "The name for the newly created service.")
	return cmd
}

func RunExpose(f *Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return util.UsageError(cmd, "<name> is required for expose")
	}

	namespace, err := f.DefaultNamespace(cmd)
	if err != nil {
		return err
	}
	client, err := f.Client(cmd)
	if err != nil {
		return err
	}

	generatorName := util.GetFlagString(cmd, "generator")

	generator, found := kubectl.Generators[generatorName]
	if !found {
		return util.UsageError(cmd, fmt.Sprintf("Generator: %s not found.", generator))
	}
	if util.GetFlagInt(cmd, "port") < 1 {
		return util.UsageError(cmd, "--port is required and must be a positive integer.")
	}
	names := generator.ParamNames()
	params := kubectl.MakeParams(cmd, names)
	if len(util.GetFlagString(cmd, "service-name")) == 0 {
		params["name"] = args[0]
	} else {
		params["name"] = util.GetFlagString(cmd, "service-name")
	}
	if _, found := params["selector"]; !found {
		rc, err := client.ReplicationControllers(namespace).Get(args[0])
		if err != nil {
			return err
		}
		params["selector"] = kubectl.MakeLabels(rc.Spec.Selector)
	}
	if util.GetFlagBool(cmd, "create-external-load-balancer") {
		params["create-external-load-balancer"] = "true"
	}

	err = kubectl.ValidateParams(names, params)
	if err != nil {
		return err
	}

	service, err := generator.Generate(params)
	if err != nil {
		return err
	}

	inline := util.GetFlagString(cmd, "overrides")
	if len(inline) > 0 {
		service, err = util.Merge(service, inline, "Service")
		if err != nil {
			return err
		}
	}

	// TODO: extract this flag to a central location, when such a location exists.
	if !util.GetFlagBool(cmd, "dry-run") {
		service, err = client.Services(namespace).Create(service.(*api.Service))
		if err != nil {
			return err
		}
	}

	return f.PrintObject(cmd, service, out)
}
