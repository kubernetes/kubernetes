/*
Copyright 2014 The Kubernetes Authors.

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
	"regexp"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation"
)

// ExposeOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ExposeOptions struct {
	Filenames []string
	Recursive bool
}

var (
	expose_resources = dedent.Dedent(`
		pod (po), service (svc), replicationcontroller (rc),
		deployment (deploy), replicaset (rs)
	`)

	expose_long = dedent.Dedent(`
		Expose a resource as a new Kubernetes service.

		Looks up a deployment, service, replica set, replication controller or pod by name and uses the selector
		for that resource as the selector for a new service on the specified port. A deployment or replica set
		will be exposed as a service only if its selector is convertible to a selector that service supports,
		i.e. when the selector contains only the matchLabels component. Note that if no port is specified via
		--port and the exposed resource has multiple ports, all will be re-used by the new service. Also if no
		labels are specified, the new service will re-use the labels from the resource it exposes.

		Possible resources include (case insensitive): `) + expose_resources

	expose_example = dedent.Dedent(`
		# Create a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
		kubectl expose rc nginx --port=80 --target-port=8000

		# Create a service for a replication controller identified by type and name specified in "nginx-controller.yaml", which serves on port 80 and connects to the containers on port 8000.
		kubectl expose -f nginx-controller.yaml --port=80 --target-port=8000

		# Create a service for a pod valid-pod, which serves on port 444 with the name "frontend"
		kubectl expose pod valid-pod --port=444 --name=frontend

		# Create a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
		kubectl expose service nginx --port=443 --target-port=8443 --name=nginx-https

		# Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
		kubectl expose rc streamer --port=4100 --protocol=udp --name=video-stream

		# Create a service for a replicated nginx using replica set, which serves on port 80 and connects to the containers on port 8000.
		kubectl expose rs nginx --port=80 --target-port=8000

		# Create a service for an nginx deployment, which serves on port 80 and connects to the containers on port 8000.
		kubectl expose deployment nginx --port=80 --target-port=8000`)
)

func NewCmdExposeService(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ExposeOptions{}

	validArgs, argAliases := []string{}, []string{}
	resources := regexp.MustCompile(`\s*,`).Split(expose_resources, -1)
	for _, r := range resources {
		validArgs = append(validArgs, strings.Fields(r)[0])
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "expose (-f FILENAME | TYPE NAME) [--port=port] [--protocol=TCP|UDP] [--target-port=number-or-name] [--name=name] [--external-ip=external-ip-of-service] [--type=type]",
		Short:   "Take a replication controller, service, deployment or pod and expose it as a new Kubernetes Service",
		Long:    expose_long,
		Example: expose_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExpose(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("generator", "service/v2", "The name of the API generator to use. There are 2 generators: 'service/v1' and 'service/v2'. The only difference between them is that service port in v1 is named 'default', while it is left unnamed in v2. Default is 'service/v2'.")
	cmd.Flags().String("protocol", "", "The network protocol for the service to be created. Default is 'TCP'.")
	cmd.Flags().String("port", "", "The port that the service should serve on. Copied from the resource being exposed, if unspecified")
	cmd.Flags().String("type", "", "Type for this service: ClusterIP, NodePort, or LoadBalancer. Default is 'ClusterIP'.")
	// TODO: remove create-external-load-balancer in code on or after Aug 25, 2016.
	cmd.Flags().Bool("create-external-load-balancer", false, "If true, create an external load balancer for this service (trumped by --type). Implementation is cloud provider dependent. Default is 'false'.")
	cmd.Flags().MarkDeprecated("create-external-load-balancer", "use --type=\"LoadBalancer\" instead")
	cmd.Flags().String("load-balancer-ip", "", "IP to assign to the Load Balancer. If empty, an ephemeral IP will be created and used (cloud-provider specific).")
	cmd.Flags().String("selector", "", "A label selector to use for this service. Only equality-based selector requirements are supported. If empty (the default) infer the selector from the replication controller or replica set.")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the service created by this call.")
	cmd.Flags().String("container-port", "", "Synonym for --target-port")
	cmd.Flags().MarkDeprecated("container-port", "--container-port will be removed in the future, please use --target-port instead")
	cmd.Flags().String("target-port", "", "Name or number for the port on the container that the service should direct traffic to. Optional.")
	cmd.Flags().String("external-ip", "", "Additional external IP address (not managed by Kubernetes) to accept for the service. If this IP is routed to a node, the service can be accessed by this IP in addition to its generated service IP.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().String("name", "", "The name for the newly created object.")
	cmd.Flags().String("session-affinity", "", "If non-empty, set the session affinity for the service to this; legal values: 'None', 'ClientIP'")
	cmd.Flags().String("cluster-ip", "", "ClusterIP to be assigned to the service. Leave empty to auto-allocate, or set to 'None' to create a headless service.")

	usage := "Filename, directory, or URL to a file identifying the resource to expose a service"
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	return cmd
}

func RunExpose(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *ExposeOptions) error {
	namespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object(false)
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(namespace).DefaultNamespace().
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	// Get the generator, setup and validate all required parameters
	generatorName := cmdutil.GetFlagString(cmd, "generator")
	generators := f.Generators("expose")
	generator, found := generators[generatorName]
	if !found {
		return cmdutil.UsageError(cmd, fmt.Sprintf("generator %q not found.", generatorName))
	}
	names := generator.ParamNames()

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		if err := f.CanBeExposed(mapping.GroupVersionKind.GroupKind()); err != nil {
			return err
		}

		params := kubectl.MakeParams(cmd, names)
		name := info.Name
		if len(name) > validation.DNS1035LabelMaxLength {
			name = name[:validation.DNS1035LabelMaxLength]
		}
		params["default-name"] = name

		// For objects that need a pod selector, derive it from the exposed object in case a user
		// didn't explicitly specify one via --selector
		if s, found := params["selector"]; found && kubectl.IsZero(s) {
			s, err := f.MapBasedSelectorForObject(info.Object)
			if err != nil {
				return cmdutil.UsageError(cmd, fmt.Sprintf("couldn't retrieve selectors via --selector flag or introspection: %s", err))
			}
			params["selector"] = s
		}

		// For objects that need a port, derive it from the exposed object in case a user
		// didn't explicitly specify one via --port
		if port, found := params["port"]; found && kubectl.IsZero(port) {
			ports, err := f.PortsForObject(info.Object)
			if err != nil {
				return cmdutil.UsageError(cmd, fmt.Sprintf("couldn't find port via --port flag or introspection: %s", err))
			}
			switch len(ports) {
			case 0:
				return cmdutil.UsageError(cmd, "couldn't find port via --port flag or introspection")
			case 1:
				params["port"] = ports[0]
			default:
				params["ports"] = strings.Join(ports, ",")
			}
		}

		// Always try to derive protocols from the exposed object, may use
		// different protocols for different ports.
		if _, found := params["protocol"]; found {
			protocolsMap, err := f.ProtocolsForObject(info.Object)
			if err != nil {
				return cmdutil.UsageError(cmd, fmt.Sprintf("couldn't find protocol via introspection: %s", err))
			}
			if protocols := kubectl.MakeProtocols(protocolsMap); !kubectl.IsZero(protocols) {
				params["protocols"] = protocols
			}
		}

		if kubectl.IsZero(params["labels"]) {
			labels, err := f.LabelsForObject(info.Object)
			if err != nil {
				return err
			}
			params["labels"] = kubectl.MakeLabels(labels)
		}
		if err = kubectl.ValidateParams(names, params); err != nil {
			return err
		}
		// Check for invalid flags used against the present generator.
		if err := kubectl.EnsureFlagsValid(cmd, generators, generatorName); err != nil {
			return err
		}

		// Generate new object
		object, err := generator.Generate(params)
		if err != nil {
			return err
		}

		if inline := cmdutil.GetFlagString(cmd, "overrides"); len(inline) > 0 {
			codec := runtime.NewCodec(f.JSONEncoder(), f.Decoder(true))
			object, err = cmdutil.Merge(codec, object, inline, mapping.GroupVersionKind.Kind)
			if err != nil {
				return err
			}
		}

		resourceMapper := &resource.Mapper{
			ObjectTyper:  typer,
			RESTMapper:   mapper,
			ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),
			Decoder:      f.Decoder(true),
		}
		info, err = resourceMapper.InfoForObject(object, nil)
		if err != nil {
			return err
		}
		if cmdutil.ShouldRecord(cmd, info) {
			if err := cmdutil.RecordChangeCause(object, f.Command()); err != nil {
				return err
			}
		}
		info.Refresh(object, true)
		if cmdutil.GetDryRunFlag(cmd) {
			return f.PrintObject(cmd, mapper, object, out)
		}
		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return err
		}

		// Serialize the object with the annotation applied.
		object, err = resource.NewHelper(info.Client, info.Mapping).Create(namespace, false, object)
		if err != nil {
			return err
		}

		if len(cmdutil.GetFlagString(cmd, "output")) > 0 {
			return f.PrintObject(cmd, mapper, object, out)
		}

		cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, "exposed")
		return nil
	})
	if err != nil {
		return err
	}
	return nil
}
