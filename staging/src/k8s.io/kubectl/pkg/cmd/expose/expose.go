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

package expose

import (
	"regexp"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/generate"
	generateversioned "k8s.io/kubectl/pkg/generate/versioned"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	exposeResources = `pod (po), service (svc), replicationcontroller (rc), deployment (deploy), replicaset (rs)`

	exposeLong = templates.LongDesc(`
		Expose a resource as a new Kubernetes service.

		Looks up a deployment, service, replica set, replication controller or pod by name and uses the selector
		for that resource as the selector for a new service on the specified port. A deployment or replica set
		will be exposed as a service only if its selector is convertible to a selector that service supports,
		i.e. when the selector contains only the matchLabels component. Note that if no port is specified via
		--port and the exposed resource has multiple ports, all will be re-used by the new service. Also if no
		labels are specified, the new service will re-use the labels from the resource it exposes.

		Possible resources include (case insensitive):

		` + exposeResources)

	exposeExample = templates.Examples(i18n.T(`
		# Create a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000.
		kubectl expose rc nginx --port=80 --target-port=8000

		# Create a service for a replication controller identified by type and name specified in "nginx-controller.yaml", which serves on port 80 and connects to the containers on port 8000.
		kubectl expose -f nginx-controller.yaml --port=80 --target-port=8000

		# Create a service for a pod valid-pod, which serves on port 444 with the name "frontend"
		kubectl expose pod valid-pod --port=444 --name=frontend

		# Create a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
		kubectl expose service nginx --port=443 --target-port=8443 --name=nginx-https

		# Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
		kubectl expose rc streamer --port=4100 --protocol=UDP --name=video-stream

		# Create a service for a replicated nginx using replica set, which serves on port 80 and connects to the containers on port 8000.
		kubectl expose rs nginx --port=80 --target-port=8000

		# Create a service for an nginx deployment, which serves on port 80 and connects to the containers on port 8000.
		kubectl expose deployment nginx --port=80 --target-port=8000`))
)

type ExposeServiceOptions struct {
	FilenameOptions resource.FilenameOptions
	RecordFlags     *genericclioptions.RecordFlags
	PrintFlags      *genericclioptions.PrintFlags
	PrintObj        printers.ResourcePrinterFunc

	DryRunStrategy   cmdutil.DryRunStrategy
	DryRunVerifier   *resource.DryRunVerifier
	EnforceNamespace bool

	fieldManager string

	Generators                func(string) map[string]generate.Generator
	CanBeExposed              polymorphichelpers.CanBeExposedFunc
	MapBasedSelectorForObject func(runtime.Object) (string, error)
	PortsForObject            polymorphichelpers.PortsForObjectFunc
	ProtocolsForObject        func(runtime.Object) (map[string]string, error)

	Namespace string
	Mapper    meta.RESTMapper

	Builder          *resource.Builder
	ClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)

	Recorder genericclioptions.Recorder
	genericclioptions.IOStreams
}

func NewExposeServiceOptions(ioStreams genericclioptions.IOStreams) *ExposeServiceOptions {
	return &ExposeServiceOptions{
		RecordFlags: genericclioptions.NewRecordFlags(),
		PrintFlags:  genericclioptions.NewPrintFlags("exposed").WithTypeSetter(scheme.Scheme),

		Recorder:  genericclioptions.NoopRecorder{},
		IOStreams: ioStreams,
	}
}

func NewCmdExposeService(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewExposeServiceOptions(streams)

	validArgs := []string{}
	resources := regexp.MustCompile(`\s*,`).Split(exposeResources, -1)
	for _, r := range resources {
		validArgs = append(validArgs, strings.Fields(r)[0])
	}

	cmd := &cobra.Command{
		Use:                   "expose (-f FILENAME | TYPE NAME) [--port=port] [--protocol=TCP|UDP|SCTP] [--target-port=number-or-name] [--name=name] [--external-ip=external-ip-of-service] [--type=type]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Take a replication controller, service, deployment or pod and expose it as a new Kubernetes Service"),
		Long:                  exposeLong,
		Example:               exposeExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.RunExpose(cmd, args))
		},
		ValidArgs: validArgs,
	}

	o.RecordFlags.AddFlags(cmd)
	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().String("generator", "service/v2", i18n.T("The name of the API generator to use. There are 2 generators: 'service/v1' and 'service/v2'. The only difference between them is that service port in v1 is named 'default', while it is left unnamed in v2. Default is 'service/v2'."))
	cmd.Flags().String("protocol", "", i18n.T("The network protocol for the service to be created. Default is 'TCP'."))
	cmd.Flags().String("port", "", i18n.T("The port that the service should serve on. Copied from the resource being exposed, if unspecified"))
	cmd.Flags().String("type", "", i18n.T("Type for this service: ClusterIP, NodePort, LoadBalancer, or ExternalName. Default is 'ClusterIP'."))
	cmd.Flags().String("load-balancer-ip", "", i18n.T("IP to assign to the LoadBalancer. If empty, an ephemeral IP will be created and used (cloud-provider specific)."))
	cmd.Flags().String("selector", "", i18n.T("A label selector to use for this service. Only equality-based selector requirements are supported. If empty (the default) infer the selector from the replication controller or replica set.)"))
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the service created by this call.")
	cmd.Flags().String("container-port", "", i18n.T("Synonym for --target-port"))
	cmd.Flags().MarkDeprecated("container-port", "--container-port will be removed in the future, please use --target-port instead")
	cmd.Flags().String("target-port", "", i18n.T("Name or number for the port on the container that the service should direct traffic to. Optional."))
	cmd.Flags().String("external-ip", "", i18n.T("Additional external IP address (not managed by Kubernetes) to accept for the service. If this IP is routed to a node, the service can be accessed by this IP in addition to its generated service IP."))
	cmd.Flags().String("overrides", "", i18n.T("An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field."))
	cmd.Flags().String("name", "", i18n.T("The name for the newly created object."))
	cmd.Flags().String("session-affinity", "", i18n.T("If non-empty, set the session affinity for the service to this; legal values: 'None', 'ClientIP'"))
	cmd.Flags().String("cluster-ip", "", i18n.T("ClusterIP to be assigned to the service. Leave empty to auto-allocate, or set to 'None' to create a headless service."))
	cmdutil.AddFieldManagerFlagVar(cmd, &o.fieldManager, "kubectl-expose")

	usage := "identifying the resource to expose a service"
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, usage)
	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddApplyAnnotationFlags(cmd)
	return cmd
}

func (o *ExposeServiceOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	var err error
	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	discoveryClient, err := f.ToDiscoveryClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, discoveryClient)

	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.Generators = generateversioned.GeneratorFn
	o.Builder = f.NewBuilder()
	o.ClientForMapping = f.ClientForMapping
	o.CanBeExposed = polymorphichelpers.CanBeExposedFn
	o.MapBasedSelectorForObject = polymorphichelpers.MapBasedSelectorForObjectFn
	o.ProtocolsForObject = polymorphichelpers.ProtocolsForObjectFn
	o.PortsForObject = polymorphichelpers.PortsForObjectFn

	o.Mapper, err = f.ToRESTMapper()
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	return err
}

func (o *ExposeServiceOptions) RunExpose(cmd *cobra.Command, args []string) error {
	r := o.Builder.
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		ResourceTypeOrNameArgs(false, args...).
		Flatten().
		Do()
	err := r.Err()
	if err != nil {
		return cmdutil.UsageErrorf(cmd, err.Error())
	}

	// Get the generator, setup and validate all required parameters
	generatorName := cmdutil.GetFlagString(cmd, "generator")
	generators := o.Generators("expose")
	generator, found := generators[generatorName]
	if !found {
		return cmdutil.UsageErrorf(cmd, "generator %q not found.", generatorName)
	}
	names := generator.ParamNames()

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		if err := o.CanBeExposed(mapping.GroupVersionKind.GroupKind()); err != nil {
			return err
		}

		params := generate.MakeParams(cmd, names)
		name := info.Name
		if len(name) > validation.DNS1035LabelMaxLength {
			name = name[:validation.DNS1035LabelMaxLength]
		}
		params["default-name"] = name

		// For objects that need a pod selector, derive it from the exposed object in case a user
		// didn't explicitly specify one via --selector
		if s, found := params["selector"]; found && generate.IsZero(s) {
			s, err := o.MapBasedSelectorForObject(info.Object)
			if err != nil {
				return cmdutil.UsageErrorf(cmd, "couldn't retrieve selectors via --selector flag or introspection: %v", err)
			}
			params["selector"] = s
		}

		isHeadlessService := params["cluster-ip"] == "None"

		// For objects that need a port, derive it from the exposed object in case a user
		// didn't explicitly specify one via --port
		if port, found := params["port"]; found && generate.IsZero(port) {
			ports, err := o.PortsForObject(info.Object)
			if err != nil {
				return cmdutil.UsageErrorf(cmd, "couldn't find port via --port flag or introspection: %v", err)
			}
			switch len(ports) {
			case 0:
				if !isHeadlessService {
					return cmdutil.UsageErrorf(cmd, "couldn't find port via --port flag or introspection")
				}
			case 1:
				params["port"] = ports[0]
			default:
				params["ports"] = strings.Join(ports, ",")
			}
		}

		// Always try to derive protocols from the exposed object, may use
		// different protocols for different ports.
		if _, found := params["protocol"]; found {
			protocolsMap, err := o.ProtocolsForObject(info.Object)
			if err != nil {
				return cmdutil.UsageErrorf(cmd, "couldn't find protocol via introspection: %v", err)
			}
			if protocols := generate.MakeProtocols(protocolsMap); !generate.IsZero(protocols) {
				params["protocols"] = protocols
			}
		}

		if generate.IsZero(params["labels"]) {
			labels, err := meta.NewAccessor().Labels(info.Object)
			if err != nil {
				return err
			}
			params["labels"] = polymorphichelpers.MakeLabels(labels)
		}
		if err = generate.ValidateParams(names, params); err != nil {
			return err
		}
		// Check for invalid flags used against the present generator.
		if err := generate.EnsureFlagsValid(cmd, generators, generatorName); err != nil {
			return err
		}

		// Generate new object
		object, err := generator.Generate(params)
		if err != nil {
			return err
		}

		if inline := cmdutil.GetFlagString(cmd, "overrides"); len(inline) > 0 {
			codec := runtime.NewCodec(scheme.DefaultJSONEncoder(), scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...))
			object, err = cmdutil.Merge(codec, object, inline)
			if err != nil {
				return err
			}
		}

		if err := o.Recorder.Record(object); err != nil {
			klog.V(4).Infof("error recording current command: %v", err)
		}

		if o.DryRunStrategy == cmdutil.DryRunClient {
			return o.PrintObj(object, o.Out)
		}
		if err := util.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), object, scheme.DefaultJSONEncoder()); err != nil {
			return err
		}

		asUnstructured := &unstructured.Unstructured{}
		if err := scheme.Scheme.Convert(object, asUnstructured, nil); err != nil {
			return err
		}
		gvks, _, err := unstructuredscheme.NewUnstructuredObjectTyper().ObjectKinds(asUnstructured)
		if err != nil {
			return err
		}
		objMapping, err := o.Mapper.RESTMapping(gvks[0].GroupKind(), gvks[0].Version)
		if err != nil {
			return err
		}
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(objMapping.GroupVersionKind); err != nil {
				return err
			}
		}
		// Serialize the object with the annotation applied.
		client, err := o.ClientForMapping(objMapping)
		if err != nil {
			return err
		}
		actualObject, err := resource.
			NewHelper(client, objMapping).
			DryRun(o.DryRunStrategy == cmdutil.DryRunServer).
			WithFieldManager(o.fieldManager).
			Create(o.Namespace, false, asUnstructured)
		if err != nil {
			return err
		}

		return o.PrintObj(actualObject, o.Out)
	})
	if err != nil {
		return err
	}
	return nil
}
