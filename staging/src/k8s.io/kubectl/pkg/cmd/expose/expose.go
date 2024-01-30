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
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	exposeResources = i18n.T(`pod (po), service (svc), replicationcontroller (rc), deployment (deploy), replicaset (rs)`)

	exposeLong = templates.LongDesc(i18n.T(`
		Expose a resource as a new Kubernetes service.

		Looks up a deployment, service, replica set, replication controller or pod by name and uses the selector
		for that resource as the selector for a new service on the specified port. A deployment or replica set
		will be exposed as a service only if its selector is convertible to a selector that service supports,
		i.e. when the selector contains only the matchLabels component. Note that if no port is specified via
		--port and the exposed resource has multiple ports, all will be re-used by the new service. Also if no
		labels are specified, the new service will re-use the labels from the resource it exposes.

		Possible resources include (case insensitive):

		`) + exposeResources)

	exposeExample = templates.Examples(i18n.T(`
		# Create a service for a replicated nginx, which serves on port 80 and connects to the containers on port 8000
		kubectl expose rc nginx --port=80 --target-port=8000

		# Create a service for a replication controller identified by type and name specified in "nginx-controller.yaml", which serves on port 80 and connects to the containers on port 8000
		kubectl expose -f nginx-controller.yaml --port=80 --target-port=8000

		# Create a service for a pod valid-pod, which serves on port 444 with the name "frontend"
		kubectl expose pod valid-pod --port=444 --name=frontend

		# Create a second service based on the above service, exposing the container port 8443 as port 443 with the name "nginx-https"
		kubectl expose service nginx --port=443 --target-port=8443 --name=nginx-https

		# Create a service for a replicated streaming application on port 4100 balancing UDP traffic and named 'video-stream'.
		kubectl expose rc streamer --port=4100 --protocol=UDP --name=video-stream

		# Create a service for a replicated nginx using replica set, which serves on port 80 and connects to the containers on port 8000
		kubectl expose rs nginx --port=80 --target-port=8000

		# Create a service for an nginx deployment, which serves on port 80 and connects to the containers on port 8000
		kubectl expose deployment nginx --port=80 --target-port=8000`))
)

// ExposeServiceOptions holds the options for kubectl expose command
type ExposeServiceOptions struct {
	cmdutil.OverrideOptions

	FilenameOptions resource.FilenameOptions
	RecordFlags     *genericclioptions.RecordFlags
	PrintFlags      *genericclioptions.PrintFlags
	PrintObj        printers.ResourcePrinterFunc

	Name        string
	DefaultName string
	Selector    string
	// Port will be used if a user specifies --port OR the exposed object as one port
	Port string
	// Ports will be used iff a user doesn't specify --port AND the exposed object has multiple ports
	Ports          string
	Labels         string
	ExternalIP     string
	LoadBalancerIP string
	Type           string
	Protocol       string
	// Protocols will be used to keep port-protocol mapping derived from exposed object
	Protocols       string
	TargetPort      string
	PortName        string
	SessionAffinity string
	ClusterIP       string

	DryRunStrategy   cmdutil.DryRunStrategy
	EnforceNamespace bool

	fieldManager string

	CanBeExposed              polymorphichelpers.CanBeExposedFunc
	MapBasedSelectorForObject func(runtime.Object) (string, error)
	PortsForObject            polymorphichelpers.PortsForObjectFunc
	ProtocolsForObject        polymorphichelpers.MultiProtocolsWithForObjectFunc

	Namespace string
	Mapper    meta.RESTMapper

	Builder          *resource.Builder
	ClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)

	Recorder genericclioptions.Recorder
	genericiooptions.IOStreams
}

// exposeServiceFlags is a struct that contains the user input flags to the command.
type ExposeServiceFlags struct {
	cmdutil.OverrideOptions
	PrintFlags  *genericclioptions.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	fieldManager string
	Protocol     string

	// Port will be used if a user specifies --port OR the exposed object as one port
	Port            string
	Type            string
	LoadBalancerIP  string
	Selector        string
	Labels          string
	TargetPort      string
	ExternalIP      string
	Name            string
	SessionAffinity string
	ClusterIP       string
	Recorder        genericclioptions.Recorder
	FilenameOptions resource.FilenameOptions
	genericiooptions.IOStreams
}

func NewExposeFlags(ioStreams genericiooptions.IOStreams) *ExposeServiceFlags {
	return &ExposeServiceFlags{
		RecordFlags: genericclioptions.NewRecordFlags(),
		PrintFlags:  genericclioptions.NewPrintFlags("exposed").WithTypeSetter(scheme.Scheme),

		Recorder:  genericclioptions.NoopRecorder{},
		IOStreams: ioStreams,
	}
}

// NewCmdExposeService is a command to expose the service from user's input
func NewCmdExposeService(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	flags := NewExposeFlags(streams)

	validArgs := []string{}
	resources := regexp.MustCompile(`\s*,`).Split(exposeResources, -1)
	for _, r := range resources {
		validArgs = append(validArgs, strings.Fields(r)[0])
	}

	cmd := &cobra.Command{
		Use:                   "expose (-f FILENAME | TYPE NAME) [--port=port] [--protocol=TCP|UDP|SCTP] [--target-port=number-or-name] [--name=name] [--external-ip=external-ip-of-service] [--type=type]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Take a replication controller, service, deployment or pod and expose it as a new Kubernetes service"),
		Long:                  exposeLong,
		Example:               exposeExample,
		ValidArgsFunction:     completion.SpecifiedResourceTypeAndNameCompletionFunc(f, validArgs),
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(cmd, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Complete(f))
			cmdutil.CheckErr(o.RunExpose(cmd, args))
		},
	}

	flags.AddFlags(cmd)
	return cmd
}

func (flags *ExposeServiceFlags) AddFlags(cmd *cobra.Command) {
	flags.PrintFlags.AddFlags(cmd)
	flags.RecordFlags.AddFlags(cmd)

	cmd.Flags().StringVar(&flags.Protocol, "protocol", flags.Protocol, i18n.T("The network protocol for the service to be created. Default is 'TCP'."))
	cmd.Flags().StringVar(&flags.Port, "port", flags.Port, i18n.T("The port that the service should serve on. Copied from the resource being exposed, if unspecified"))
	cmd.Flags().StringVar(&flags.Type, "type", flags.Type, i18n.T("Type for this service: ClusterIP, NodePort, LoadBalancer, or ExternalName. Default is 'ClusterIP'."))
	cmd.Flags().StringVar(&flags.LoadBalancerIP, "load-balancer-ip", flags.LoadBalancerIP, i18n.T("IP to assign to the LoadBalancer. If empty, an ephemeral IP will be created and used (cloud-provider specific)."))
	cmd.Flags().StringVar(&flags.Selector, "selector", flags.Selector, i18n.T("A label selector to use for this service. Only equality-based selector requirements are supported. If empty (the default) infer the selector from the replication controller or replica set.)"))
	cmd.Flags().StringVarP(&flags.Labels, "labels", "l", flags.Labels, "Labels to apply to the service created by this call.")
	cmd.Flags().StringVar(&flags.TargetPort, "target-port", flags.TargetPort, i18n.T("Name or number for the port on the container that the service should direct traffic to. Optional."))
	cmd.Flags().StringVar(&flags.ExternalIP, "external-ip", flags.ExternalIP, i18n.T("Additional external IP address (not managed by Kubernetes) to accept for the service. If this IP is routed to a node, the service can be accessed by this IP in addition to its generated service IP."))
	cmd.Flags().StringVar(&flags.Name, "name", flags.Name, i18n.T("The name for the newly created object."))
	cmd.Flags().StringVar(&flags.SessionAffinity, "session-affinity", flags.SessionAffinity, i18n.T("If non-empty, set the session affinity for the service to this; legal values: 'None', 'ClientIP'"))
	cmd.Flags().StringVar(&flags.ClusterIP, "cluster-ip", flags.ClusterIP, i18n.T("ClusterIP to be assigned to the service. Leave empty to auto-allocate, or set to 'None' to create a headless service."))

	cmdutil.AddFieldManagerFlagVar(cmd, &flags.fieldManager, "kubectl-expose")
	flags.AddOverrideFlags(cmd)

	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddApplyAnnotationFlags(cmd)

	usage := "identifying the resource to expose a service"
	cmdutil.AddFilenameOptionFlags(cmd, &flags.FilenameOptions, usage)
}

func (flags *ExposeServiceFlags) ToOptions(cmd *cobra.Command, args []string) (*ExposeServiceOptions, error) {
	dryRunStratergy, err := cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return nil, err
	}

	cmdutil.PrintFlagsWithDryRunStrategy(flags.PrintFlags, dryRunStratergy)
	printer, err := flags.PrintFlags.ToPrinter()
	if err != nil {
		return nil, err
	}

	flags.RecordFlags.Complete(cmd)
	recorder, err := flags.RecordFlags.ToRecorder()
	if err != nil {
		return nil, err
	}

	e := &ExposeServiceOptions{
		DryRunStrategy:  dryRunStratergy,
		PrintObj:        printer.PrintObj,
		Recorder:        recorder,
		IOStreams:       flags.IOStreams,
		fieldManager:    flags.fieldManager,
		PrintFlags:      flags.PrintFlags,
		RecordFlags:     flags.RecordFlags,
		FilenameOptions: flags.FilenameOptions,
		Protocol:        flags.Protocol,
		Port:            flags.Port,
		Type:            flags.Type,
		LoadBalancerIP:  flags.LoadBalancerIP,
		Selector:        flags.Selector,
		Labels:          flags.Labels,
		TargetPort:      flags.TargetPort,
		ExternalIP:      flags.ExternalIP,
		Name:            flags.Name,
		SessionAffinity: flags.SessionAffinity,
		ClusterIP:       flags.ClusterIP,
		OverrideOptions: flags.OverrideOptions,
	}
	return e, nil
}

// Complete loads data from the command line environment
func (o *ExposeServiceOptions) Complete(f cmdutil.Factory) error {
	var err error

	o.Builder = f.NewBuilder()
	o.ClientForMapping = f.ClientForMapping
	o.CanBeExposed = polymorphichelpers.CanBeExposedFn
	o.MapBasedSelectorForObject = polymorphichelpers.MapBasedSelectorForObjectFn
	o.ProtocolsForObject = polymorphichelpers.MultiProtocolsForObjectFn
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

// RunExpose retrieves the Kubernetes Object from the API server and expose it to a
// Kubernetes Service
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
		return err
	}

	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		mapping := info.ResourceMapping()
		if err := o.CanBeExposed(mapping.GroupVersionKind.GroupKind()); err != nil {
			return err
		}

		name := info.Name
		if len(name) > validation.DNS1035LabelMaxLength {
			name = name[:validation.DNS1035LabelMaxLength]
		}
		o.DefaultName = name

		// For objects that need a pod selector, derive it from the exposed object in case a user
		// didn't explicitly specify one via --selector
		if len(o.Selector) == 0 {
			s, err := o.MapBasedSelectorForObject(info.Object)
			if err != nil {
				return fmt.Errorf("couldn't retrieve selectors via --selector flag or introspection: %v", err)
			}
			o.Selector = s
		}

		isHeadlessService := o.ClusterIP == "None"

		// For objects that need a port, derive it from the exposed object in case a user
		// didn't explicitly specify one via --port
		if len(o.Port) == 0 {
			ports, err := o.PortsForObject(info.Object)
			if err != nil {
				return fmt.Errorf("couldn't find port via --port flag or introspection: %v", err)
			}
			switch len(ports) {
			case 0:
				if !isHeadlessService {
					return fmt.Errorf("couldn't find port via --port flag or introspection")
				}
			case 1:
				o.Port = ports[0]
			default:
				o.Ports = strings.Join(ports, ",")
			}
		}

		// Always try to derive protocols from the exposed object, may use
		// different protocols for different ports.
		protocolsMap, err := o.ProtocolsForObject(info.Object)
		if err != nil {
			return fmt.Errorf("couldn't find protocol via introspection: %v", err)
		}
		if protocols := makeProtocols(protocolsMap); len(protocols) > 0 {
			o.Protocols = protocols
		}

		if len(o.Labels) == 0 {
			labels, err := meta.NewAccessor().Labels(info.Object)
			if err != nil {
				return err
			}
			o.Labels = polymorphichelpers.MakeLabels(labels)
		}

		// Generate new object
		service, err := o.createService()
		if err != nil {
			return err
		}

		overrideService, err := o.NewOverrider(&corev1.Service{}).Apply(service)
		if err != nil {
			return err
		}

		if err := o.Recorder.Record(overrideService); err != nil {
			klog.V(4).Infof("error recording current command: %v", err)
		}

		if o.DryRunStrategy == cmdutil.DryRunClient {
			if meta, err := meta.Accessor(overrideService); err == nil && o.EnforceNamespace {
				meta.SetNamespace(o.Namespace)
			}
			return o.PrintObj(overrideService, o.Out)
		}
		if err := util.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), overrideService, scheme.DefaultJSONEncoder()); err != nil {
			return err
		}

		asUnstructured := &unstructured.Unstructured{}
		if err := scheme.Scheme.Convert(overrideService, asUnstructured, nil); err != nil {
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
	return err
}

func (o *ExposeServiceOptions) createService() (*corev1.Service, error) {
	if len(o.Selector) == 0 {
		return nil, fmt.Errorf("selector must be specified")
	}
	selector, err := parseLabels(o.Selector)
	if err != nil {
		return nil, err
	}

	var labels map[string]string
	if len(o.Labels) > 0 {
		labels, err = parseLabels(o.Labels)
		if err != nil {
			return nil, err
		}
	}

	name := o.Name
	if len(name) == 0 {
		name = o.DefaultName
		if len(name) == 0 {
			return nil, fmt.Errorf("name must be specified")
		}
	}

	var portProtocolMap map[string][]string
	if o.Protocols != "" {
		portProtocolMap, err = parseProtocols(o.Protocols)
		if err != nil {
			return nil, err
		}
	}

	// ports takes precedence over port since it will be
	// specified only when the user hasn't specified a port
	// via --port and the exposed object has multiple ports.
	var portString string
	portString = o.Ports
	if len(o.Ports) == 0 {
		portString = o.Port
	}

	ports := []corev1.ServicePort{}
	if len(portString) != 0 {
		portStringSlice := strings.Split(portString, ",")
		servicePortName := o.PortName
		for i, stillPortString := range portStringSlice {
			port, err := strconv.Atoi(stillPortString)
			if err != nil {
				return nil, err
			}
			name := servicePortName
			// If we are going to assign multiple ports to a service, we need to
			// generate a different name for each one.
			if len(portStringSlice) > 1 {
				name = fmt.Sprintf("port-%d", i+1)
			}
			protocol := o.Protocol

			switch {
			case len(protocol) == 0 && len(portProtocolMap) == 0:
				// Default to TCP, what the flag was doing previously.
				protocol = "TCP"
			case len(protocol) > 0 && len(portProtocolMap) > 0:
				// User has specified the --protocol while exposing a multiprotocol resource
				// We should stomp multiple protocols with the one specified ie. do nothing
			case len(protocol) == 0 && len(portProtocolMap) > 0:
				// no --protocol and we expose a multiprotocol resource
				protocol = "TCP" // have the default so we can stay sane
				if exposeProtocols, found := portProtocolMap[stillPortString]; found {
					if len(exposeProtocols) == 1 {
						protocol = exposeProtocols[0]
						break
					}
					for _, exposeProtocol := range exposeProtocols {
						name := fmt.Sprintf("port-%d-%s", i+1, strings.ToLower(exposeProtocol))
						ports = append(ports, corev1.ServicePort{
							Name:     name,
							Port:     int32(port),
							Protocol: corev1.Protocol(exposeProtocol),
						})
					}
					continue
				}
			}
			ports = append(ports, corev1.ServicePort{
				Name:     name,
				Port:     int32(port),
				Protocol: corev1.Protocol(protocol),
			})
		}
	}

	service := corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: corev1.ServiceSpec{
			Selector: selector,
			Ports:    ports,
		},
	}
	targetPortString := o.TargetPort
	if len(targetPortString) > 0 {
		targetPort := intstr.Parse(targetPortString)
		// Use the same target-port for every port
		for i := range service.Spec.Ports {
			service.Spec.Ports[i].TargetPort = targetPort
		}
	} else {
		// If --target-port or --container-port haven't been specified, this
		// should be the same as Port
		for i := range service.Spec.Ports {
			port := service.Spec.Ports[i].Port
			service.Spec.Ports[i].TargetPort = intstr.FromInt32(port)
		}
	}
	if len(o.ExternalIP) > 0 {
		service.Spec.ExternalIPs = []string{o.ExternalIP}
	}
	if len(o.Type) != 0 {
		service.Spec.Type = corev1.ServiceType(o.Type)
	}
	if service.Spec.Type == corev1.ServiceTypeLoadBalancer {
		service.Spec.LoadBalancerIP = o.LoadBalancerIP
	}
	if len(o.SessionAffinity) != 0 {
		switch corev1.ServiceAffinity(o.SessionAffinity) {
		case corev1.ServiceAffinityNone:
			service.Spec.SessionAffinity = corev1.ServiceAffinityNone
		case corev1.ServiceAffinityClientIP:
			service.Spec.SessionAffinity = corev1.ServiceAffinityClientIP
		default:
			return nil, fmt.Errorf("unknown session affinity: %s", o.SessionAffinity)
		}
	}
	if len(o.ClusterIP) != 0 {
		if o.ClusterIP == "None" {
			service.Spec.ClusterIP = corev1.ClusterIPNone
		} else {
			service.Spec.ClusterIP = o.ClusterIP
		}
	}
	return &service, nil
}

// parseLabels turns a string representation of a label set into a map[string]string
func parseLabels(labelSpec string) (map[string]string, error) {
	if len(labelSpec) == 0 {
		return nil, fmt.Errorf("no label spec passed")
	}
	labels := map[string]string{}
	labelSpecs := strings.Split(labelSpec, ",")
	for ix := range labelSpecs {
		labelSpec := strings.Split(labelSpecs[ix], "=")
		if len(labelSpec) != 2 {
			return nil, fmt.Errorf("unexpected label spec: %s", labelSpecs[ix])
		}
		if len(labelSpec[0]) == 0 {
			return nil, fmt.Errorf("unexpected empty label key")
		}
		labels[labelSpec[0]] = labelSpec[1]
	}
	return labels, nil
}

func makeProtocols(protocols map[string][]string) string {
	var out []string
	for key, value := range protocols {
		for _, s := range value {
			out = append(out, fmt.Sprintf("%s/%s", key, s))
		}
	}
	return strings.Join(out, ",")
}

// parseProtocols turns a string representation of a protocols set into a map[string]string
func parseProtocols(protocols string) (map[string][]string, error) {
	if len(protocols) == 0 {
		return nil, fmt.Errorf("no protocols passed")
	}
	portProtocolMap := map[string][]string{}
	protocolsSlice := strings.Split(protocols, ",")
	for ix := range protocolsSlice {
		portProtocol := strings.Split(protocolsSlice[ix], "/")
		if len(portProtocol) != 2 {
			return nil, fmt.Errorf("unexpected port protocol mapping: %s", protocolsSlice[ix])
		}
		if len(portProtocol[0]) == 0 {
			return nil, fmt.Errorf("unexpected empty port")
		}
		if len(portProtocol[1]) == 0 {
			return nil, fmt.Errorf("unexpected empty protocol")
		}
		port := portProtocol[0]
		portProtocolMap[port] = append(portProtocolMap[port], portProtocol[1])
	}
	return portProtocolMap, nil
}
