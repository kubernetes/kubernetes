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

package util

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/user"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/util/homedir"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	FlagMatchBinaryVersion = "match-server-version"
)

// Factory provides abstractions that allow the Kubectl command to be extended across multiple types
// of resources and different API sets.
// TODO: make the functions interfaces
// TODO: pass the various interfaces on the factory directly into the command constructors (so the
// commands are decoupled from the factory).
type Factory interface {
	// Returns internal flagset
	FlagSet() *pflag.FlagSet

	// Returns a discovery client
	DiscoveryClient() (discovery.CachedDiscoveryInterface, error)
	// Returns interfaces for dealing with arbitrary runtime.Objects.
	Object() (meta.RESTMapper, runtime.ObjectTyper)
	// Returns interfaces for dealing with arbitrary
	// runtime.Unstructured. This performs API calls to discover types.
	UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error)
	// Returns interfaces for decoding objects - if toInternal is set, decoded objects will be converted
	// into their internal form (if possible). Eventually the internal form will be removed as an option,
	// and only versioned objects will be returned.
	Decoder(toInternal bool) runtime.Decoder
	// Returns an encoder capable of encoding a provided object into JSON in the default desired version.
	JSONEncoder() runtime.Encoder
	// ClientSet gives you back an internal, generated clientset
	ClientSet() (*internalclientset.Clientset, error)
	// Returns a RESTClient for accessing Kubernetes resources or an error.
	RESTClient() (*restclient.RESTClient, error)
	// Returns a client.Config for accessing the Kubernetes server.
	ClientConfig() (*restclient.Config, error)
	// Returns a RESTClient for working with the specified RESTMapping or an error. This is intended
	// for working with arbitrary resources and is not guaranteed to point to a Kubernetes APIServer.
	ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error)
	// Returns a RESTClient for working with Unstructured objects.
	UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error)
	// Returns a Describer for displaying the specified RESTMapping type or an error.
	Describer(mapping *meta.RESTMapping) (kubectl.Describer, error)
	// Returns a Printer for formatting objects of the given type or an error.
	Printer(mapping *meta.RESTMapping, options kubectl.PrintOptions) (kubectl.ResourcePrinter, error)
	// Returns a Scaler for changing the size of the specified RESTMapping type or an error
	Scaler(mapping *meta.RESTMapping) (kubectl.Scaler, error)
	// Returns a Reaper for gracefully shutting down resources.
	Reaper(mapping *meta.RESTMapping) (kubectl.Reaper, error)
	// Returns a HistoryViewer for viewing change history
	HistoryViewer(mapping *meta.RESTMapping) (kubectl.HistoryViewer, error)
	// Returns a Rollbacker for changing the rollback version of the specified RESTMapping type or an error
	Rollbacker(mapping *meta.RESTMapping) (kubectl.Rollbacker, error)
	// Returns a StatusViewer for printing rollout status.
	StatusViewer(mapping *meta.RESTMapping) (kubectl.StatusViewer, error)
	// MapBasedSelectorForObject returns the map-based selector associated with the provided object. If a
	// new set-based selector is provided, an error is returned if the selector cannot be converted to a
	// map-based selector
	MapBasedSelectorForObject(object runtime.Object) (string, error)
	// PortsForObject returns the ports associated with the provided object
	PortsForObject(object runtime.Object) ([]string, error)
	// ProtocolsForObject returns the <port, protocol> mapping associated with the provided object
	ProtocolsForObject(object runtime.Object) (map[string]string, error)
	// LabelsForObject returns the labels associated with the provided object
	LabelsForObject(object runtime.Object) (map[string]string, error)
	// LogsForObject returns a request for the logs associated with the provided object
	LogsForObject(object, options runtime.Object) (*restclient.Request, error)
	// Pauser marks the object in the info as paused ie. it will not be reconciled by its controller.
	Pauser(info *resource.Info) (bool, error)
	// Resumer resumes a paused object inside the info ie. it will be reconciled by its controller.
	Resumer(info *resource.Info) (bool, error)
	// Returns a schema that can validate objects stored on disk.
	Validator(validate bool, cacheDir string) (validation.Schema, error)
	// SwaggerSchema returns the schema declaration for the provided group version kind.
	SwaggerSchema(schema.GroupVersionKind) (*swagger.ApiDeclaration, error)
	// Returns the default namespace to use in cases where no
	// other namespace is specified and whether the namespace was
	// overridden.
	DefaultNamespace() (string, bool, error)
	// Generators returns the generators for the provided command
	Generators(cmdName string) map[string]kubectl.Generator
	// Check whether the kind of resources could be exposed
	CanBeExposed(kind schema.GroupKind) error
	// Check whether the kind of resources could be autoscaled
	CanBeAutoscaled(kind schema.GroupKind) error
	// AttachablePodForObject returns the pod to which to attach given an object.
	AttachablePodForObject(object runtime.Object) (*api.Pod, error)
	// UpdatePodSpecForObject will call the provided function on the pod spec this object supports,
	// return false if no pod spec is supported, or return an error.
	UpdatePodSpecForObject(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
	// EditorEnvs returns a group of environment variables that the edit command
	// can range over in order to determine if the user has specified an editor
	// of their choice.
	EditorEnvs() []string
	// PrintObjectSpecificMessage prints object-specific messages on the provided writer
	PrintObjectSpecificMessage(obj runtime.Object, out io.Writer)

	// Command will stringify and return all environment arguments ie. a command run by a client
	// using the factory.
	Command() string
	// BindFlags adds any flags that are common to all kubectl sub commands.
	BindFlags(flags *pflag.FlagSet)
	// BindExternalFlags adds any flags defined by external projects (not part of pflags)
	BindExternalFlags(flags *pflag.FlagSet)

	DefaultResourceFilterOptions(cmd *cobra.Command, withNamespace bool) *kubectl.PrintOptions
	// DefaultResourceFilterFunc returns a collection of FilterFuncs suitable for filtering specific resource types.
	DefaultResourceFilterFunc() kubectl.Filters

	// PrintObject prints an api object given command line flags to modify the output format
	PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	// PrinterForMapping returns a printer suitable for displaying the provided resource type.
	// Requires that printer flags have been added to cmd (see AddPrinterFlags).
	PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error)
	// One stop shopping for a Builder
	NewBuilder() *resource.Builder

	// SuggestedPodTemplateResources returns a list of resource types that declare a pod template
	SuggestedPodTemplateResources() []schema.GroupResource
}

const (
	RunV1GeneratorName                          = "run/v1"
	RunPodV1GeneratorName                       = "run-pod/v1"
	ServiceV1GeneratorName                      = "service/v1"
	ServiceV2GeneratorName                      = "service/v2"
	ServiceNodePortGeneratorV1Name              = "service-nodeport/v1"
	ServiceClusterIPGeneratorV1Name             = "service-clusterip/v1"
	ServiceLoadBalancerGeneratorV1Name          = "service-loadbalancer/v1"
	ServiceExternalNameGeneratorV1Name          = "service-externalname/v1"
	ServiceAccountV1GeneratorName               = "serviceaccount/v1"
	HorizontalPodAutoscalerV1Beta1GeneratorName = "horizontalpodautoscaler/v1beta1"
	HorizontalPodAutoscalerV1GeneratorName      = "horizontalpodautoscaler/v1"
	DeploymentV1Beta1GeneratorName              = "deployment/v1beta1"
	DeploymentBasicV1Beta1GeneratorName         = "deployment-basic/v1beta1"
	JobV1Beta1GeneratorName                     = "job/v1beta1"
	JobV1GeneratorName                          = "job/v1"
	CronJobV2Alpha1GeneratorName                = "cronjob/v2alpha1"
	ScheduledJobV2Alpha1GeneratorName           = "scheduledjob/v2alpha1"
	NamespaceV1GeneratorName                    = "namespace/v1"
	ResourceQuotaV1GeneratorName                = "resourcequotas/v1"
	SecretV1GeneratorName                       = "secret/v1"
	SecretForDockerRegistryV1GeneratorName      = "secret-for-docker-registry/v1"
	SecretForTLSV1GeneratorName                 = "secret-for-tls/v1"
	ConfigMapV1GeneratorName                    = "configmap/v1"
	ClusterRoleBindingV1GeneratorName           = "clusterrolebinding.rbac.authorization.k8s.io/v1alpha1"
	ClusterV1Beta1GeneratorName                 = "cluster/v1beta1"
	PodDisruptionBudgetV1GeneratorName          = "poddisruptionbudget/v1beta1"
)

// DefaultGenerators returns the set of default generators for use in Factory instances
func DefaultGenerators(cmdName string) map[string]kubectl.Generator {
	var generator map[string]kubectl.Generator
	switch cmdName {
	case "expose":
		generator = map[string]kubectl.Generator{
			ServiceV1GeneratorName: kubectl.ServiceGeneratorV1{},
			ServiceV2GeneratorName: kubectl.ServiceGeneratorV2{},
		}
	case "service-clusterip":
		generator = map[string]kubectl.Generator{
			ServiceClusterIPGeneratorV1Name: kubectl.ServiceClusterIPGeneratorV1{},
		}
	case "service-nodeport":
		generator = map[string]kubectl.Generator{
			ServiceNodePortGeneratorV1Name: kubectl.ServiceNodePortGeneratorV1{},
		}
	case "service-loadbalancer":
		generator = map[string]kubectl.Generator{
			ServiceLoadBalancerGeneratorV1Name: kubectl.ServiceLoadBalancerGeneratorV1{},
		}
	case "deployment":
		generator = map[string]kubectl.Generator{
			DeploymentBasicV1Beta1GeneratorName: kubectl.DeploymentBasicGeneratorV1{},
		}
	case "run":
		generator = map[string]kubectl.Generator{
			RunV1GeneratorName:                kubectl.BasicReplicationController{},
			RunPodV1GeneratorName:             kubectl.BasicPod{},
			DeploymentV1Beta1GeneratorName:    kubectl.DeploymentV1Beta1{},
			JobV1Beta1GeneratorName:           kubectl.JobV1Beta1{},
			JobV1GeneratorName:                kubectl.JobV1{},
			ScheduledJobV2Alpha1GeneratorName: kubectl.CronJobV2Alpha1{},
			CronJobV2Alpha1GeneratorName:      kubectl.CronJobV2Alpha1{},
		}
	case "autoscale":
		generator = map[string]kubectl.Generator{
			HorizontalPodAutoscalerV1Beta1GeneratorName: kubectl.HorizontalPodAutoscalerV1Beta1{},
			HorizontalPodAutoscalerV1GeneratorName:      kubectl.HorizontalPodAutoscalerV1{},
		}
	case "namespace":
		generator = map[string]kubectl.Generator{
			NamespaceV1GeneratorName: kubectl.NamespaceGeneratorV1{},
		}
	case "quota":
		generator = map[string]kubectl.Generator{
			ResourceQuotaV1GeneratorName: kubectl.ResourceQuotaGeneratorV1{},
		}
	case "secret":
		generator = map[string]kubectl.Generator{
			SecretV1GeneratorName: kubectl.SecretGeneratorV1{},
		}
	case "secret-for-docker-registry":
		generator = map[string]kubectl.Generator{
			SecretForDockerRegistryV1GeneratorName: kubectl.SecretForDockerRegistryGeneratorV1{},
		}
	case "secret-for-tls":
		generator = map[string]kubectl.Generator{
			SecretForTLSV1GeneratorName: kubectl.SecretForTLSGeneratorV1{},
		}
	}

	return generator
}

func getGroupVersionKinds(gvks []schema.GroupVersionKind, group string) []schema.GroupVersionKind {
	result := []schema.GroupVersionKind{}
	for ix := range gvks {
		if gvks[ix].Group == group {
			result = append(result, gvks[ix])
		}
	}
	return result
}

func makeInterfacesFor(versionList []schema.GroupVersion) func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
	accessor := meta.NewAccessor()
	return func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
		for ix := range versionList {
			if versionList[ix].String() == version.String() {
				return &meta.VersionInterfaces{
					ObjectConvertor:  thirdpartyresourcedata.NewThirdPartyObjectConverter(api.Scheme),
					MetadataAccessor: accessor,
				}, nil
			}
		}
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", version, versionList)
	}
}

type factory struct {
	flags        *pflag.FlagSet
	clientConfig clientcmd.ClientConfig

	clients *ClientCache
}

// NewFactory creates a factory with the default Kubernetes resources defined
// if optionalClientConfig is nil, then flags will be bound to a new clientcmd.ClientConfig.
// if optionalClientConfig is not nil, then this factory will make use of it.
func NewFactory(optionalClientConfig clientcmd.ClientConfig) Factory {
	flags := pflag.NewFlagSet("", pflag.ContinueOnError)
	flags.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc) // Warn for "_" flags

	clientConfig := optionalClientConfig
	if optionalClientConfig == nil {
		clientConfig = DefaultClientConfig(flags)
	}

	clients := NewClientCache(clientConfig)

	f := &factory{
		flags:        flags,
		clientConfig: clientConfig,
		clients:      clients,
	}

	return f
}

func (f *factory) FlagSet() *pflag.FlagSet {
	return f.flags
}

func (f *factory) DiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	cfg, err := f.clientConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		return nil, err
	}
	cacheDir := computeDiscoverCacheDir(filepath.Join(homedir.HomeDir(), ".kube", "cache", "discovery"), cfg.Host)
	return NewCachedDiscoveryClient(discoveryClient, cacheDir, time.Duration(10*time.Minute)), nil
}

func (f *factory) Object() (meta.RESTMapper, runtime.ObjectTyper) {
	mapper := registered.RESTMapper()
	discoveryClient, err := f.DiscoveryClient()
	if err == nil {
		mapper = meta.FirstHitRESTMapper{
			MultiRESTMapper: meta.MultiRESTMapper{
				discovery.NewDeferredDiscoveryRESTMapper(discoveryClient, registered.InterfacesFor),
				registered.RESTMapper(), // hardcoded fall back
			},
		}
	}

	// wrap with shortcuts
	mapper = NewShortcutExpander(mapper, discoveryClient)

	// wrap with output preferences
	cfg, err := f.clients.ClientConfigForVersion(nil)
	checkErrWithPrefix("failed to get client config: ", err)
	cmdApiVersion := schema.GroupVersion{}
	if cfg.GroupVersion != nil {
		cmdApiVersion = *cfg.GroupVersion
	}
	mapper = kubectl.OutputVersionMapper{RESTMapper: mapper, OutputVersions: []schema.GroupVersion{cmdApiVersion}}
	return mapper, api.Scheme
}

func (f *factory) UnstructuredObject() (meta.RESTMapper, runtime.ObjectTyper, error) {
	discoveryClient, err := f.DiscoveryClient()
	if err != nil {
		return nil, nil, err
	}
	groupResources, err := discovery.GetAPIGroupResources(discoveryClient)
	if err != nil && !discoveryClient.Fresh() {
		discoveryClient.Invalidate()
		groupResources, err = discovery.GetAPIGroupResources(discoveryClient)
	}
	if err != nil {
		return nil, nil, err
	}

	mapper := discovery.NewDeferredDiscoveryRESTMapper(discoveryClient, meta.InterfacesForUnstructured)
	typer := discovery.NewUnstructuredObjectTyper(groupResources)
	return NewShortcutExpander(mapper, discoveryClient), typer, nil
}

func (f *factory) RESTClient() (*restclient.RESTClient, error) {
	clientConfig, err := f.clients.ClientConfigForVersion(nil)
	if err != nil {
		return nil, err
	}
	return restclient.RESTClientFor(clientConfig)
}

func (f *factory) ClientSet() (*internalclientset.Clientset, error) {
	return f.clients.ClientSetForVersion(nil)
}

func (f *factory) ClientConfig() (*restclient.Config, error) {
	return f.clients.ClientConfigForVersion(nil)
}

func (f *factory) ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	cfg, err := f.clientConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	if err := client.SetKubernetesDefaults(cfg); err != nil {
		return nil, err
	}
	gvk := mapping.GroupVersionKind
	switch gvk.Group {
	case federation.GroupName:
		mappingVersion := mapping.GroupVersionKind.GroupVersion()
		return f.clients.FederationClientForVersion(&mappingVersion)
	case api.GroupName:
		cfg.APIPath = "/api"
	default:
		cfg.APIPath = "/apis"
	}
	gv := gvk.GroupVersion()
	cfg.GroupVersion = &gv
	if registered.IsThirdPartyAPIGroupVersion(gvk.GroupVersion()) {
		cfg.NegotiatedSerializer = thirdpartyresourcedata.NewNegotiatedSerializer(api.Codecs, gvk.Kind, gv, gv)
	}
	return restclient.RESTClientFor(cfg)
}

func (f *factory) UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	cfg, err := f.clientConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	if err := restclient.SetKubernetesDefaults(cfg); err != nil {
		return nil, err
	}
	cfg.APIPath = "/apis"
	if mapping.GroupVersionKind.Group == api.GroupName {
		cfg.APIPath = "/api"
	}
	gv := mapping.GroupVersionKind.GroupVersion()
	cfg.ContentConfig = dynamic.ContentConfig()
	cfg.GroupVersion = &gv
	return restclient.RESTClientFor(cfg)
}

func (f *factory) Describer(mapping *meta.RESTMapping) (kubectl.Describer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	if mapping.GroupVersionKind.Group == federation.GroupName {
		fedClientSet, err := f.clients.FederationClientSetForVersion(&mappingVersion)
		if err != nil {
			return nil, err
		}
		if mapping.GroupVersionKind.Kind == "Cluster" {
			return &kubectl.ClusterDescriber{Interface: fedClientSet}, nil
		}
	}
	clientset, err := f.clients.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	if describer, ok := kubectl.DescriberFor(mapping.GroupVersionKind.GroupKind(), clientset); ok {
		return describer, nil
	}
	return nil, fmt.Errorf("no description has been implemented for %q", mapping.GroupVersionKind.Kind)
}

func (f *factory) Decoder(toInternal bool) runtime.Decoder {
	var decoder runtime.Decoder
	if toInternal {
		decoder = api.Codecs.UniversalDecoder()
	} else {
		decoder = api.Codecs.UniversalDeserializer()
	}
	return thirdpartyresourcedata.NewDecoder(decoder, "")
}

func (f *factory) JSONEncoder() runtime.Encoder {
	return api.Codecs.LegacyCodec(registered.EnabledVersions()...)
}

func (f *factory) Printer(mapping *meta.RESTMapping, options kubectl.PrintOptions) (kubectl.ResourcePrinter, error) {
	return kubectl.NewHumanReadablePrinter(options), nil
}

func (f *factory) MapBasedSelectorForObject(object runtime.Object) (string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *api.ReplicationController:
		return kubectl.MakeLabels(t.Spec.Selector), nil
	case *api.Pod:
		if len(t.Labels) == 0 {
			return "", fmt.Errorf("the pod has no labels and cannot be exposed")
		}
		return kubectl.MakeLabels(t.Labels), nil
	case *api.Service:
		if t.Spec.Selector == nil {
			return "", fmt.Errorf("the service has no pod selector set")
		}
		return kubectl.MakeLabels(t.Spec.Selector), nil
	case *extensions.Deployment:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *extensions.ReplicaSet:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return "", err
		}
		return "", fmt.Errorf("cannot extract pod selector from %v", gvks[0])
	}
}

func (f *factory) PortsForObject(object runtime.Object) ([]string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *api.ReplicationController:
		return getPorts(t.Spec.Template.Spec), nil
	case *api.Pod:
		return getPorts(t.Spec), nil
	case *api.Service:
		return getServicePorts(t.Spec), nil
	case *extensions.Deployment:
		return getPorts(t.Spec.Template.Spec), nil
	case *extensions.ReplicaSet:
		return getPorts(t.Spec.Template.Spec), nil
	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot extract ports from %v", gvks[0])
	}
}

func (f *factory) ProtocolsForObject(object runtime.Object) (map[string]string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *api.ReplicationController:
		return getProtocols(t.Spec.Template.Spec), nil
	case *api.Pod:
		return getProtocols(t.Spec), nil
	case *api.Service:
		return getServiceProtocols(t.Spec), nil
	case *extensions.Deployment:
		return getProtocols(t.Spec.Template.Spec), nil
	case *extensions.ReplicaSet:
		return getProtocols(t.Spec.Template.Spec), nil
	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot extract protocols from %v", gvks[0])
	}
}

func (f *factory) LabelsForObject(object runtime.Object) (map[string]string, error) {
	return meta.NewAccessor().Labels(object)
}

func (f *factory) LogsForObject(object, options runtime.Object) (*restclient.Request, error) {
	clientset, err := f.clients.ClientSetForVersion(nil)
	if err != nil {
		return nil, err
	}

	switch t := object.(type) {
	case *api.Pod:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		return clientset.Core().Pods(t.Namespace).GetLogs(t.Name, opts), nil

	case *api.ReplicationController:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		selector := labels.SelectorFromSet(t.Spec.Selector)
		sortBy := func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) }
		pod, numPods, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 20*time.Second, sortBy)
		if err != nil {
			return nil, err
		}
		if numPods > 1 {
			fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
		}

		return clientset.Core().Pods(pod.Namespace).GetLogs(pod.Name, opts), nil

	case *extensions.ReplicaSet:
		opts, ok := options.(*api.PodLogOptions)
		if !ok {
			return nil, errors.New("provided options object is not a PodLogOptions")
		}
		selector, err := metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		sortBy := func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) }
		pod, numPods, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 20*time.Second, sortBy)
		if err != nil {
			return nil, err
		}
		if numPods > 1 {
			fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
		}

		return clientset.Core().Pods(pod.Namespace).GetLogs(pod.Name, opts), nil

	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot get the logs from %v", gvks[0])
	}
}

func (f *factory) Pauser(info *resource.Info) (bool, error) {
	switch obj := info.Object.(type) {
	case *extensions.Deployment:
		if obj.Spec.Paused {
			return true, errors.New("is already paused")
		}
		obj.Spec.Paused = true
		return true, nil
	default:
		return false, fmt.Errorf("pausing is not supported")
	}
}

func (f *factory) Resumer(info *resource.Info) (bool, error) {
	switch obj := info.Object.(type) {
	case *extensions.Deployment:
		if !obj.Spec.Paused {
			return true, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return true, nil
	default:
		return false, fmt.Errorf("resuming is not supported")
	}
}

func (f *factory) Scaler(mapping *meta.RESTMapping) (kubectl.Scaler, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clients.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.ScalerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *factory) Reaper(mapping *meta.RESTMapping) (kubectl.Reaper, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, clientsetErr := f.clients.ClientSetForVersion(&mappingVersion)
	reaper, reaperErr := kubectl.ReaperFor(mapping.GroupVersionKind.GroupKind(), clientset)

	if kubectl.IsNoSuchReaperError(reaperErr) {
		return nil, reaperErr
	}
	if clientsetErr != nil {
		return nil, clientsetErr
	}
	return reaper, reaperErr
}

func (f *factory) HistoryViewer(mapping *meta.RESTMapping) (kubectl.HistoryViewer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clients.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.HistoryViewerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *factory) Rollbacker(mapping *meta.RESTMapping) (kubectl.Rollbacker, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clients.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.RollbackerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *factory) StatusViewer(mapping *meta.RESTMapping) (kubectl.StatusViewer, error) {
	mappingVersion := mapping.GroupVersionKind.GroupVersion()
	clientset, err := f.clients.ClientSetForVersion(&mappingVersion)
	if err != nil {
		return nil, err
	}
	return kubectl.StatusViewerFor(mapping.GroupVersionKind.GroupKind(), clientset)
}

func (f *factory) Validator(validate bool, cacheDir string) (validation.Schema, error) {
	if validate {
		discovery, err := f.DiscoveryClient()
		if err != nil {
			return nil, err
		}
		dir := cacheDir
		if len(dir) > 0 {
			version, err := discovery.ServerVersion()
			if err == nil {
				dir = path.Join(cacheDir, version.String())
			} else {
				dir = "" // disable caching as a fallback
			}
		}
		swaggerSchema := &clientSwaggerSchema{
			c:        discovery.RESTClient(),
			cacheDir: dir,
		}
		return validation.ConjunctiveSchema{
			swaggerSchema,
			validation.NoDoubleKeySchema{},
		}, nil
	}
	return validation.NullSchema{}, nil
}

func (f *factory) SwaggerSchema(gvk schema.GroupVersionKind) (*swagger.ApiDeclaration, error) {
	version := gvk.GroupVersion()
	discovery, err := f.DiscoveryClient()
	if err != nil {
		return nil, err
	}
	return discovery.SwaggerSchema(version)
}

func (f *factory) DefaultNamespace() (string, bool, error) {
	return f.clientConfig.Namespace()
}

func (f *factory) Generators(cmdName string) map[string]kubectl.Generator {
	return DefaultGenerators(cmdName)
}

func (f *factory) CanBeExposed(kind schema.GroupKind) error {
	switch kind {
	case api.Kind("ReplicationController"), api.Kind("Service"), api.Kind("Pod"), extensions.Kind("Deployment"), extensions.Kind("ReplicaSet"):
		// nothing to do here
	default:
		return fmt.Errorf("cannot expose a %s", kind)
	}
	return nil
}

func (f *factory) CanBeAutoscaled(kind schema.GroupKind) error {
	switch kind {
	case api.Kind("ReplicationController"), extensions.Kind("Deployment"), extensions.Kind("ReplicaSet"):
		// nothing to do here
	default:
		return fmt.Errorf("cannot autoscale a %v", kind)
	}
	return nil
}

func (f *factory) AttachablePodForObject(object runtime.Object) (*api.Pod, error) {
	clientset, err := f.clients.ClientSetForVersion(nil)
	if err != nil {
		return nil, err
	}
	switch t := object.(type) {
	case *api.ReplicationController:
		selector := labels.SelectorFromSet(t.Spec.Selector)
		sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
		pod, _, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 1*time.Minute, sortBy)
		return pod, err
	case *extensions.Deployment:
		selector, err := metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
		pod, _, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 1*time.Minute, sortBy)
		return pod, err
	case *batch.Job:
		selector, err := metav1.LabelSelectorAsSelector(t.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
		pod, _, err := GetFirstPod(clientset.Core(), t.Namespace, selector, 1*time.Minute, sortBy)
		return pod, err
	case *api.Pod:
		return t, nil
	default:
		gvks, _, err := api.Scheme.ObjectKinds(object)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("cannot attach to %v: not implemented", gvks[0])
	}
}

func (f *factory) UpdatePodSpecForObject(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error) {
	// TODO: replace with a swagger schema based approach (identify pod template via schema introspection)
	switch t := obj.(type) {
	case *api.Pod:
		return true, fn(&t.Spec)
	case *api.ReplicationController:
		if t.Spec.Template == nil {
			t.Spec.Template = &api.PodTemplateSpec{}
		}
		return true, fn(&t.Spec.Template.Spec)
	case *extensions.Deployment:
		return true, fn(&t.Spec.Template.Spec)
	case *extensions.DaemonSet:
		return true, fn(&t.Spec.Template.Spec)
	case *extensions.ReplicaSet:
		return true, fn(&t.Spec.Template.Spec)
	case *apps.StatefulSet:
		return true, fn(&t.Spec.Template.Spec)
	case *batch.Job:
		return true, fn(&t.Spec.Template.Spec)
	default:
		return false, fmt.Errorf("the object is not a pod or does not have a pod template")
	}
}

func (f *factory) EditorEnvs() []string {
	return []string{"KUBE_EDITOR", "EDITOR"}
}

func (f *factory) PrintObjectSpecificMessage(obj runtime.Object, out io.Writer) {
	switch obj := obj.(type) {
	case *api.Service:
		if obj.Spec.Type == api.ServiceTypeNodePort {
			msg := fmt.Sprintf(
				`You have exposed your service on an external port on all nodes in your
cluster.  If you want to expose this service to the external internet, you may
need to set up firewall rules for the service port(s) (%s) to serve traffic.

See http://kubernetes.io/docs/user-guide/services-firewalls for more details.
`,
				makePortsString(obj.Spec.Ports, true))
			out.Write([]byte(msg))
		}

		if _, ok := obj.Annotations[service.AnnotationLoadBalancerSourceRangesKey]; ok {
			msg := fmt.Sprintf(
				`You are using service annotation [service.beta.kubernetes.io/load-balancer-source-ranges].
It has been promoted to field [loadBalancerSourceRanges] in service spec. This annotation will be deprecated in the future.
Please use the loadBalancerSourceRanges field instead.

See http://kubernetes.io/docs/user-guide/services-firewalls for more details.
`)
			out.Write([]byte(msg))
		}
	}
}

// GetFirstPod returns a pod matching the namespace and label selector
// and the number of all pods that match the label selector.
func GetFirstPod(client coreclient.PodsGetter, namespace string, selector labels.Selector, timeout time.Duration, sortBy func([]*v1.Pod) sort.Interface) (*api.Pod, int, error) {
	options := api.ListOptions{LabelSelector: selector}

	podList, err := client.Pods(namespace).List(options)
	if err != nil {
		return nil, 0, err
	}
	pods := []*v1.Pod{}
	for i := range podList.Items {
		pod := podList.Items[i]
		externalPod := &v1.Pod{}
		v1.Convert_api_Pod_To_v1_Pod(&pod, externalPod, nil)
		pods = append(pods, externalPod)
	}
	if len(pods) > 0 {
		sort.Sort(sortBy(pods))
		internalPod := &api.Pod{}
		v1.Convert_v1_Pod_To_api_Pod(pods[0], internalPod, nil)
		return internalPod, len(podList.Items), nil
	}

	// Watch until we observe a pod
	options.ResourceVersion = podList.ResourceVersion
	w, err := client.Pods(namespace).Watch(options)
	if err != nil {
		return nil, 0, err
	}
	defer w.Stop()

	condition := func(event watch.Event) (bool, error) {
		return event.Type == watch.Added || event.Type == watch.Modified, nil
	}
	event, err := watch.Until(timeout, w, condition)
	if err != nil {
		return nil, 0, err
	}
	pod, ok := event.Object.(*api.Pod)
	if !ok {
		return nil, 0, fmt.Errorf("%#v is not a pod event", event)
	}
	return pod, 1, nil
}

// TODO: We need to filter out stuff like secrets.
func (f *factory) Command() string {
	if len(os.Args) == 0 {
		return ""
	}
	base := filepath.Base(os.Args[0])
	args := append([]string{base}, os.Args[1:]...)
	return strings.Join(args, " ")
}

func (f *factory) BindFlags(flags *pflag.FlagSet) {
	// Merge factory's flags
	flags.AddFlagSet(f.flags)

	// Globally persistent flags across all subcommands.
	// TODO Change flag names to consts to allow safer lookup from subcommands.
	// TODO Add a verbose flag that turns on glog logging. Probably need a way
	// to do that automatically for every subcommand.
	flags.BoolVar(&f.clients.matchVersion, FlagMatchBinaryVersion, false, "Require server version to match client version")

	// Normalize all flags that are coming from other packages or pre-configurations
	// a.k.a. change all "_" to "-". e.g. glog package
	flags.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)
}

func (f *factory) BindExternalFlags(flags *pflag.FlagSet) {
	// any flags defined by external projects (not part of pflags)
	flags.AddGoFlagSet(flag.CommandLine)
}

func makePortsString(ports []api.ServicePort, useNodePort bool) string {
	pieces := make([]string, len(ports))
	for ix := range ports {
		var port int32
		if useNodePort {
			port = ports[ix].NodePort
		} else {
			port = ports[ix].Port
		}
		pieces[ix] = fmt.Sprintf("%s:%d", strings.ToLower(string(ports[ix].Protocol)), port)
	}
	return strings.Join(pieces, ",")
}

func getPorts(spec api.PodSpec) []string {
	result := []string{}
	for _, container := range spec.Containers {
		for _, port := range container.Ports {
			result = append(result, strconv.Itoa(int(port.ContainerPort)))
		}
	}
	return result
}

func getProtocols(spec api.PodSpec) map[string]string {
	result := make(map[string]string)
	for _, container := range spec.Containers {
		for _, port := range container.Ports {
			result[strconv.Itoa(int(port.ContainerPort))] = string(port.Protocol)
		}
	}
	return result
}

// Extracts the ports exposed by a service from the given service spec.
func getServicePorts(spec api.ServiceSpec) []string {
	result := []string{}
	for _, servicePort := range spec.Ports {
		result = append(result, strconv.Itoa(int(servicePort.Port)))
	}
	return result
}

// Extracts the protocols exposed by a service from the given service spec.
func getServiceProtocols(spec api.ServiceSpec) map[string]string {
	result := make(map[string]string)
	for _, servicePort := range spec.Ports {
		result[strconv.Itoa(int(servicePort.Port))] = string(servicePort.Protocol)
	}
	return result
}

type clientSwaggerSchema struct {
	c        restclient.Interface
	cacheDir string
}

const schemaFileName = "schema.json"

type schemaClient interface {
	Get() *restclient.Request
}

func recursiveSplit(dir string) []string {
	parent, file := path.Split(dir)
	if len(parent) == 0 {
		return []string{file}
	}
	return append(recursiveSplit(parent[:len(parent)-1]), file)
}

func substituteUserHome(dir string) (string, error) {
	if len(dir) == 0 || dir[0] != '~' {
		return dir, nil
	}
	parts := recursiveSplit(dir)
	if len(parts[0]) == 1 {
		parts[0] = os.Getenv("HOME")
	} else {
		usr, err := user.Lookup(parts[0][1:])
		if err != nil {
			return "", err
		}
		parts[0] = usr.HomeDir
	}
	return path.Join(parts...), nil
}

func writeSchemaFile(schemaData []byte, cacheDir, cacheFile, prefix, groupVersion string) error {
	if err := os.MkdirAll(path.Join(cacheDir, prefix, groupVersion), 0755); err != nil {
		return err
	}
	tmpFile, err := ioutil.TempFile(cacheDir, "schema")
	if err != nil {
		// If we can't write, keep going.
		if os.IsPermission(err) {
			return nil
		}
		return err
	}
	if _, err := io.Copy(tmpFile, bytes.NewBuffer(schemaData)); err != nil {
		return err
	}
	if err := os.Link(tmpFile.Name(), cacheFile); err != nil {
		// If we can't write due to file existing, or permission problems, keep going.
		if os.IsExist(err) || os.IsPermission(err) {
			return nil
		}
		return err
	}
	return nil
}

func getSchemaAndValidate(c schemaClient, data []byte, prefix, groupVersion, cacheDir string, delegate validation.Schema) (err error) {
	var schemaData []byte
	var firstSeen bool
	fullDir, err := substituteUserHome(cacheDir)
	if err != nil {
		return err
	}
	cacheFile := path.Join(fullDir, prefix, groupVersion, schemaFileName)

	if len(cacheDir) != 0 {
		if schemaData, err = ioutil.ReadFile(cacheFile); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if schemaData == nil {
		firstSeen = true
		schemaData, err = downloadSchemaAndStore(c, cacheDir, fullDir, cacheFile, prefix, groupVersion)
		if err != nil {
			return err
		}
	}
	schema, err := validation.NewSwaggerSchemaFromBytes(schemaData, delegate)
	if err != nil {
		return err
	}
	err = schema.ValidateBytes(data)
	if _, ok := err.(validation.TypeNotFoundError); ok && !firstSeen {
		// As a temporary hack, kubectl would re-get the schema if validation
		// fails for type not found reason.
		// TODO: runtime-config settings needs to make into the file's name
		schemaData, err = downloadSchemaAndStore(c, cacheDir, fullDir, cacheFile, prefix, groupVersion)
		if err != nil {
			return err
		}
		schema, err := validation.NewSwaggerSchemaFromBytes(schemaData, delegate)
		if err != nil {
			return err
		}
		return schema.ValidateBytes(data)
	}

	return err
}

// Download swagger schema from apiserver and store it to file.
func downloadSchemaAndStore(c schemaClient, cacheDir, fullDir, cacheFile, prefix, groupVersion string) (schemaData []byte, err error) {
	schemaData, err = c.Get().
		AbsPath("/swaggerapi", prefix, groupVersion).
		Do().
		Raw()
	if err != nil {
		return
	}
	if len(cacheDir) != 0 {
		if err = writeSchemaFile(schemaData, fullDir, cacheFile, prefix, groupVersion); err != nil {
			return
		}
	}
	return
}

func (c *clientSwaggerSchema) ValidateBytes(data []byte) error {
	gvk, err := json.DefaultMetaFactory.Interpret(data)
	if err != nil {
		return err
	}
	if ok := registered.IsEnabledVersion(gvk.GroupVersion()); !ok {
		// if we don't have this in our scheme, just skip validation because its an object we don't recognize
		return nil
	}

	switch gvk.Group {
	case api.GroupName:
		return getSchemaAndValidate(c.c, data, "api", gvk.GroupVersion().String(), c.cacheDir, c)

	default:
		return getSchemaAndValidate(c.c, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	}
}

// DefaultClientConfig creates a clientcmd.ClientConfig with the following hierarchy:
//   1.  Use the kubeconfig builder.  The number of merges and overrides here gets a little crazy.  Stay with me.
//       1.  Merge the kubeconfig itself.  This is done with the following hierarchy rules:
//           1.  CommandLineLocation - this parsed from the command line, so it must be late bound.  If you specify this,
//               then no other kubeconfig files are merged.  This file must exist.
//           2.  If $KUBECONFIG is set, then it is treated as a list of files that should be merged.
//	     3.  HomeDirectoryLocation
//           Empty filenames are ignored.  Files with non-deserializable content produced errors.
//           The first file to set a particular value or map key wins and the value or map key is never changed.
//           This means that the first file to set CurrentContext will have its context preserved.  It also means
//           that if two files specify a "red-user", only values from the first file's red-user are used.  Even
//           non-conflicting entries from the second file's "red-user" are discarded.
//       2.  Determine the context to use based on the first hit in this chain
//           1.  command line argument - again, parsed from the command line, so it must be late bound
//           2.  CurrentContext from the merged kubeconfig file
//           3.  Empty is allowed at this stage
//       3.  Determine the cluster info and auth info to use.  At this point, we may or may not have a context.  They
//           are built based on the first hit in this chain.  (run it twice, once for auth, once for cluster)
//           1.  command line argument
//           2.  If context is present, then use the context value
//           3.  Empty is allowed
//       4.  Determine the actual cluster info to use.  At this point, we may or may not have a cluster info.  Build
//           each piece of the cluster info based on the chain:
//           1.  command line argument
//           2.  If cluster info is present and a value for the attribute is present, use it.
//           3.  If you don't have a server location, bail.
//       5.  Auth info is build using the same rules as cluster info, EXCEPT that you can only have one authentication
//           technique per auth info.  The following conditions result in an error:
//           1.  If there are two conflicting techniques specified from the command line, fail.
//           2.  If the command line does not specify one, and the auth info has conflicting techniques, fail.
//           3.  If the command line specifies one and the auth info specifies another, honor the command line technique.
//   2.  Use default values and potentially prompt for auth information
//
//   However, if it appears that we're running in a kubernetes cluster
//   container environment, then run with the auth info kubernetes mounted for
//   us. Specifically:
//     The env vars KUBERNETES_SERVICE_HOST and KUBERNETES_SERVICE_PORT are
//     set, and the file /var/run/secrets/kubernetes.io/serviceaccount/token
//     exists and is not a directory.
func DefaultClientConfig(flags *pflag.FlagSet) clientcmd.ClientConfig {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	// use the standard defaults for this client command
	// DEPRECATED: remove and replace with something more accurate
	loadingRules.DefaultClientConfig = &clientcmd.DefaultClientConfig

	flags.StringVar(&loadingRules.ExplicitPath, "kubeconfig", "", "Path to the kubeconfig file to use for CLI requests.")

	overrides := &clientcmd.ConfigOverrides{ClusterDefaults: clientcmd.ClusterDefaults}

	flagNames := clientcmd.RecommendedConfigOverrideFlags("")
	// short flagnames are disabled by default.  These are here for compatibility with existing scripts
	flagNames.ClusterOverrideFlags.APIServer.ShortName = "s"

	clientcmd.BindOverrideFlags(overrides, flags, flagNames)
	clientConfig := clientcmd.NewInteractiveDeferredLoadingClientConfig(loadingRules, overrides, os.Stdin)

	return clientConfig
}

func (f *factory) DefaultResourceFilterOptions(cmd *cobra.Command, withNamespace bool) *kubectl.PrintOptions {
	columnLabel, err := cmd.Flags().GetStringSlice("label-columns")
	if err != nil {
		columnLabel = []string{}
	}
	opts := &kubectl.PrintOptions{
		NoHeaders:          GetFlagBool(cmd, "no-headers"),
		WithNamespace:      withNamespace,
		Wide:               GetWideFlag(cmd),
		ShowAll:            GetFlagBool(cmd, "show-all"),
		ShowLabels:         GetFlagBool(cmd, "show-labels"),
		AbsoluteTimestamps: isWatch(cmd),
		ColumnLabels:       columnLabel,
	}

	return opts
}

func (f *factory) DefaultResourceFilterFunc() kubectl.Filters {
	return kubectl.NewResourceFilter()
}

func (f *factory) PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
	gvks, _, err := api.Scheme.ObjectKinds(obj)
	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(gvks[0].GroupKind())
	if err != nil {
		return err
	}

	printer, err := f.PrinterForMapping(cmd, mapping, false)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

func (f *factory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error) {
	printer, ok, err := PrinterForCommand(cmd)
	if err != nil {
		return nil, err
	}
	if ok {
		clientConfig, err := f.ClientConfig()
		if err != nil {
			return nil, err
		}

		version, err := OutputVersion(cmd, clientConfig.GroupVersion)
		if err != nil {
			return nil, err
		}
		if version.Empty() && mapping != nil {
			version = mapping.GroupVersionKind.GroupVersion()
		}
		if version.Empty() {
			return nil, fmt.Errorf("you must specify an output-version when using this output format")
		}

		if mapping != nil {
			printer = kubectl.NewVersionedPrinter(printer, mapping.ObjectConvertor, version, mapping.GroupVersionKind.GroupVersion())
		}

	} else {
		// Some callers do not have "label-columns" so we can't use the GetFlagStringSlice() helper
		columnLabel, err := cmd.Flags().GetStringSlice("label-columns")
		if err != nil {
			columnLabel = []string{}
		}
		printer, err = f.Printer(mapping, kubectl.PrintOptions{
			NoHeaders:          GetFlagBool(cmd, "no-headers"),
			WithNamespace:      withNamespace,
			Wide:               GetWideFlag(cmd),
			ShowAll:            GetFlagBool(cmd, "show-all"),
			ShowLabels:         GetFlagBool(cmd, "show-labels"),
			AbsoluteTimestamps: isWatch(cmd),
			ColumnLabels:       columnLabel,
		})
		if err != nil {
			return nil, err
		}
		printer = maybeWrapSortingPrinter(cmd, printer)
	}

	return printer, nil
}

func (f *factory) NewBuilder() *resource.Builder {
	mapper, typer := f.Object()

	return resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true))
}

func (f *factory) SuggestedPodTemplateResources() []schema.GroupResource {
	return []schema.GroupResource{
		{Resource: "replicationcontroller"},
		{Resource: "deployment"},
		{Resource: "daemonset"},
		{Resource: "job"},
		{Resource: "replicaset"},
	}
}

// overlyCautiousIllegalFileCharacters matches characters that *might* not be supported.  Windows is really restrictive, so this is really restrictive
var overlyCautiousIllegalFileCharacters = regexp.MustCompile(`[^(\w/\.)]`)

// computeDiscoverCacheDir takes the parentDir and the host and comes up with a "usually non-colliding" name.
func computeDiscoverCacheDir(parentDir, host string) string {
	// strip the optional scheme from host if its there:
	schemelessHost := strings.Replace(strings.Replace(host, "https://", "", 1), "http://", "", 1)
	// now do a simple collapse of non-AZ09 characters.  Collisions are possible but unlikely.  Even if we do collide the problem is short lived
	safeHost := overlyCautiousIllegalFileCharacters.ReplaceAllString(schemelessHost, "_")

	return filepath.Join(parentDir, safeHost)
}
