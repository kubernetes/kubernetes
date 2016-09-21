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
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/unversioned/adapters/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
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
type Factory struct {
	clients *ClientCache
	flags   *pflag.FlagSet

	// Returns interfaces for dealing with arbitrary runtime.Objects. If thirdPartyDiscovery is true, performs API calls
	// to discovery dynamic API objects registered by third parties.
	Object func(thirdPartyDiscovery bool) (meta.RESTMapper, runtime.ObjectTyper)
	// Returns interfaces for dealing with arbitrary
	// runtime.Unstructured. This performs API calls to discover types.
	UnstructuredObject func() (meta.RESTMapper, runtime.ObjectTyper, error)
	// Returns interfaces for decoding objects - if toInternal is set, decoded objects will be converted
	// into their internal form (if possible). Eventually the internal form will be removed as an option,
	// and only versioned objects will be returned.
	Decoder func(toInternal bool) runtime.Decoder
	// Returns an encoder capable of encoding a provided object into JSON in the default desired version.
	JSONEncoder func() runtime.Encoder
	// Returns a client for accessing Kubernetes resources or an error.
	Client func() (*client.Client, error)
	// Returns a client.Config for accessing the Kubernetes server.
	ClientConfig func() (*restclient.Config, error)
	// Returns a RESTClient for working with the specified RESTMapping or an error. This is intended
	// for working with arbitrary resources and is not guaranteed to point to a Kubernetes APIServer.
	ClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	// Returns a RESTClient for working with Unstructured objects.
	UnstructuredClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	// Returns a Describer for displaying the specified RESTMapping type or an error.
	Describer func(mapping *meta.RESTMapping) (kubectl.Describer, error)
	// Returns a Printer for formatting objects of the given type or an error.
	Printer func(mapping *meta.RESTMapping, options kubectl.PrintOptions) (kubectl.ResourcePrinter, error)
	// Returns a Scaler for changing the size of the specified RESTMapping type or an error
	Scaler func(mapping *meta.RESTMapping) (kubectl.Scaler, error)
	// Returns a Reaper for gracefully shutting down resources.
	Reaper func(mapping *meta.RESTMapping) (kubectl.Reaper, error)
	// Returns a HistoryViewer for viewing change history
	HistoryViewer func(mapping *meta.RESTMapping) (kubectl.HistoryViewer, error)
	// Returns a Rollbacker for changing the rollback version of the specified RESTMapping type or an error
	Rollbacker func(mapping *meta.RESTMapping) (kubectl.Rollbacker, error)
	// Returns a StatusViewer for printing rollout status.
	StatusViewer func(mapping *meta.RESTMapping) (kubectl.StatusViewer, error)
	// MapBasedSelectorForObject returns the map-based selector associated with the provided object. If a
	// new set-based selector is provided, an error is returned if the selector cannot be converted to a
	// map-based selector
	MapBasedSelectorForObject func(object runtime.Object) (string, error)
	// PortsForObject returns the ports associated with the provided object
	PortsForObject func(object runtime.Object) ([]string, error)
	// ProtocolsForObject returns the <port, protocol> mapping associated with the provided object
	ProtocolsForObject func(object runtime.Object) (map[string]string, error)
	// LabelsForObject returns the labels associated with the provided object
	LabelsForObject func(object runtime.Object) (map[string]string, error)
	// LogsForObject returns a request for the logs associated with the provided object
	LogsForObject func(object, options runtime.Object) (*restclient.Request, error)
	// PauseObject marks the provided object as paused ie. it will not be reconciled by its controller.
	PauseObject func(object runtime.Object) (bool, error)
	// ResumeObject resumes a paused object ie. it will be reconciled by its controller.
	ResumeObject func(object runtime.Object) (bool, error)
	// Returns a schema that can validate objects stored on disk.
	Validator func(validate bool, cacheDir string) (validation.Schema, error)
	// SwaggerSchema returns the schema declaration for the provided group version kind.
	SwaggerSchema func(unversioned.GroupVersionKind) (*swagger.ApiDeclaration, error)
	// Returns the default namespace to use in cases where no
	// other namespace is specified and whether the namespace was
	// overridden.
	DefaultNamespace func() (string, bool, error)
	// Generators returns the generators for the provided command
	Generators func(cmdName string) map[string]kubectl.Generator
	// Check whether the kind of resources could be exposed
	CanBeExposed func(kind unversioned.GroupKind) error
	// Check whether the kind of resources could be autoscaled
	CanBeAutoscaled func(kind unversioned.GroupKind) error
	// AttachablePodForObject returns the pod to which to attach given an object.
	AttachablePodForObject func(object runtime.Object) (*api.Pod, error)
	// UpdatePodSpecForObject will call the provided function on the pod spec this object supports,
	// return false if no pod spec is supported, or return an error.
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
	// EditorEnvs returns a group of environment variables that the edit command
	// can range over in order to determine if the user has specified an editor
	// of their choice.
	EditorEnvs func() []string
	// PrintObjectSpecificMessage prints object-specific messages on the provided writer
	PrintObjectSpecificMessage func(obj runtime.Object, out io.Writer)
}

const (
	RunV1GeneratorName                          = "run/v1"
	RunPodV1GeneratorName                       = "run-pod/v1"
	ServiceV1GeneratorName                      = "service/v1"
	ServiceV2GeneratorName                      = "service/v2"
	ServiceNodePortGeneratorV1Name              = "service-nodeport/v1"
	ServiceClusterIPGeneratorV1Name             = "service-clusterip/v1"
	ServiceLoadBalancerGeneratorV1Name          = "service-loadbalancer/v1"
	ServiceAccountV1GeneratorName               = "serviceaccount/v1"
	HorizontalPodAutoscalerV1Beta1GeneratorName = "horizontalpodautoscaler/v1beta1"
	HorizontalPodAutoscalerV1GeneratorName      = "horizontalpodautoscaler/v1"
	DeploymentV1Beta1GeneratorName              = "deployment/v1beta1"
	DeploymentBasicV1Beta1GeneratorName         = "deployment-basic/v1beta1"
	JobV1Beta1GeneratorName                     = "job/v1beta1"
	JobV1GeneratorName                          = "job/v1"
	ScheduledJobV2Alpha1GeneratorName           = "scheduledjob/v2alpha1"
	NamespaceV1GeneratorName                    = "namespace/v1"
	ResourceQuotaV1GeneratorName                = "resourcequotas/v1"
	SecretV1GeneratorName                       = "secret/v1"
	SecretForDockerRegistryV1GeneratorName      = "secret-for-docker-registry/v1"
	SecretForTLSV1GeneratorName                 = "secret-for-tls/v1"
	ConfigMapV1GeneratorName                    = "configmap/v1"
)

// DefaultGenerators returns the set of default generators for use in Factory instances
func DefaultGenerators(cmdName string) map[string]kubectl.Generator {
	generators := map[string]map[string]kubectl.Generator{}
	generators["expose"] = map[string]kubectl.Generator{
		ServiceV1GeneratorName: kubectl.ServiceGeneratorV1{},
		ServiceV2GeneratorName: kubectl.ServiceGeneratorV2{},
	}
	generators["service-clusterip"] = map[string]kubectl.Generator{
		ServiceClusterIPGeneratorV1Name: kubectl.ServiceClusterIPGeneratorV1{},
	}
	generators["service-nodeport"] = map[string]kubectl.Generator{
		ServiceNodePortGeneratorV1Name: kubectl.ServiceNodePortGeneratorV1{},
	}
	generators["service-loadbalancer"] = map[string]kubectl.Generator{
		ServiceLoadBalancerGeneratorV1Name: kubectl.ServiceLoadBalancerGeneratorV1{},
	}
	generators["deployment"] = map[string]kubectl.Generator{
		DeploymentBasicV1Beta1GeneratorName: kubectl.DeploymentBasicGeneratorV1{},
	}
	generators["run"] = map[string]kubectl.Generator{
		RunV1GeneratorName:                kubectl.BasicReplicationController{},
		RunPodV1GeneratorName:             kubectl.BasicPod{},
		DeploymentV1Beta1GeneratorName:    kubectl.DeploymentV1Beta1{},
		JobV1Beta1GeneratorName:           kubectl.JobV1Beta1{},
		JobV1GeneratorName:                kubectl.JobV1{},
		ScheduledJobV2Alpha1GeneratorName: kubectl.ScheduledJobV2Alpha1{},
	}
	generators["autoscale"] = map[string]kubectl.Generator{
		HorizontalPodAutoscalerV1Beta1GeneratorName: kubectl.HorizontalPodAutoscalerV1Beta1{},
		HorizontalPodAutoscalerV1GeneratorName:      kubectl.HorizontalPodAutoscalerV1{},
	}
	generators["namespace"] = map[string]kubectl.Generator{
		NamespaceV1GeneratorName: kubectl.NamespaceGeneratorV1{},
	}

	generators["quota"] = map[string]kubectl.Generator{
		ResourceQuotaV1GeneratorName: kubectl.ResourceQuotaGeneratorV1{},
	}

	generators["secret"] = map[string]kubectl.Generator{
		SecretV1GeneratorName: kubectl.SecretGeneratorV1{},
	}
	generators["secret-for-docker-registry"] = map[string]kubectl.Generator{
		SecretForDockerRegistryV1GeneratorName: kubectl.SecretForDockerRegistryGeneratorV1{},
	}
	generators["secret-for-tls"] = map[string]kubectl.Generator{
		SecretForTLSV1GeneratorName: kubectl.SecretForTLSGeneratorV1{},
	}

	return generators[cmdName]
}

func getGroupVersionKinds(gvks []unversioned.GroupVersionKind, group string) []unversioned.GroupVersionKind {
	result := []unversioned.GroupVersionKind{}
	for ix := range gvks {
		if gvks[ix].Group == group {
			result = append(result, gvks[ix])
		}
	}
	return result
}

func makeInterfacesFor(versionList []unversioned.GroupVersion) func(version unversioned.GroupVersion) (*meta.VersionInterfaces, error) {
	accessor := meta.NewAccessor()
	return func(version unversioned.GroupVersion) (*meta.VersionInterfaces, error) {
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

// NewFactory creates a factory with the default Kubernetes resources defined
// if optionalClientConfig is nil, then flags will be bound to a new clientcmd.ClientConfig.
// if optionalClientConfig is not nil, then this factory will make use of it.
func NewFactory(optionalClientConfig clientcmd.ClientConfig) *Factory {
	mapper := kubectl.ShortcutExpander{RESTMapper: registered.RESTMapper()}

	flags := pflag.NewFlagSet("", pflag.ContinueOnError)
	flags.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc) // Warn for "_" flags

	clientConfig := optionalClientConfig
	if optionalClientConfig == nil {
		clientConfig = DefaultClientConfig(flags)
	}

	clients := NewClientCache(clientConfig)

	return &Factory{
		clients: clients,
		flags:   flags,

		// If discoverDynamicAPIs is true, make API calls to the discovery service to find APIs that
		// have been dynamically added to the apiserver
		Object: func(discoverDynamicAPIs bool) (meta.RESTMapper, runtime.ObjectTyper) {
			cfg, err := clientConfig.ClientConfig()
			checkErrWithPrefix("failed to get client config: ", err)
			cmdApiVersion := unversioned.GroupVersion{}
			if cfg.GroupVersion != nil {
				cmdApiVersion = *cfg.GroupVersion
			}
			if discoverDynamicAPIs {
				client, err := clients.ClientForVersion(&unversioned.GroupVersion{Version: "v1"})
				checkErrWithPrefix("failed to find client for version v1: ", err)

				var versions []unversioned.GroupVersion
				var gvks []unversioned.GroupVersionKind
				retries := 3
				for i := 0; i < retries; i++ {
					versions, gvks, err = GetThirdPartyGroupVersions(client.Discovery())
					// Retry if we got a NotFound error, because user may delete
					// a thirdparty group when the GetThirdPartyGroupVersions is
					// running.
					if err == nil || !apierrors.IsNotFound(err) {
						break
					}
				}
				checkErrWithPrefix("failed to get third-party group versions: ", err)
				if len(versions) > 0 {
					priorityMapper, ok := mapper.RESTMapper.(meta.PriorityRESTMapper)
					if !ok {
						CheckErr(fmt.Errorf("expected PriorityMapper, saw: %v", mapper.RESTMapper))
						return nil, nil
					}
					multiMapper, ok := priorityMapper.Delegate.(meta.MultiRESTMapper)
					if !ok {
						CheckErr(fmt.Errorf("unexpected type: %v", mapper.RESTMapper))
						return nil, nil
					}
					groupsMap := map[string][]unversioned.GroupVersion{}
					for _, version := range versions {
						groupsMap[version.Group] = append(groupsMap[version.Group], version)
					}
					for group, versionList := range groupsMap {
						preferredExternalVersion := versionList[0]

						thirdPartyMapper, err := kubectl.NewThirdPartyResourceMapper(versionList, getGroupVersionKinds(gvks, group))
						checkErrWithPrefix("failed to create third party resource mapper: ", err)
						accessor := meta.NewAccessor()
						groupMeta := apimachinery.GroupMeta{
							GroupVersion:  preferredExternalVersion,
							GroupVersions: versionList,
							RESTMapper:    thirdPartyMapper,
							SelfLinker:    runtime.SelfLinker(accessor),
							InterfacesFor: makeInterfacesFor(versionList),
						}

						checkErrWithPrefix("failed to register group: ", registered.RegisterGroup(groupMeta))
						registered.AddThirdPartyAPIGroupVersions(versionList...)
						multiMapper = append(meta.MultiRESTMapper{thirdPartyMapper}, multiMapper...)
					}
					priorityMapper.Delegate = multiMapper
					// Reassign to the RESTMapper here because priorityMapper is actually a copy, so if we
					// don't reassign, the above assignement won't actually update mapper.RESTMapper
					mapper.RESTMapper = priorityMapper
				}
			}
			outputRESTMapper := kubectl.OutputVersionMapper{RESTMapper: mapper, OutputVersions: []unversioned.GroupVersion{cmdApiVersion}}
			priorityRESTMapper := meta.PriorityRESTMapper{
				Delegate: outputRESTMapper,
			}
			// TODO: this should come from registered versions
			groups := []string{api.GroupName, autoscaling.GroupName, extensions.GroupName, federation.GroupName, batch.GroupName}
			// set a preferred version
			for _, group := range groups {
				gvs := registered.EnabledVersionsForGroup(group)
				if len(gvs) == 0 {
					continue
				}
				priorityRESTMapper.ResourcePriority = append(priorityRESTMapper.ResourcePriority, unversioned.GroupVersionResource{Group: group, Version: gvs[0].Version, Resource: meta.AnyResource})
				priorityRESTMapper.KindPriority = append(priorityRESTMapper.KindPriority, unversioned.GroupVersionKind{Group: group, Version: gvs[0].Version, Kind: meta.AnyKind})
			}
			for _, group := range groups {
				priorityRESTMapper.ResourcePriority = append(priorityRESTMapper.ResourcePriority, unversioned.GroupVersionResource{Group: group, Version: meta.AnyVersion, Resource: meta.AnyResource})
				priorityRESTMapper.KindPriority = append(priorityRESTMapper.KindPriority, unversioned.GroupVersionKind{Group: group, Version: meta.AnyVersion, Kind: meta.AnyKind})
			}
			return priorityRESTMapper, api.Scheme
		},
		UnstructuredObject: func() (meta.RESTMapper, runtime.ObjectTyper, error) {
			cfg, err := clients.ClientConfigForVersion(nil)
			if err != nil {
				return nil, nil, err
			}

			dc, err := discovery.NewDiscoveryClientForConfig(cfg)
			if err != nil {
				return nil, nil, err
			}

			groupResources, err := discovery.GetAPIGroupResources(dc)
			if err != nil {
				return nil, nil, err
			}

			// Register unknown APIs as third party for now to make
			// validation happy. TODO perhaps make a dynamic schema
			// validator to avoid this.
			for _, group := range groupResources {
				for _, version := range group.Group.Versions {
					gv := unversioned.GroupVersion{Group: group.Group.Name, Version: version.Version}
					if !registered.IsRegisteredVersion(gv) {
						registered.AddThirdPartyAPIGroupVersions(gv)
					}
				}
			}

			mapper := discovery.NewRESTMapper(groupResources, meta.InterfacesForUnstructured)

			typer := discovery.NewUnstructuredObjectTyper(groupResources)

			return kubectl.ShortcutExpander{RESTMapper: mapper}, typer, nil
		},
		Client: func() (*client.Client, error) {
			return clients.ClientForVersion(nil)
		},
		ClientConfig: func() (*restclient.Config, error) {
			return clients.ClientConfigForVersion(nil)
		},
		ClientForMapping: func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
			cfg, err := clientConfig.ClientConfig()
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
				return clients.FederationClientForVersion(&mappingVersion)
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
		},
		UnstructuredClientForMapping: func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
			cfg, err := clientConfig.ClientConfig()
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
		},
		Describer: func(mapping *meta.RESTMapping) (kubectl.Describer, error) {
			mappingVersion := mapping.GroupVersionKind.GroupVersion()
			if mapping.GroupVersionKind.Group == federation.GroupName {
				fedClientSet, err := clients.FederationClientSetForVersion(&mappingVersion)
				if err != nil {
					return nil, err
				}
				if mapping.GroupVersionKind.Kind == "Cluster" {
					return &kubectl.ClusterDescriber{Interface: fedClientSet}, nil
				}
			}
			client, err := clients.ClientForVersion(&mappingVersion)
			if err != nil {
				return nil, err
			}
			if describer, ok := kubectl.DescriberFor(mapping.GroupVersionKind.GroupKind(), client); ok {
				return describer, nil
			}
			return nil, fmt.Errorf("no description has been implemented for %q", mapping.GroupVersionKind.Kind)
		},
		Decoder: func(toInternal bool) runtime.Decoder {
			var decoder runtime.Decoder
			if toInternal {
				decoder = api.Codecs.UniversalDecoder()
			} else {
				decoder = api.Codecs.UniversalDeserializer()
			}
			return thirdpartyresourcedata.NewDecoder(decoder, "")

		},
		JSONEncoder: func() runtime.Encoder {
			return api.Codecs.LegacyCodec(registered.EnabledVersions()...)
		},
		Printer: func(mapping *meta.RESTMapping, options kubectl.PrintOptions) (kubectl.ResourcePrinter, error) {
			return kubectl.NewHumanReadablePrinter(options), nil
		},
		MapBasedSelectorForObject: func(object runtime.Object) (string, error) {
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
		},
		PortsForObject: func(object runtime.Object) ([]string, error) {
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
		},
		ProtocolsForObject: func(object runtime.Object) (map[string]string, error) {
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
		},
		LabelsForObject: func(object runtime.Object) (map[string]string, error) {
			return meta.NewAccessor().Labels(object)
		},
		LogsForObject: func(object, options runtime.Object) (*restclient.Request, error) {
			c, err := clients.ClientForVersion(nil)
			if err != nil {
				return nil, err
			}

			switch t := object.(type) {
			case *api.Pod:
				opts, ok := options.(*api.PodLogOptions)
				if !ok {
					return nil, errors.New("provided options object is not a PodLogOptions")
				}
				return c.Pods(t.Namespace).GetLogs(t.Name, opts), nil

			case *api.ReplicationController:
				opts, ok := options.(*api.PodLogOptions)
				if !ok {
					return nil, errors.New("provided options object is not a PodLogOptions")
				}
				selector := labels.SelectorFromSet(t.Spec.Selector)
				sortBy := func(pods []*api.Pod) sort.Interface { return controller.ByLogging(pods) }
				pod, numPods, err := GetFirstPod(c, t.Namespace, selector, 20*time.Second, sortBy)
				if err != nil {
					return nil, err
				}
				if numPods > 1 {
					fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
				}

				return c.Pods(pod.Namespace).GetLogs(pod.Name, opts), nil

			case *extensions.ReplicaSet:
				opts, ok := options.(*api.PodLogOptions)
				if !ok {
					return nil, errors.New("provided options object is not a PodLogOptions")
				}
				selector, err := unversioned.LabelSelectorAsSelector(t.Spec.Selector)
				if err != nil {
					return nil, fmt.Errorf("invalid label selector: %v", err)
				}
				sortBy := func(pods []*api.Pod) sort.Interface { return controller.ByLogging(pods) }
				pod, numPods, err := GetFirstPod(c, t.Namespace, selector, 20*time.Second, sortBy)
				if err != nil {
					return nil, err
				}
				if numPods > 1 {
					fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
				}

				return c.Pods(pod.Namespace).GetLogs(pod.Name, opts), nil

			default:
				gvks, _, err := api.Scheme.ObjectKinds(object)
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("cannot get the logs from %v", gvks[0])
			}
		},
		PauseObject: func(object runtime.Object) (bool, error) {
			c, err := clients.ClientForVersion(nil)
			if err != nil {
				return false, err
			}

			switch t := object.(type) {
			case *extensions.Deployment:
				if t.Spec.Paused {
					return true, nil
				}
				t.Spec.Paused = true
				_, err := c.Extensions().Deployments(t.Namespace).Update(t)
				return false, err
			default:
				gvks, _, err := api.Scheme.ObjectKinds(object)
				if err != nil {
					return false, err
				}
				return false, fmt.Errorf("cannot pause %v", gvks[0])
			}
		},
		ResumeObject: func(object runtime.Object) (bool, error) {
			c, err := clients.ClientForVersion(nil)
			if err != nil {
				return false, err
			}

			switch t := object.(type) {
			case *extensions.Deployment:
				if !t.Spec.Paused {
					return true, nil
				}
				t.Spec.Paused = false
				_, err := c.Extensions().Deployments(t.Namespace).Update(t)
				return false, err
			default:
				gvks, _, err := api.Scheme.ObjectKinds(object)
				if err != nil {
					return false, err
				}
				return false, fmt.Errorf("cannot resume %v", gvks[0])
			}
		},
		Scaler: func(mapping *meta.RESTMapping) (kubectl.Scaler, error) {
			mappingVersion := mapping.GroupVersionKind.GroupVersion()
			client, err := clients.ClientForVersion(&mappingVersion)
			if err != nil {
				return nil, err
			}
			return kubectl.ScalerFor(mapping.GroupVersionKind.GroupKind(), client)
		},
		Reaper: func(mapping *meta.RESTMapping) (kubectl.Reaper, error) {
			mappingVersion := mapping.GroupVersionKind.GroupVersion()
			client, err := clients.ClientForVersion(&mappingVersion)
			if err != nil {
				return nil, err
			}
			return kubectl.ReaperFor(mapping.GroupVersionKind.GroupKind(), client)
		},
		HistoryViewer: func(mapping *meta.RESTMapping) (kubectl.HistoryViewer, error) {
			mappingVersion := mapping.GroupVersionKind.GroupVersion()
			client, err := clients.ClientForVersion(&mappingVersion)
			clientset := clientset.FromUnversionedClient(client)
			if err != nil {
				return nil, err
			}
			return kubectl.HistoryViewerFor(mapping.GroupVersionKind.GroupKind(), clientset)
		},
		Rollbacker: func(mapping *meta.RESTMapping) (kubectl.Rollbacker, error) {
			mappingVersion := mapping.GroupVersionKind.GroupVersion()
			client, err := clients.ClientForVersion(&mappingVersion)
			if err != nil {
				return nil, err
			}
			return kubectl.RollbackerFor(mapping.GroupVersionKind.GroupKind(), client)
		},
		StatusViewer: func(mapping *meta.RESTMapping) (kubectl.StatusViewer, error) {
			mappingVersion := mapping.GroupVersionKind.GroupVersion()
			client, err := clients.ClientForVersion(&mappingVersion)
			if err != nil {
				return nil, err
			}
			return kubectl.StatusViewerFor(mapping.GroupVersionKind.GroupKind(), client)
		},
		Validator: func(validate bool, cacheDir string) (validation.Schema, error) {
			if validate {
				client, err := clients.ClientForVersion(nil)
				if err != nil {
					return nil, err
				}
				dir := cacheDir
				if len(dir) > 0 {
					version, err := client.ServerVersion()
					if err != nil {
						return nil, err
					}
					dir = path.Join(cacheDir, version.String())
				}
				fedClient, err := clients.FederationClientForVersion(nil)
				if err != nil {
					return nil, err
				}
				return &clientSwaggerSchema{
					c:        client,
					fedc:     fedClient,
					cacheDir: dir,
					mapper:   api.RESTMapper,
				}, nil
			}
			return validation.NullSchema{}, nil
		},
		SwaggerSchema: func(gvk unversioned.GroupVersionKind) (*swagger.ApiDeclaration, error) {
			version := gvk.GroupVersion()
			client, err := clients.ClientForVersion(&version)
			if err != nil {
				return nil, err
			}
			return client.Discovery().SwaggerSchema(version)
		},
		DefaultNamespace: func() (string, bool, error) {
			return clientConfig.Namespace()
		},
		Generators: func(cmdName string) map[string]kubectl.Generator {
			return DefaultGenerators(cmdName)
		},
		CanBeExposed: func(kind unversioned.GroupKind) error {
			switch kind {
			case api.Kind("ReplicationController"), api.Kind("Service"), api.Kind("Pod"), extensions.Kind("Deployment"), extensions.Kind("ReplicaSet"):
				// nothing to do here
			default:
				return fmt.Errorf("cannot expose a %s", kind)
			}
			return nil
		},
		CanBeAutoscaled: func(kind unversioned.GroupKind) error {
			switch kind {
			case api.Kind("ReplicationController"), extensions.Kind("Deployment"), extensions.Kind("ReplicaSet"):
				// nothing to do here
			default:
				return fmt.Errorf("cannot autoscale a %v", kind)
			}
			return nil
		},
		AttachablePodForObject: func(object runtime.Object) (*api.Pod, error) {
			client, err := clients.ClientForVersion(nil)
			if err != nil {
				return nil, err
			}
			switch t := object.(type) {
			case *api.ReplicationController:
				selector := labels.SelectorFromSet(t.Spec.Selector)
				sortBy := func(pods []*api.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
				pod, _, err := GetFirstPod(client, t.Namespace, selector, 1*time.Minute, sortBy)
				return pod, err
			case *extensions.Deployment:
				selector, err := unversioned.LabelSelectorAsSelector(t.Spec.Selector)
				if err != nil {
					return nil, fmt.Errorf("invalid label selector: %v", err)
				}
				sortBy := func(pods []*api.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
				pod, _, err := GetFirstPod(client, t.Namespace, selector, 1*time.Minute, sortBy)
				return pod, err
			case *batch.Job:
				selector, err := unversioned.LabelSelectorAsSelector(t.Spec.Selector)
				if err != nil {
					return nil, fmt.Errorf("invalid label selector: %v", err)
				}
				sortBy := func(pods []*api.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
				pod, _, err := GetFirstPod(client, t.Namespace, selector, 1*time.Minute, sortBy)
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
		},
		// UpdatePodSpecForObject update the pod specification for the provided object
		UpdatePodSpecForObject: func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error) {
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
			case *apps.PetSet:
				return true, fn(&t.Spec.Template.Spec)
			case *batch.Job:
				return true, fn(&t.Spec.Template.Spec)
			default:
				return false, fmt.Errorf("the object is not a pod or does not have a pod template")
			}
		},
		EditorEnvs: func() []string {
			return []string{"KUBE_EDITOR", "EDITOR"}
		},
		PrintObjectSpecificMessage: func(obj runtime.Object, out io.Writer) {
			switch obj := obj.(type) {
			case *api.Service:
				if obj.Spec.Type == api.ServiceTypeNodePort {
					msg := fmt.Sprintf(
						`You have exposed your service on an external port on all nodes in your
cluster.  If you want to expose this service to the external internet, you may
need to set up firewall rules for the service port(s) (%s) to serve traffic.

See http://releases.k8s.io/release-1.4/docs/user-guide/services-firewalls.md for more details.
`,
						makePortsString(obj.Spec.Ports, true))
					out.Write([]byte(msg))
				}

				if _, ok := obj.Annotations[service.AnnotationLoadBalancerSourceRangesKey]; ok {
					msg := fmt.Sprintf(
						`You are using service annotation [service.beta.kubernetes.io/load-balancer-source-ranges].
It has been promoted to field [loadBalancerSourceRanges] in service spec. This annotation will be deprecated in the future.
Please use the loadBalancerSourceRanges field instead.

See http://releases.k8s.io/release-1.4/docs/user-guide/services-firewalls.md for more details.
`)
					out.Write([]byte(msg))
				}
			}
		},
	}
}

// GetFirstPod returns a pod matching the namespace and label selector
// and the number of all pods that match the label selector.
func GetFirstPod(client client.PodsNamespacer, namespace string, selector labels.Selector, timeout time.Duration, sortBy func([]*api.Pod) sort.Interface) (*api.Pod, int, error) {
	options := api.ListOptions{LabelSelector: selector}

	podList, err := client.Pods(namespace).List(options)
	if err != nil {
		return nil, 0, err
	}
	pods := []*api.Pod{}
	for i := range podList.Items {
		pod := podList.Items[i]
		pods = append(pods, &pod)
	}
	if len(pods) > 0 {
		sort.Sort(sortBy(pods))
		return pods[0], len(podList.Items), nil
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

// Command will stringify and return all environment arguments ie. a command run by a client
// using the factory.
// TODO: We need to filter out stuff like secrets.
func (f *Factory) Command() string {
	if len(os.Args) == 0 {
		return ""
	}
	base := filepath.Base(os.Args[0])
	args := append([]string{base}, os.Args[1:]...)
	return strings.Join(args, " ")
}

// BindFlags adds any flags that are common to all kubectl sub commands.
func (f *Factory) BindFlags(flags *pflag.FlagSet) {
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

// BindCommonFlags adds any flags defined by external projects (not part of pflags)
func (f *Factory) BindExternalFlags(flags *pflag.FlagSet) {
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
	c        *client.Client
	fedc     *restclient.RESTClient
	cacheDir string
	mapper   meta.RESTMapper
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
		return fmt.Errorf("API version %q isn't supported, only supports API versions %q", gvk.GroupVersion().String(), registered.EnabledVersions())
	}
	switch gvk.Group {
	case autoscaling.GroupName:
		if c.c.AutoscalingClient == nil {
			return errors.New("unable to validate: no autoscaling client")
		}
		return getSchemaAndValidate(c.c.AutoscalingClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case policy.GroupName:
		if c.c.PolicyClient == nil {
			return errors.New("unable to validate: no policy client")
		}
		return getSchemaAndValidate(c.c.PolicyClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case apps.GroupName:
		if c.c.AppsClient == nil {
			return errors.New("unable to validate: no apps client")
		}
		return getSchemaAndValidate(c.c.AppsClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case batch.GroupName:
		if c.c.BatchClient == nil {
			return errors.New("unable to validate: no batch client")
		}
		return getSchemaAndValidate(c.c.BatchClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case rbac.GroupName:
		if c.c.RbacClient == nil {
			return errors.New("unable to validate: no rbac client")
		}
		return getSchemaAndValidate(c.c.RbacClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case storage.GroupName:
		if c.c.StorageClient == nil {
			return errors.New("unable to validate: no storage client")
		}
		return getSchemaAndValidate(c.c.StorageClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	}
	if registered.IsThirdPartyAPIGroupVersion(gvk.GroupVersion()) {
		// Don't attempt to validate third party objects
		return nil
	}
	switch gvk.Group {
	case extensions.GroupName:
		if c.c.ExtensionsClient == nil {
			return errors.New("unable to validate: no experimental client")
		}
		return getSchemaAndValidate(c.c.ExtensionsClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case federation.GroupName:
		if c.fedc == nil {
			return errors.New("unable to validate: no federation client")
		}
		return getSchemaAndValidate(c.fedc, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	case certificates.GroupName:
		if c.c.CertificatesClient == nil {
			return errors.New("unable to validate: no certificates client")
		}
		return getSchemaAndValidate(c.c.CertificatesClient.RESTClient, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	}
	return getSchemaAndValidate(c.c.RESTClient, data, "api", gvk.GroupVersion().String(), c.cacheDir, c)
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

// PrintObject prints an api object given command line flags to modify the output format
func (f *Factory) PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
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

// PrinterForMapping returns a printer suitable for displaying the provided resource type.
// Requires that printer flags have been added to cmd (see AddPrinterFlags).
func (f *Factory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error) {
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

// One stop shopping for a Builder
func (f *Factory) NewBuilder(thirdPartyDiscovery bool) *resource.Builder {
	mapper, typer := f.Object(thirdPartyDiscovery)

	return resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true))
}
