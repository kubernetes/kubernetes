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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube cache.
package app

import (
	"fmt"
	"strings"
	"time"

	"github.com/go-openapi/spec"
	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	apimachineryopenapi "k8s.io/apimachinery/pkg/openapi"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/filters"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/routes"
	"k8s.io/kubernetes/pkg/version"
)

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand() *cobra.Command {
	s := options.NewServerRunOptions()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "federation-apiserver",
		Long: `The Kubernetes federation API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}
	return cmd
}

// Run runs the specified APIServer.  This should never exit.
func Run(s *options.ServerRunOptions) error {
	// set defaults
	if err := s.GenericServerRunOptions.DefaultAdvertiseAddress(s.SecureServing, s.InsecureServing); err != nil {
		return err
	}
	if err := s.SecureServing.MaybeDefaultWithSelfSignedCerts(s.GenericServerRunOptions.AdvertiseAddress.String()); err != nil {
		return fmt.Errorf("error creating self-signed certificates: %v", err)
	}
	if err := s.CloudProvider.DefaultExternalHost(s.GenericServerRunOptions); err != nil {
		return fmt.Errorf("error setting the external host value: %v", err)
	}

	s.Authentication.ApplyAuthorization(s.Authorization)

	// validate options
	if errs := s.Validate(); len(errs) != 0 {
		return utilerrors.NewAggregate(errs)
	}

	genericConfig := genericapiserver.NewConfig().
		WithSerializer(api.Codecs).
		ApplyOptions(s.GenericServerRunOptions).
		ApplyInsecureServingOptions(s.InsecureServing)

	if _, err := genericConfig.ApplySecureServingOptions(s.SecureServing); err != nil {
		return fmt.Errorf("failed to configure https: %s", err)
	}
	if err := s.Authentication.Apply(genericConfig); err != nil {
		return fmt.Errorf("failed to configure authentication: %s", err)
	}

	// TODO: register cluster federation resources here.
	resourceConfig := genericapiserver.NewResourceConfig()

	if s.Etcd.StorageConfig.DeserializationCacheSize == 0 {
		// When size of cache is not explicitly set, set it to 50000
		s.Etcd.StorageConfig.DeserializationCacheSize = 50000
	}
	storageGroupsToEncodingVersion, err := s.StorageSerialization.StorageGroupsToEncodingVersion()
	if err != nil {
		return fmt.Errorf("error generating storage version map: %s", err)
	}
	storageFactory, err := kubeapiserver.BuildDefaultStorageFactory(
		s.Etcd.StorageConfig, s.GenericServerRunOptions.DefaultStorageMediaType, api.Codecs,
		genericapiserver.NewDefaultResourceEncodingConfig(api.Registry), storageGroupsToEncodingVersion,
		[]schema.GroupVersionResource{}, resourceConfig, s.GenericServerRunOptions.RuntimeConfig)
	if err != nil {
		return fmt.Errorf("error in initializing storage factory: %s", err)
	}

	for _, override := range s.Etcd.EtcdServersOverrides {
		tokens := strings.Split(override, "#")
		if len(tokens) != 2 {
			glog.Errorf("invalid value of etcd server overrides: %s", override)
			continue
		}

		apiresource := strings.Split(tokens[0], "/")
		if len(apiresource) != 2 {
			glog.Errorf("invalid resource definition: %s", tokens[0])
			continue
		}
		group := apiresource[0]
		resource := apiresource[1]
		groupResource := schema.GroupResource{Group: group, Resource: resource}

		servers := strings.Split(tokens[1], ";")
		storageFactory.SetEtcdLocation(groupResource, servers)
	}

	apiAuthenticator, securityDefinitions, err := s.Authentication.ToAuthenticationConfig().New()
	if err != nil {
		return fmt.Errorf("invalid Authentication Config: %v", err)
	}

	privilegedLoopbackToken := uuid.NewRandom().String()
	selfClientConfig, err := genericapiserver.NewSelfClientConfig(genericConfig.SecureServingInfo, genericConfig.InsecureServingInfo, privilegedLoopbackToken)
	if err != nil {
		return fmt.Errorf("failed to create clientset: %v", err)
	}
	client, err := internalclientset.NewForConfig(selfClientConfig)
	if err != nil {
		return fmt.Errorf("failed to create clientset: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(nil, client, 10*time.Minute)

	authorizationConfig := s.Authorization.ToAuthorizationConfig(sharedInformers)
	apiAuthorizer, err := authorizationConfig.New()
	if err != nil {
		return fmt.Errorf("invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.GenericServerRunOptions.AdmissionControl, ",")
	pluginInitializer := kubeapiserveradmission.NewPluginInitializer(client, sharedInformers, apiAuthorizer)
	admissionConfigProvider, err := kubeapiserveradmission.ReadAdmissionConfiguration(admissionControlPluginNames, s.GenericServerRunOptions.AdmissionControlConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read plugin config: %v", err)
	}
	admissionController, err := admission.NewFromPlugins(admissionControlPluginNames, admissionConfigProvider, pluginInitializer)
	if err != nil {
		return fmt.Errorf("failed to initialize plugins: %v", err)
	}

	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion
	genericConfig.LoopbackClientConfig = selfClientConfig
	genericConfig.Authenticator = apiAuthenticator
	genericConfig.Authorizer = apiAuthorizer
	genericConfig.AdmissionControl = admissionController
	genericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(openapi.GetOpenAPIDefinitions, api.Scheme)
	genericConfig.OpenAPIConfig.PostProcessSpec = postProcessOpenAPISpecForBackwardCompatibility
	genericConfig.OpenAPIConfig.SecurityDefinitions = securityDefinitions
	genericConfig.SwaggerConfig = genericapiserver.DefaultSwaggerConfig()
	genericConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	// TODO: Move this to generic api server (Need to move the command line flag).
	if s.GenericServerRunOptions.EnableWatchCache {
		cachesize.InitializeWatchCacheSizes(s.GenericServerRunOptions.TargetRAMMB)
		cachesize.SetWatchCacheSizes(s.GenericServerRunOptions.WatchCacheSizes)
	}

	m, err := genericConfig.Complete().New()
	if err != nil {
		return err
	}

	routes.UIRedirect{}.Install(m.HandlerContainer)
	routes.Logs{}.Install(m.HandlerContainer)

	// TODO: Refactor this code to share it with kube-apiserver rather than duplicating it here.
	restOptionsFactory := &restOptionsFactory{
		storageFactory:          storageFactory,
		enableGarbageCollection: s.GenericServerRunOptions.EnableGarbageCollection,
		deleteCollectionWorkers: s.GenericServerRunOptions.DeleteCollectionWorkers,
	}
	if s.GenericServerRunOptions.EnableWatchCache {
		restOptionsFactory.storageDecorator = genericregistry.StorageWithCacher
	} else {
		restOptionsFactory.storageDecorator = generic.UndecoratedStorage
	}

	installFederationAPIs(m, restOptionsFactory)
	installCoreAPIs(s, m, restOptionsFactory)
	installExtensionsAPIs(m, restOptionsFactory)
	installBatchAPIs(m, restOptionsFactory)
	installAutoscalingAPIs(m, restOptionsFactory)

	sharedInformers.Start(wait.NeverStop)
	m.PrepareRun().Run(wait.NeverStop)
	return nil
}

type restOptionsFactory struct {
	storageFactory          genericapiserver.StorageFactory
	storageDecorator        generic.StorageDecorator
	deleteCollectionWorkers int
	enableGarbageCollection bool
}

func (f *restOptionsFactory) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	config, err := f.storageFactory.NewConfig(resource)
	if err != nil {
		return generic.RESTOptions{}, fmt.Errorf("Unable to find storage config for %v, due to %v", resource, err.Error())
	}
	return generic.RESTOptions{
		StorageConfig:           config,
		Decorator:               f.storageDecorator,
		DeleteCollectionWorkers: f.deleteCollectionWorkers,
		EnableGarbageCollection: f.enableGarbageCollection,
		ResourcePrefix:          f.storageFactory.ResourcePrefix(resource),
	}, nil
}

// PostProcessSpec adds removed definitions for backward compatibility
func postProcessOpenAPISpecForBackwardCompatibility(s *spec.Swagger) (*spec.Swagger, error) {
	compatibilityMap := map[string]string{
		"v1beta1.ReplicaSetList":           "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetList",
		"v1.FlockerVolumeSource":           "io.k8s.kubernetes.pkg.api.v1.FlockerVolumeSource",
		"v1.FlexVolumeSource":              "io.k8s.kubernetes.pkg.api.v1.FlexVolumeSource",
		"v1.SecretKeySelector":             "io.k8s.kubernetes.pkg.api.v1.SecretKeySelector",
		"v1.DeleteOptions":                 "io.k8s.apimachinery.pkg.apis.meta.v1.DeleteOptions",
		"v1.ServiceSpec":                   "io.k8s.kubernetes.pkg.api.v1.ServiceSpec",
		"v1.NamespaceStatus":               "io.k8s.kubernetes.pkg.api.v1.NamespaceStatus",
		"v1.Affinity":                      "io.k8s.kubernetes.pkg.api.v1.Affinity",
		"v1.PodAffinity":                   "io.k8s.kubernetes.pkg.api.v1.PodAffinity",
		"v1.EnvVarSource":                  "io.k8s.kubernetes.pkg.api.v1.EnvVarSource",
		"v1.ListMeta":                      "io.k8s.apimachinery.pkg.apis.meta.v1.ListMeta",
		"v1.ObjectMeta":                    "io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta",
		"v1.APIGroupList":                  "io.k8s.apimachinery.pkg.apis.meta.v1.APIGroupList",
		"v1.EnvFromSource":                 "io.k8s.kubernetes.pkg.api.v1.EnvFromSource",
		"v1.Service":                       "io.k8s.kubernetes.pkg.api.v1.Service",
		"v1.HorizontalPodAutoscaler":       "io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscaler",
		"v1.StatusCause":                   "io.k8s.apimachinery.pkg.apis.meta.v1.StatusCause",
		"v1.ObjectFieldSelector":           "io.k8s.kubernetes.pkg.api.v1.ObjectFieldSelector",
		"v1.QuobyteVolumeSource":           "io.k8s.kubernetes.pkg.api.v1.QuobyteVolumeSource",
		"v1beta1.ReplicaSetSpec":           "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetSpec",
		"v1.LabelSelector":                 "io.k8s.apimachinery.pkg.apis.meta.v1.LabelSelector",
		"v1.DownwardAPIVolumeFile":         "io.k8s.kubernetes.pkg.api.v1.DownwardAPIVolumeFile",
		"v1.GCEPersistentDiskVolumeSource": "io.k8s.kubernetes.pkg.api.v1.GCEPersistentDiskVolumeSource",
		"v1beta1.ClusterCondition":         "io.k8s.kubernetes.federation.apis.federation.v1beta1.ClusterCondition",
		"v1.JobCondition":                  "io.k8s.kubernetes.pkg.apis.batch.v1.JobCondition",
		"v1.LabelSelectorRequirement":      "io.k8s.apimachinery.pkg.apis.meta.v1.LabelSelectorRequirement",
		"v1beta1.Deployment":               "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.Deployment",
		"v1.LoadBalancerIngress":           "io.k8s.kubernetes.pkg.api.v1.LoadBalancerIngress",
		"v1.SecretList":                    "io.k8s.kubernetes.pkg.api.v1.SecretList",
		"v1.ServicePort":                   "io.k8s.kubernetes.pkg.api.v1.ServicePort",
		"v1.Namespace":                     "io.k8s.kubernetes.pkg.api.v1.Namespace",
		"v1beta1.ReplicaSetCondition":      "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetCondition",
		"v1.CrossVersionObjectReference":   "io.k8s.kubernetes.pkg.apis.autoscaling.v1.CrossVersionObjectReference",
		"v1.ConfigMapVolumeSource":         "io.k8s.kubernetes.pkg.api.v1.ConfigMapVolumeSource",
		"v1.FCVolumeSource":                "io.k8s.kubernetes.pkg.api.v1.FCVolumeSource",
		"v1.GroupVersionForDiscovery":      "io.k8s.apimachinery.pkg.apis.meta.v1.GroupVersionForDiscovery",
		"v1beta1.ClusterStatus":            "io.k8s.kubernetes.federation.apis.federation.v1beta1.ClusterStatus",
		"v1.Job":                           "io.k8s.kubernetes.pkg.apis.batch.v1.Job",
		"v1.PersistentVolumeClaimVolumeSource": "io.k8s.kubernetes.pkg.api.v1.PersistentVolumeClaimVolumeSource",
		"v1.Handler":                           "io.k8s.kubernetes.pkg.api.v1.Handler",
		"v1.ServerAddressByClientCIDR":         "io.k8s.apimachinery.pkg.apis.meta.v1.ServerAddressByClientCIDR",
		"v1.PodAntiAffinity":                   "io.k8s.kubernetes.pkg.api.v1.PodAntiAffinity",
		"v1.ISCSIVolumeSource":                 "io.k8s.kubernetes.pkg.api.v1.ISCSIVolumeSource",
		"v1.WeightedPodAffinityTerm":           "io.k8s.kubernetes.pkg.api.v1.WeightedPodAffinityTerm",
		"v1.HorizontalPodAutoscalerSpec":       "io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscalerSpec",
		"v1.HorizontalPodAutoscalerList":       "io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscalerList",
		"v1.Probe":                             "io.k8s.kubernetes.pkg.api.v1.Probe",
		"v1.APIGroup":                          "io.k8s.apimachinery.pkg.apis.meta.v1.APIGroup",
		"v1beta1.DeploymentList":               "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentList",
		"v1.NodeAffinity":                      "io.k8s.kubernetes.pkg.api.v1.NodeAffinity",
		"v1.SecretEnvSource":                   "io.k8s.kubernetes.pkg.api.v1.SecretEnvSource",
		"v1beta1.DeploymentStatus":             "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentStatus",
		"v1.CinderVolumeSource":                "io.k8s.kubernetes.pkg.api.v1.CinderVolumeSource",
		"v1.NodeSelectorTerm":                  "io.k8s.kubernetes.pkg.api.v1.NodeSelectorTerm",
		"v1.Patch":                             "io.k8s.apimachinery.pkg.apis.meta.v1.Patch",
		"v1.SecretVolumeSource":                "io.k8s.kubernetes.pkg.api.v1.SecretVolumeSource",
		"v1.Secret":                            "io.k8s.kubernetes.pkg.api.v1.Secret",
		"v1.NodeSelector":                      "io.k8s.kubernetes.pkg.api.v1.NodeSelector",
		"runtime.RawExtension":                 "io.k8s.apimachinery.pkg.runtime.RawExtension",
		"v1.PreferredSchedulingTerm":           "io.k8s.kubernetes.pkg.api.v1.PreferredSchedulingTerm",
		"v1beta1.ClusterList":                  "io.k8s.kubernetes.federation.apis.federation.v1beta1.ClusterList",
		"v1.KeyToPath":                         "io.k8s.kubernetes.pkg.api.v1.KeyToPath",
		"intstr.IntOrString":                   "io.k8s.apimachinery.pkg.util.intstr.IntOrString",
		"v1beta1.ClusterSpec":                  "io.k8s.kubernetes.federation.apis.federation.v1beta1.ClusterSpec",
		"v1.ServiceList":                       "io.k8s.kubernetes.pkg.api.v1.ServiceList",
		"v1beta1.DeploymentStrategy":           "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentStrategy",
		"v1beta1.IngressBackend":               "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressBackend",
		"v1.Time":                              "io.k8s.apimachinery.pkg.apis.meta.v1.Time",
		"v1.ContainerPort":                     "io.k8s.kubernetes.pkg.api.v1.ContainerPort",
		"v1beta1.HTTPIngressRuleValue":         "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.HTTPIngressRuleValue",
		"v1.AzureFileVolumeSource":             "io.k8s.kubernetes.pkg.api.v1.AzureFileVolumeSource",
		"v1.PodTemplateSpec":                   "io.k8s.kubernetes.pkg.api.v1.PodTemplateSpec",
		"v1.PodSpec":                           "io.k8s.kubernetes.pkg.api.v1.PodSpec",
		"v1beta1.ReplicaSetStatus":             "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetStatus",
		"v1.CephFSVolumeSource":                "io.k8s.kubernetes.pkg.api.v1.CephFSVolumeSource",
		"v1.Volume":                            "io.k8s.kubernetes.pkg.api.v1.Volume",
		"v1beta1.Ingress":                      "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.Ingress",
		"v1.PodAffinityTerm":                   "io.k8s.kubernetes.pkg.api.v1.PodAffinityTerm",
		"v1.ObjectReference":                   "io.k8s.kubernetes.pkg.api.v1.ObjectReference",
		"v1.ServiceStatus":                     "io.k8s.kubernetes.pkg.api.v1.ServiceStatus",
		"v1.APIResource":                       "io.k8s.apimachinery.pkg.apis.meta.v1.APIResource",
		"v1.AzureDiskVolumeSource":             "io.k8s.kubernetes.pkg.api.v1.AzureDiskVolumeSource",
		"v1.ConfigMap":                         "io.k8s.kubernetes.pkg.api.v1.ConfigMap",
		"v1beta1.IngressSpec":                  "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressSpec",
		"v1.APIVersions":                       "io.k8s.apimachinery.pkg.apis.meta.v1.APIVersions",
		"resource.Quantity":                    "io.k8s.apimachinery.pkg.api.resource.Quantity",
		"v1.Event":                             "io.k8s.kubernetes.pkg.api.v1.Event",
		"v1.JobStatus":                         "io.k8s.kubernetes.pkg.apis.batch.v1.JobStatus",
		"v1beta1.ServerAddressByClientCIDR":    "io.k8s.kubernetes.federation.apis.federation.v1beta1.ServerAddressByClientCIDR",
		"v1.LocalObjectReference":              "io.k8s.kubernetes.pkg.api.v1.LocalObjectReference",
		"v1.HostPathVolumeSource":              "io.k8s.kubernetes.pkg.api.v1.HostPathVolumeSource",
		"v1.LoadBalancerStatus":                "io.k8s.kubernetes.pkg.api.v1.LoadBalancerStatus",
		"v1beta1.HTTPIngressPath":              "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.HTTPIngressPath",
		"v1beta1.DeploymentSpec":               "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentSpec",
		"v1.ExecAction":                        "io.k8s.kubernetes.pkg.api.v1.ExecAction",
		"v1.HorizontalPodAutoscalerStatus":     "io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscalerStatus",
		"v1.JobSpec":                           "io.k8s.kubernetes.pkg.apis.batch.v1.JobSpec",
		"v1beta1.DaemonSetSpec":                "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetSpec",
		"v1.SELinuxOptions":                    "io.k8s.kubernetes.pkg.api.v1.SELinuxOptions",
		"v1beta1.IngressTLS":                   "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressTLS",
		"v1beta1.ScaleStatus":                  "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ScaleStatus",
		"v1.NamespaceSpec":                     "io.k8s.kubernetes.pkg.api.v1.NamespaceSpec",
		"v1.StatusDetails":                     "io.k8s.apimachinery.pkg.apis.meta.v1.StatusDetails",
		"v1beta1.IngressList":                  "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressList",
		"v1beta1.DeploymentRollback":           "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentRollback",
		"v1.GlusterfsVolumeSource":             "io.k8s.kubernetes.pkg.api.v1.GlusterfsVolumeSource",
		"v1.JobList":                           "io.k8s.kubernetes.pkg.apis.batch.v1.JobList",
		"v1.EventList":                         "io.k8s.kubernetes.pkg.api.v1.EventList",
		"v1beta1.IngressRule":                  "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressRule",
		"v1.APIResourceList":                   "io.k8s.apimachinery.pkg.apis.meta.v1.APIResourceList",
		"v1.ConfigMapKeySelector":              "io.k8s.kubernetes.pkg.api.v1.ConfigMapKeySelector",
		"v1.PhotonPersistentDiskVolumeSource":  "io.k8s.kubernetes.pkg.api.v1.PhotonPersistentDiskVolumeSource",
		"v1.HTTPHeader":                        "io.k8s.kubernetes.pkg.api.v1.HTTPHeader",
		"version.Info":                         "io.k8s.apimachinery.pkg.version.Info",
		"v1.EventSource":                       "io.k8s.kubernetes.pkg.api.v1.EventSource",
		"v1.OwnerReference":                    "io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference",
		"v1beta1.ScaleSpec":                    "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ScaleSpec",
		"v1.GitRepoVolumeSource":               "io.k8s.kubernetes.pkg.api.v1.GitRepoVolumeSource",
		"v1.ConfigMapEnvSource":                "io.k8s.kubernetes.pkg.api.v1.ConfigMapEnvSource",
		"v1beta1.DeploymentCondition":          "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentCondition",
		"v1.EnvVar":                            "io.k8s.kubernetes.pkg.api.v1.EnvVar",
		"v1.DownwardAPIVolumeSource":           "io.k8s.kubernetes.pkg.api.v1.DownwardAPIVolumeSource",
		"v1.SecurityContext":                   "io.k8s.kubernetes.pkg.api.v1.SecurityContext",
		"v1beta1.IngressStatus":                "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressStatus",
		"v1beta1.Cluster":                      "io.k8s.kubernetes.federation.apis.federation.v1beta1.Cluster",
		"v1.Capabilities":                      "io.k8s.kubernetes.pkg.api.v1.Capabilities",
		"v1.AWSElasticBlockStoreVolumeSource":  "io.k8s.kubernetes.pkg.api.v1.AWSElasticBlockStoreVolumeSource",
		"v1beta1.ReplicaSet":                   "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSet",
		"v1.ConfigMapList":                     "io.k8s.kubernetes.pkg.api.v1.ConfigMapList",
		"v1.Lifecycle":                         "io.k8s.kubernetes.pkg.api.v1.Lifecycle",
		"v1beta1.Scale":                        "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.Scale",
		"v1beta1.DaemonSet":                    "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSet",
		"v1beta1.RollingUpdateDeployment":      "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.RollingUpdateDeployment",
		"v1beta1.DaemonSetStatus":              "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetStatus",
		"v1.Preconditions":                     "io.k8s.apimachinery.pkg.apis.meta.v1.Preconditions",
		"v1beta1.DaemonSetList":                "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetList",
		"v1.RBDVolumeSource":                   "io.k8s.kubernetes.pkg.api.v1.RBDVolumeSource",
		"v1.NFSVolumeSource":                   "io.k8s.kubernetes.pkg.api.v1.NFSVolumeSource",
		"v1.NodeSelectorRequirement":           "io.k8s.kubernetes.pkg.api.v1.NodeSelectorRequirement",
		"v1.ResourceRequirements":              "io.k8s.kubernetes.pkg.api.v1.ResourceRequirements",
		"v1.WatchEvent":                        "io.k8s.apimachinery.pkg.apis.meta.v1.WatchEvent",
		"v1.HTTPGetAction":                     "io.k8s.kubernetes.pkg.api.v1.HTTPGetAction",
		"v1beta1.RollbackConfig":               "io.k8s.kubernetes.pkg.apis.extensions.v1beta1.RollbackConfig",
		"v1.PodSecurityContext":                "io.k8s.kubernetes.pkg.api.v1.PodSecurityContext",
		"v1.VolumeMount":                       "io.k8s.kubernetes.pkg.api.v1.VolumeMount",
		"v1.NamespaceList":                     "io.k8s.kubernetes.pkg.api.v1.NamespaceList",
		"v1.TCPSocketAction":                   "io.k8s.kubernetes.pkg.api.v1.TCPSocketAction",
		"v1.ResourceFieldSelector":             "io.k8s.kubernetes.pkg.api.v1.ResourceFieldSelector",
		"v1.Container":                         "io.k8s.kubernetes.pkg.api.v1.Container",
		"v1.VsphereVirtualDiskVolumeSource":    "io.k8s.kubernetes.pkg.api.v1.VsphereVirtualDiskVolumeSource",
		"v1.EmptyDirVolumeSource":              "io.k8s.kubernetes.pkg.api.v1.EmptyDirVolumeSource",
		"v1.Status":                            "io.k8s.apimachinery.pkg.apis.meta.v1.Status",
	}

	for k, v := range compatibilityMap {
		if _, found := s.Definitions[v]; !found {
			continue
		}
		s.Definitions[k] = spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref:         spec.MustCreateRef("#/definitions/" + apimachineryopenapi.EscapeJsonPointer(v)),
				Description: fmt.Sprintf("Deprecated. Please use %s instead.", v),
			},
		}
	}
	return s, nil
}
