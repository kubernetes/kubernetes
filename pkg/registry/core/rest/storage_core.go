/*
Copyright 2016 The Kubernetes Authors.

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

package rest

import (
	"crypto/tls"
	goerrors "errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	networkingv1client "k8s.io/client-go/kubernetes/typed/networking/v1"
	policyclient "k8s.io/client-go/kubernetes/typed/policy/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/core/componentstatus"
	endpointsstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	limitrangestore "k8s.io/kubernetes/pkg/registry/core/limitrange/storage"
	nodestore "k8s.io/kubernetes/pkg/registry/core/node/storage"
	pvstore "k8s.io/kubernetes/pkg/registry/core/persistentvolume/storage"
	pvcstore "k8s.io/kubernetes/pkg/registry/core/persistentvolumeclaim/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	podtemplatestore "k8s.io/kubernetes/pkg/registry/core/podtemplate/storage"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	controllerstore "k8s.io/kubernetes/pkg/registry/core/replicationcontroller/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	serviceallocator "k8s.io/kubernetes/pkg/registry/core/service/allocator/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	serviceipallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
	servicestore "k8s.io/kubernetes/pkg/registry/core/service/storage"
	serviceaccountstore "k8s.io/kubernetes/pkg/registry/core/serviceaccount/storage"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/util/async"
	netutils "k8s.io/utils/net"
)

// Config provides information needed to build RESTStorage for core.
type Config struct {
	GenericConfig

	Proxy    ProxyConfig
	Services ServicesConfig
}

type ProxyConfig struct {
	Transport           http.RoundTripper
	KubeletClientConfig kubeletclient.KubeletClientConfig
}

type ServicesConfig struct {
	// Service IP ranges
	ClusterIPRange          net.IPNet
	SecondaryClusterIPRange net.IPNet
	NodePortRange           utilnet.PortRange

	IPRepairInterval time.Duration
}

type rangeRegistries struct {
	clusterIP          rangeallocation.RangeRegistry
	secondaryClusterIP rangeallocation.RangeRegistry
	nodePort           rangeallocation.RangeRegistry
}

type legacyProvider struct {
	Config

	primaryServiceClusterIPAllocator ipallocator.Interface
	serviceClusterIPAllocators       map[api.IPFamily]ipallocator.Interface
	serviceNodePortAllocator         *portallocator.PortAllocator
	authorizer                       authorizer.Authorizer

	startServiceNodePortsRepair, startServiceClusterIPRepair func(onFirstSuccess func(), stopCh chan struct{})
}

func New(c Config, authorizer authorizer.Authorizer) (*legacyProvider, error) {
	rangeRegistries, serviceClusterIPAllocator, serviceIPAllocators, serviceNodePortAllocator, err := c.newServiceIPAllocators()
	if err != nil {
		return nil, err
	}

	p := &legacyProvider{
		Config: c,

		primaryServiceClusterIPAllocator: serviceClusterIPAllocator,
		serviceClusterIPAllocators:       serviceIPAllocators,
		serviceNodePortAllocator:         serviceNodePortAllocator,
		authorizer:                       authorizer,
	}

	// create service node port repair controller
	client, err := kubernetes.NewForConfig(c.LoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	p.startServiceNodePortsRepair = portallocatorcontroller.NewRepair(c.Services.IPRepairInterval, client.CoreV1(), client.EventsV1(), c.Services.NodePortRange, rangeRegistries.nodePort).RunUntil

	// create service cluster ip repair controller
	if !utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
		p.startServiceClusterIPRepair = serviceipallocatorcontroller.NewRepair(
			c.Services.IPRepairInterval,
			client.CoreV1(),
			client.EventsV1(),
			&c.Services.ClusterIPRange,
			rangeRegistries.clusterIP,
			&c.Services.SecondaryClusterIPRange,
			rangeRegistries.secondaryClusterIP,
		).RunUntil
	} else {
		p.startServiceClusterIPRepair = serviceipallocatorcontroller.NewRepairIPAddress(
			c.Services.IPRepairInterval,
			client,
			c.Informers.Core().V1().Services(),
			c.Informers.Networking().V1().ServiceCIDRs(),
			c.Informers.Networking().V1().IPAddresses(),
		).RunUntil
	}

	return p, nil
}

func (p *legacyProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo, err := p.GenericConfig.NewRESTStorage(apiResourceConfigSource, restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	podDisruptionClient, err := policyclient.NewForConfig(p.LoopbackClientConfig)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	podTemplateStorage, err := podtemplatestore.NewREST(restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	limitRangeStorage, err := limitrangestore.NewREST(restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	persistentVolumeStorage, persistentVolumeStatusStorage, err := pvstore.NewREST(restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}
	persistentVolumeClaimStorage, persistentVolumeClaimStatusStorage, err := pvcstore.NewREST(restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	endpointsStorage, err := endpointsstore.NewREST(restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	nodeStorage, err := nodestore.NewStorage(restOptionsGetter, p.Proxy.KubeletClientConfig, p.Proxy.Transport)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	podStorage, err := podstore.NewStorage(
		restOptionsGetter,
		nodeStorage.KubeletConnectionInfo,
		p.Proxy.Transport,
		podDisruptionClient,
		p.authorizer,
	)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	serviceRESTStorage, serviceStatusStorage, serviceRESTProxy, err := servicestore.NewREST(
		restOptionsGetter,
		p.primaryServiceClusterIPAllocator.IPFamily(),
		p.serviceClusterIPAllocators,
		p.serviceNodePortAllocator,
		endpointsStorage,
		podStorage.Pod,
		p.Proxy.Transport)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}

	storage := apiGroupInfo.VersionedResourcesStorageMap["v1"]
	if storage == nil {
		storage = map[string]rest.Storage{}
	}

	// potentially override the generic serviceaccount storage with one that supports pods
	var serviceAccountStorage *serviceaccountstore.REST
	if p.ServiceAccountIssuer != nil {
		var nodeGetter rest.Getter
		if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenNodeBinding) ||
			utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountTokenPodNodeInfo) {
			nodeGetter = nodeStorage.Node.Store
		}
		serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, p.ServiceAccountIssuer, p.APIAudiences, p.ServiceAccountMaxExpiration, podStorage.Pod.Store, storage["secrets"].(rest.Getter), nodeGetter, p.ExtendExpiration, p.MaxExtendedExpiration)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}
	}

	if resource := "pods"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = podStorage.Pod
		storage[resource+"/attach"] = podStorage.Attach
		storage[resource+"/status"] = podStorage.Status
		storage[resource+"/log"] = podStorage.Log
		storage[resource+"/exec"] = podStorage.Exec
		storage[resource+"/portforward"] = podStorage.PortForward
		storage[resource+"/proxy"] = podStorage.Proxy
		storage[resource+"/binding"] = podStorage.Binding
		if podStorage.Eviction != nil {
			storage[resource+"/eviction"] = podStorage.Eviction
		}
		storage[resource+"/ephemeralcontainers"] = podStorage.EphemeralContainers
		if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
			storage[resource+"/resize"] = podStorage.Resize
		}
	}
	if resource := "bindings"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = podStorage.LegacyBinding
	}

	if resource := "podtemplates"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = podTemplateStorage
	}

	if resource := "replicationcontrollers"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		controllerStorage, err := controllerstore.NewStorage(restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, err
		}

		storage[resource] = controllerStorage.Controller
		storage[resource+"/status"] = controllerStorage.Status
		if legacyscheme.Scheme.IsVersionRegistered(schema.GroupVersion{Group: "autoscaling", Version: "v1"}) {
			storage[resource+"/scale"] = controllerStorage.Scale
		}
	}

	// potentially override generic storage for service account (with pod support)
	if resource := "serviceaccounts"; serviceAccountStorage != nil && apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		// don't leak go routines
		storage[resource].Destroy()
		if storage[resource+"/token"] != nil {
			storage[resource+"/token"].Destroy()
		}

		storage[resource] = serviceAccountStorage
		if serviceAccountStorage.Token != nil {
			storage[resource+"/token"] = serviceAccountStorage.Token
		}
	}

	if resource := "services"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = serviceRESTStorage
		storage[resource+"/proxy"] = serviceRESTProxy
		storage[resource+"/status"] = serviceStatusStorage
	}

	if resource := "endpoints"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = endpointsStorage
	}

	if resource := "nodes"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = nodeStorage.Node
		storage[resource+"/proxy"] = nodeStorage.Proxy
		storage[resource+"/status"] = nodeStorage.Status
	}

	if resource := "limitranges"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = limitRangeStorage
	}

	if resource := "persistentvolumes"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = persistentVolumeStorage
		storage[resource+"/status"] = persistentVolumeStatusStorage
	}

	if resource := "persistentvolumeclaims"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = persistentVolumeClaimStorage
		storage[resource+"/status"] = persistentVolumeClaimStatusStorage
	}

	if resource := "componentstatuses"; apiResourceConfigSource.ResourceEnabled(corev1.SchemeGroupVersion.WithResource(resource)) {
		storage[resource] = componentstatus.NewStorage(componentStatusStorage{p.StorageFactory}.serversToValidate)
	}

	if len(storage) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap["v1"] = storage
	}

	return apiGroupInfo, nil
}

func (c *Config) newServiceIPAllocators() (registries rangeRegistries, primaryClusterIPAllocator ipallocator.Interface, clusterIPAllocators map[api.IPFamily]ipallocator.Interface, nodePortAllocator *portallocator.PortAllocator, err error) {
	clusterIPAllocators = map[api.IPFamily]ipallocator.Interface{}

	serviceStorageConfig, err := c.StorageFactory.NewConfig(api.Resource("services"), &api.Service{})
	if err != nil {
		return rangeRegistries{}, nil, nil, nil, err
	}

	serviceClusterIPRange := c.Services.ClusterIPRange
	if serviceClusterIPRange.IP == nil {
		return rangeRegistries{}, nil, nil, nil, fmt.Errorf("service clusterIPRange is missing")
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
		primaryClusterIPAllocator, err = ipallocator.New(&serviceClusterIPRange, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
			var mem allocator.Snapshottable
			mem = allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
			// TODO etcdallocator package to return a storage interface via the storageFactory
			etcd, err := serviceallocator.NewEtcd(mem, "/ranges/serviceips", serviceStorageConfig.ForResource(api.Resource("serviceipallocations")))
			if err != nil {
				return nil, err
			}
			registries.clusterIP = etcd
			return etcd, nil
		})
		if err != nil {
			return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster IP allocator: %v", err)
		}
	} else {
		networkingv1Client, err := networkingv1client.NewForConfig(c.LoopbackClientConfig)
		if err != nil {
			return rangeRegistries{}, nil, nil, nil, err
		}
		var bitmapAllocator ipallocator.Interface
		if !utilfeature.DefaultFeatureGate.Enabled(features.DisableAllocatorDualWrite) {
			bitmapAllocator, err = ipallocator.New(&serviceClusterIPRange, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
				mem := allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
				etcd, err := serviceallocator.NewEtcd(mem, "/ranges/serviceips", serviceStorageConfig.ForResource(api.Resource("serviceipallocations")))
				if err != nil {
					return nil, err
				}
				// It is possible to start apiserver clusters with the new allocator and dual write enable on new environments.
				// If this is the case we need to initialize the bitmap or it will fail to allocate IP addresses because
				// the ResourceVersion of the opaque API object is zero.
				rangeRegistry, err := etcd.Get()
				if err != nil {
					return nil, err
				}
				rangeRegistry.Range = serviceClusterIPRange.String()
				if len(rangeRegistry.ResourceVersion) == 0 {
					klog.Infof("kube-apiserver started with IP allocator and dual write enabled but bitmap allocator does not exist, recreating it ...")
					err := etcd.CreateOrUpdate(rangeRegistry)
					if err != nil {
						return nil, err
					}
				}
				registries.clusterIP = etcd
				return etcd, nil
			})
			if err != nil {
				return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster IP allocator: %w", err)
			}

		}
		// TODO(aojea) Revisit the initialization of the allocators
		// since right now it depends on the service-cidr flags and
		// sets the default IPFamily that may not be coherent with the
		// existing default ServiceCIDR
		primaryClusterIPAllocator, err = ipallocator.NewMetaAllocator(
			networkingv1Client,
			c.Informers.Networking().V1().ServiceCIDRs(),
			c.Informers.Networking().V1().IPAddresses(),
			netutils.IsIPv6CIDR(&serviceClusterIPRange),
			bitmapAllocator,
		)
		if err != nil {
			return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster IP allocator: %v", err)
		}
	}
	primaryClusterIPAllocator.EnableMetrics()
	clusterIPAllocators[primaryClusterIPAllocator.IPFamily()] = primaryClusterIPAllocator

	var secondaryClusterIPAllocator ipallocator.Interface
	if c.Services.SecondaryClusterIPRange.IP != nil {
		if !utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
			var err error
			secondaryClusterIPAllocator, err = ipallocator.New(&c.Services.SecondaryClusterIPRange, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
				var mem allocator.Snapshottable
				mem = allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
				// TODO etcdallocator package to return a storage interface via the storageFactory
				etcd, err := serviceallocator.NewEtcd(mem, "/ranges/secondaryserviceips", serviceStorageConfig.ForResource(api.Resource("serviceipallocations")))
				if err != nil {
					return nil, err
				}
				registries.secondaryClusterIP = etcd
				return etcd, nil
			})
			if err != nil {
				return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster secondary IP allocator: %v", err)
			}
		} else {
			networkingv1Client, err := networkingv1client.NewForConfig(c.LoopbackClientConfig)
			if err != nil {
				return rangeRegistries{}, nil, nil, nil, err
			}
			var bitmapAllocator ipallocator.Interface
			if !utilfeature.DefaultFeatureGate.Enabled(features.DisableAllocatorDualWrite) {
				bitmapAllocator, err = ipallocator.New(&c.Services.SecondaryClusterIPRange, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
					mem := allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
					// TODO etcdallocator package to return a storage interface via the storageFactory
					etcd, err := serviceallocator.NewEtcd(mem, "/ranges/secondaryserviceips", serviceStorageConfig.ForResource(api.Resource("serviceipallocations")))
					if err != nil {
						return nil, err
					}
					// It is possible to start apiserver clusters with the new allocator and dual write enable on new environments.
					// If this is the case we need to initialize the bitmap or it will fail to allocate IP addresses because
					// the ResourceVersion of the opaque API object is zero.
					rangeRegistry, err := etcd.Get()
					if err != nil {
						return nil, err
					}
					rangeRegistry.Range = c.Services.SecondaryClusterIPRange.String()
					if len(rangeRegistry.ResourceVersion) == 0 {
						klog.Infof("kube-apiserver started with IP allocator and dual write enabled but bitmap allocator does not exist, recreating it ...")
						err := etcd.CreateOrUpdate(rangeRegistry)
						if err != nil {
							return nil, err
						}
					}
					registries.secondaryClusterIP = etcd
					return etcd, nil
				})
				if err != nil {
					return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster secondary IP allocator: %w", err)
				}
			}
			// TODO(aojea) Revisit the initialization of the allocators
			// since right now it depends on the service-cidr flags and
			// sets the default IPFamily that may not be coherent with the
			// existing default ServiceCIDR
			secondaryClusterIPAllocator, err = ipallocator.NewMetaAllocator(
				networkingv1Client,
				c.Informers.Networking().V1().ServiceCIDRs(),
				c.Informers.Networking().V1().IPAddresses(),
				netutils.IsIPv6CIDR(&c.Services.SecondaryClusterIPRange),
				bitmapAllocator,
			)
			if err != nil {
				return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster secondary IP allocator: %v", err)
			}
		}
		secondaryClusterIPAllocator.EnableMetrics()
		clusterIPAllocators[secondaryClusterIPAllocator.IPFamily()] = secondaryClusterIPAllocator
	}

	nodePortAllocator, err = portallocator.New(c.Services.NodePortRange, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
		mem := allocator.NewAllocationMapWithOffset(max, rangeSpec, offset)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd, err := serviceallocator.NewEtcd(mem, "/ranges/servicenodeports", serviceStorageConfig.ForResource(api.Resource("servicenodeportallocations")))
		if err != nil {
			return nil, err
		}
		registries.nodePort = etcd
		return etcd, nil
	})
	if err != nil {
		return rangeRegistries{}, nil, nil, nil, fmt.Errorf("cannot create cluster port allocator: %v", err)
	}
	nodePortAllocator.EnableMetrics()

	return
}

var _ genericapiserver.PostStartHookProvider = &legacyProvider{}

func (p *legacyProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return "start-service-ip-repair-controllers", func(context genericapiserver.PostStartHookContext) error {
		// We start both repairClusterIPs and repairNodePorts to ensure repair
		// loops of ClusterIPs and NodePorts.
		// We run both repair loops using RunUntil public interface.
		// However, we want to fail liveness/readiness until the first
		// successful repair loop, so we basically pass appropriate
		// callbacks to RunUtil methods.
		// Additionally, we ensure that we don't wait for it for longer
		// than 1 minute for backward compatibility of failing the whole
		// apiserver if we can't repair them.
		wg := sync.WaitGroup{}
		wg.Add(2)
		runner := async.NewRunner(
			func(stopCh chan struct{}) { p.startServiceClusterIPRepair(wg.Done, stopCh) },
			func(stopCh chan struct{}) { p.startServiceNodePortsRepair(wg.Done, stopCh) },
		)
		runner.Start()
		go func() {
			defer runner.Stop()
			<-context.Done()
		}()

		// For backward compatibility, we ensure that if we never are able
		// to repair clusterIPs and/or nodeports, we not only fail the liveness
		// and/or readiness, but also explicitly fail.
		done := make(chan struct{})
		go func() {
			defer close(done)
			wg.Wait()
		}()
		select {
		case <-done:
		case <-time.After(time.Minute):
			return goerrors.New("unable to perform initial IP and Port allocation check")
		}

		return nil
	}, nil
}

func (p *legacyProvider) GroupName() string {
	return api.GroupName
}

type componentStatusStorage struct {
	storageFactory serverstorage.StorageFactory
}

func (s componentStatusStorage) serversToValidate() map[string]componentstatus.Server {
	// this is fragile, which assumes that the default port is being used
	// TODO: switch to secure port until these components remove the ability to serve insecurely.
	serversToValidate := map[string]componentstatus.Server{
		"controller-manager": &componentstatus.HttpServer{EnableHTTPS: true, TLSConfig: &tls.Config{InsecureSkipVerify: true}, Addr: "127.0.0.1", Port: ports.KubeControllerManagerPort, Path: "/healthz"},
		"scheduler":          &componentstatus.HttpServer{EnableHTTPS: true, TLSConfig: &tls.Config{InsecureSkipVerify: true}, Addr: "127.0.0.1", Port: kubeschedulerconfig.DefaultKubeSchedulerPort, Path: "/healthz"},
	}

	for ix, cfg := range s.storageFactory.Configs() {
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = &componentstatus.EtcdServer{Config: cfg}
	}
	return serversToValidate
}
