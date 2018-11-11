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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
//
package app

import (
	"fmt"
	"net"
	"strings"
	"time"

	"k8s.io/klog"

	"net/http"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	csiclientset "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/kubernetes/pkg/controller"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	nodeipamcontroller "k8s.io/kubernetes/pkg/controller/nodeipam"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	lifecyclecontroller "k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/controller/podgc"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	ttlcontroller "k8s.io/kubernetes/pkg/controller/ttl"
	"k8s.io/kubernetes/pkg/controller/ttlafterfinished"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	"k8s.io/kubernetes/pkg/controller/volume/expand"
	persistentvolumecontroller "k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	"k8s.io/kubernetes/pkg/controller/volume/pvcprotection"
	"k8s.io/kubernetes/pkg/controller/volume/pvprotection"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/quota/v1/generic"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
	"k8s.io/kubernetes/pkg/util/metrics"
)

func startServiceController(ctx ControllerContext) (http.Handler, bool, error) {
	serviceController, err := servicecontroller.New(
		ctx.Cloud,
		ctx.ClientBuilder.ClientOrDie("service-controller"),
		ctx.InformerFactory.Core().V1().Services(),
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.ComponentConfig.KubeCloudShared.ClusterName,
	)
	if err != nil {
		// This error shouldn't fail. It lives like this as a legacy.
		klog.Errorf("Failed to start service controller: %v", err)
		return nil, false, nil
	}
	go serviceController.Run(ctx.Stop, int(ctx.ComponentConfig.ServiceController.ConcurrentServiceSyncs))
	return nil, true, nil
}

func startNodeIpamController(ctx ControllerContext) (http.Handler, bool, error) {
	var clusterCIDR *net.IPNet = nil
	var serviceCIDR *net.IPNet = nil

	if !ctx.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs {
		return nil, false, nil
	}

	var err error
	if len(strings.TrimSpace(ctx.ComponentConfig.KubeCloudShared.ClusterCIDR)) != 0 {
		_, clusterCIDR, err = net.ParseCIDR(ctx.ComponentConfig.KubeCloudShared.ClusterCIDR)
		if err != nil {
			klog.Warningf("Unsuccessful parsing of cluster CIDR %v: %v", ctx.ComponentConfig.KubeCloudShared.ClusterCIDR, err)
		}
	}

	if len(strings.TrimSpace(ctx.ComponentConfig.NodeIPAMController.ServiceCIDR)) != 0 {
		_, serviceCIDR, err = net.ParseCIDR(ctx.ComponentConfig.NodeIPAMController.ServiceCIDR)
		if err != nil {
			klog.Warningf("Unsuccessful parsing of service CIDR %v: %v", ctx.ComponentConfig.NodeIPAMController.ServiceCIDR, err)
		}
	}

	nodeIpamController, err := nodeipamcontroller.NewNodeIpamController(
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.Cloud,
		ctx.ClientBuilder.ClientOrDie("node-controller"),
		clusterCIDR,
		serviceCIDR,
		int(ctx.ComponentConfig.NodeIPAMController.NodeCIDRMaskSize),
		ipam.CIDRAllocatorType(ctx.ComponentConfig.KubeCloudShared.CIDRAllocatorType),
	)
	if err != nil {
		return nil, true, err
	}
	go nodeIpamController.Run(ctx.Stop)
	return nil, true, nil
}

func startNodeLifecycleController(ctx ControllerContext) (http.Handler, bool, error) {
	lifecycleController, err := lifecyclecontroller.NewNodeLifecycleController(
		ctx.InformerFactory.Coordination().V1beta1().Leases(),
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.InformerFactory.Extensions().V1beta1().DaemonSets(),
		ctx.Cloud,
		ctx.ClientBuilder.ClientOrDie("node-controller"),
		ctx.ComponentConfig.KubeCloudShared.NodeMonitorPeriod.Duration,
		ctx.ComponentConfig.NodeLifecycleController.NodeStartupGracePeriod.Duration,
		ctx.ComponentConfig.NodeLifecycleController.NodeMonitorGracePeriod.Duration,
		ctx.ComponentConfig.NodeLifecycleController.PodEvictionTimeout.Duration,
		ctx.ComponentConfig.NodeLifecycleController.NodeEvictionRate,
		ctx.ComponentConfig.NodeLifecycleController.SecondaryNodeEvictionRate,
		ctx.ComponentConfig.NodeLifecycleController.LargeClusterSizeThreshold,
		ctx.ComponentConfig.NodeLifecycleController.UnhealthyZoneThreshold,
		ctx.ComponentConfig.NodeLifecycleController.EnableTaintManager,
		utilfeature.DefaultFeatureGate.Enabled(features.TaintBasedEvictions),
		utilfeature.DefaultFeatureGate.Enabled(features.TaintNodesByCondition),
	)
	if err != nil {
		return nil, true, err
	}
	go lifecycleController.Run(ctx.Stop)
	return nil, true, nil
}

func startRouteController(ctx ControllerContext) (http.Handler, bool, error) {
	if !ctx.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs || !ctx.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes {
		klog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", ctx.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs, ctx.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes)
		return nil, false, nil
	}
	if ctx.Cloud == nil {
		klog.Warning("configure-cloud-routes is set, but no cloud provider specified. Will not configure cloud provider routes.")
		return nil, false, nil
	}
	routes, ok := ctx.Cloud.Routes()
	if !ok {
		klog.Warning("configure-cloud-routes is set, but cloud provider does not support routes. Will not configure cloud provider routes.")
		return nil, false, nil
	}
	_, clusterCIDR, err := net.ParseCIDR(ctx.ComponentConfig.KubeCloudShared.ClusterCIDR)
	if err != nil {
		klog.Warningf("Unsuccessful parsing of cluster CIDR %v: %v", ctx.ComponentConfig.KubeCloudShared.ClusterCIDR, err)
	}
	routeController := routecontroller.New(routes, ctx.ClientBuilder.ClientOrDie("route-controller"), ctx.InformerFactory.Core().V1().Nodes(), ctx.ComponentConfig.KubeCloudShared.ClusterName, clusterCIDR)
	go routeController.Run(ctx.Stop, ctx.ComponentConfig.KubeCloudShared.RouteReconciliationPeriod.Duration)
	return nil, true, nil
}

func startPersistentVolumeBinderController(ctx ControllerContext) (http.Handler, bool, error) {
	params := persistentvolumecontroller.ControllerParameters{
		KubeClient:                ctx.ClientBuilder.ClientOrDie("persistent-volume-binder"),
		SyncPeriod:                ctx.ComponentConfig.PersistentVolumeBinderController.PVClaimBinderSyncPeriod.Duration,
		VolumePlugins:             ProbeControllerVolumePlugins(ctx.Cloud, ctx.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration),
		Cloud:                     ctx.Cloud,
		ClusterName:               ctx.ComponentConfig.KubeCloudShared.ClusterName,
		VolumeInformer:            ctx.InformerFactory.Core().V1().PersistentVolumes(),
		ClaimInformer:             ctx.InformerFactory.Core().V1().PersistentVolumeClaims(),
		ClassInformer:             ctx.InformerFactory.Storage().V1().StorageClasses(),
		PodInformer:               ctx.InformerFactory.Core().V1().Pods(),
		NodeInformer:              ctx.InformerFactory.Core().V1().Nodes(),
		EnableDynamicProvisioning: ctx.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration.EnableDynamicProvisioning,
	}
	volumeController, volumeControllerErr := persistentvolumecontroller.NewController(params)
	if volumeControllerErr != nil {
		return nil, true, fmt.Errorf("failed to construct persistentvolume controller: %v", volumeControllerErr)
	}
	go volumeController.Run(ctx.Stop)
	return nil, true, nil
}

func startAttachDetachController(ctx ControllerContext) (http.Handler, bool, error) {
	if ctx.ComponentConfig.AttachDetachController.ReconcilerSyncLoopPeriod.Duration < time.Second {
		return nil, true, fmt.Errorf("Duration time must be greater than one second as set via command line option reconcile-sync-loop-period.")
	}
	csiClientConfig := ctx.ClientBuilder.ConfigOrDie("attachdetach-controller")
	// csiClient works with CRDs that support json only
	csiClientConfig.ContentType = "application/json"

	attachDetachController, attachDetachControllerErr :=
		attachdetach.NewAttachDetachController(
			ctx.ClientBuilder.ClientOrDie("attachdetach-controller"),
			csiclientset.NewForConfigOrDie(csiClientConfig),
			ctx.InformerFactory.Core().V1().Pods(),
			ctx.InformerFactory.Core().V1().Nodes(),
			ctx.InformerFactory.Core().V1().PersistentVolumeClaims(),
			ctx.InformerFactory.Core().V1().PersistentVolumes(),
			ctx.Cloud,
			ProbeAttachableVolumePlugins(),
			GetDynamicPluginProber(ctx.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration),
			ctx.ComponentConfig.AttachDetachController.DisableAttachDetachReconcilerSync,
			ctx.ComponentConfig.AttachDetachController.ReconcilerSyncLoopPeriod.Duration,
			attachdetach.DefaultTimerConfig,
		)
	if attachDetachControllerErr != nil {
		return nil, true, fmt.Errorf("failed to start attach/detach controller: %v", attachDetachControllerErr)
	}
	go attachDetachController.Run(ctx.Stop)
	return nil, true, nil
}

func startVolumeExpandController(ctx ControllerContext) (http.Handler, bool, error) {
	if utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		expandController, expandControllerErr := expand.NewExpandController(
			ctx.ClientBuilder.ClientOrDie("expand-controller"),
			ctx.InformerFactory.Core().V1().PersistentVolumeClaims(),
			ctx.InformerFactory.Core().V1().PersistentVolumes(),
			ctx.Cloud,
			ProbeExpandableVolumePlugins(ctx.ComponentConfig.PersistentVolumeBinderController.VolumeConfiguration))

		if expandControllerErr != nil {
			return nil, true, fmt.Errorf("Failed to start volume expand controller : %v", expandControllerErr)
		}
		go expandController.Run(ctx.Stop)
		return nil, true, nil
	}
	return nil, false, nil
}

func startEndpointController(ctx ControllerContext) (http.Handler, bool, error) {
	go endpointcontroller.NewEndpointController(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().Services(),
		ctx.InformerFactory.Core().V1().Endpoints(),
		ctx.ClientBuilder.ClientOrDie("endpoint-controller"),
	).Run(int(ctx.ComponentConfig.EndpointController.ConcurrentEndpointSyncs), ctx.Stop)
	return nil, true, nil
}

func startReplicationController(ctx ControllerContext) (http.Handler, bool, error) {
	go replicationcontroller.NewReplicationManager(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().ReplicationControllers(),
		ctx.ClientBuilder.ClientOrDie("replication-controller"),
		replicationcontroller.BurstReplicas,
	).Run(int(ctx.ComponentConfig.ReplicationController.ConcurrentRCSyncs), ctx.Stop)
	return nil, true, nil
}

func startPodGCController(ctx ControllerContext) (http.Handler, bool, error) {
	go podgc.NewPodGC(
		ctx.ClientBuilder.ClientOrDie("pod-garbage-collector"),
		ctx.InformerFactory.Core().V1().Pods(),
		int(ctx.ComponentConfig.PodGCController.TerminatedPodGCThreshold),
	).Run(ctx.Stop)
	return nil, true, nil
}

func startResourceQuotaController(ctx ControllerContext) (http.Handler, bool, error) {
	resourceQuotaControllerClient := ctx.ClientBuilder.ClientOrDie("resourcequota-controller")
	discoveryFunc := resourceQuotaControllerClient.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(ctx.InformerFactory.ForResource)
	quotaConfiguration := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)

	resourceQuotaControllerOptions := &resourcequotacontroller.ResourceQuotaControllerOptions{
		QuotaClient:               resourceQuotaControllerClient.CoreV1(),
		ResourceQuotaInformer:     ctx.InformerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.StaticResyncPeriodFunc(ctx.ComponentConfig.ResourceQuotaController.ResourceQuotaSyncPeriod.Duration),
		InformerFactory:           ctx.InformerFactory,
		ReplenishmentResyncPeriod: ctx.ResyncPeriod,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		InformersStarted:          ctx.InformersStarted,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
	}
	if resourceQuotaControllerClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := metrics.RegisterMetricAndTrackRateLimiterUsage("resource_quota_controller", resourceQuotaControllerClient.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, true, err
		}
	}

	resourceQuotaController, err := resourcequotacontroller.NewResourceQuotaController(resourceQuotaControllerOptions)
	if err != nil {
		return nil, false, err
	}
	go resourceQuotaController.Run(int(ctx.ComponentConfig.ResourceQuotaController.ConcurrentResourceQuotaSyncs), ctx.Stop)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(discoveryFunc, 30*time.Second, ctx.Stop)

	return nil, true, nil
}

func startNamespaceController(ctx ControllerContext) (http.Handler, bool, error) {
	// the namespace cleanup controller is very chatty.  It makes lots of discovery calls and then it makes lots of delete calls
	// the ratelimiter negatively affects its speed.  Deleting 100 total items in a namespace (that's only a few of each resource
	// including events), takes ~10 seconds by default.
	nsKubeconfig := ctx.ClientBuilder.ConfigOrDie("namespace-controller")
	nsKubeconfig.QPS *= 20
	nsKubeconfig.Burst *= 100
	namespaceKubeClient := clientset.NewForConfigOrDie(nsKubeconfig)

	dynamicClient, err := dynamic.NewForConfig(nsKubeconfig)
	if err != nil {
		return nil, true, err
	}

	discoverResourcesFn := namespaceKubeClient.Discovery().ServerPreferredNamespacedResources

	namespaceController := namespacecontroller.NewNamespaceController(
		namespaceKubeClient,
		dynamicClient,
		discoverResourcesFn,
		ctx.InformerFactory.Core().V1().Namespaces(),
		ctx.ComponentConfig.NamespaceController.NamespaceSyncPeriod.Duration,
		v1.FinalizerKubernetes,
	)
	go namespaceController.Run(int(ctx.ComponentConfig.NamespaceController.ConcurrentNamespaceSyncs), ctx.Stop)

	return nil, true, nil
}

func startServiceAccountController(ctx ControllerContext) (http.Handler, bool, error) {
	sac, err := serviceaccountcontroller.NewServiceAccountsController(
		ctx.InformerFactory.Core().V1().ServiceAccounts(),
		ctx.InformerFactory.Core().V1().Namespaces(),
		ctx.ClientBuilder.ClientOrDie("service-account-controller"),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating ServiceAccount controller: %v", err)
	}
	go sac.Run(1, ctx.Stop)
	return nil, true, nil
}

func startTTLController(ctx ControllerContext) (http.Handler, bool, error) {
	go ttlcontroller.NewTTLController(
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.ClientBuilder.ClientOrDie("ttl-controller"),
	).Run(5, ctx.Stop)
	return nil, true, nil
}

func startGarbageCollectorController(ctx ControllerContext) (http.Handler, bool, error) {
	if !ctx.ComponentConfig.GarbageCollectorController.EnableGarbageCollector {
		return nil, false, nil
	}

	gcClientset := ctx.ClientBuilder.ClientOrDie("generic-garbage-collector")
	discoveryClient := cacheddiscovery.NewMemCacheClient(gcClientset.Discovery())

	config := ctx.ClientBuilder.ConfigOrDie("generic-garbage-collector")
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, true, err
	}

	// Get an initial set of deletable resources to prime the garbage collector.
	deletableResources := garbagecollector.GetDeletableResources(discoveryClient)
	ignoredResources := make(map[schema.GroupResource]struct{})
	for _, r := range ctx.ComponentConfig.GarbageCollectorController.GCIgnoredResources {
		ignoredResources[schema.GroupResource{Group: r.Group, Resource: r.Resource}] = struct{}{}
	}
	garbageCollector, err := garbagecollector.NewGarbageCollector(
		dynamicClient,
		ctx.RESTMapper,
		deletableResources,
		ignoredResources,
		ctx.InformerFactory,
		ctx.InformersStarted,
	)
	if err != nil {
		return nil, true, fmt.Errorf("Failed to start the generic garbage collector: %v", err)
	}

	// Start the garbage collector.
	workers := int(ctx.ComponentConfig.GarbageCollectorController.ConcurrentGCSyncs)
	go garbageCollector.Run(workers, ctx.Stop)

	// Periodically refresh the RESTMapper with new discovery information and sync
	// the garbage collector.
	go garbageCollector.Sync(gcClientset.Discovery(), 30*time.Second, ctx.Stop)

	return garbagecollector.NewDebugHandler(garbageCollector), true, nil
}

func startPVCProtectionController(ctx ControllerContext) (http.Handler, bool, error) {
	go pvcprotection.NewPVCProtectionController(
		ctx.InformerFactory.Core().V1().PersistentVolumeClaims(),
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.ClientBuilder.ClientOrDie("pvc-protection-controller"),
		utilfeature.DefaultFeatureGate.Enabled(features.StorageObjectInUseProtection),
	).Run(1, ctx.Stop)
	return nil, true, nil
}

func startPVProtectionController(ctx ControllerContext) (http.Handler, bool, error) {
	go pvprotection.NewPVProtectionController(
		ctx.InformerFactory.Core().V1().PersistentVolumes(),
		ctx.ClientBuilder.ClientOrDie("pv-protection-controller"),
		utilfeature.DefaultFeatureGate.Enabled(features.StorageObjectInUseProtection),
	).Run(1, ctx.Stop)
	return nil, true, nil
}

func startTTLAfterFinishedController(ctx ControllerContext) (http.Handler, bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.TTLAfterFinished) {
		return nil, false, nil
	}
	go ttlafterfinished.New(
		ctx.InformerFactory.Batch().V1().Jobs(),
		ctx.ClientBuilder.ClientOrDie("ttl-after-finished-controller"),
	).Run(int(ctx.ComponentConfig.TTLAfterFinishedController.ConcurrentTTLSyncs), ctx.Stop)
	return nil, true, nil
}
