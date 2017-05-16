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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	namespacecontroller "k8s.io/kubernetes/pkg/controller/namespace"
	"k8s.io/kubernetes/pkg/controller/podgc"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	ttlcontroller "k8s.io/kubernetes/pkg/controller/ttl"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
)

func startEndpointController(ctx ControllerContext) (bool, error) {
	go endpointcontroller.NewEndpointController(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().Services(),
		ctx.ClientBuilder.ClientOrDie("endpoint-controller"),
	).Run(int(ctx.Options.ConcurrentEndpointSyncs), ctx.Stop)
	return true, nil
}

func startReplicationController(ctx ControllerContext) (bool, error) {
	go replicationcontroller.NewReplicationManager(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().ReplicationControllers(),
		ctx.ClientBuilder.ClientOrDie("replication-controller"),
		replicationcontroller.BurstReplicas,
	).Run(int(ctx.Options.ConcurrentRCSyncs), ctx.Stop)
	return true, nil
}

func startPodGCController(ctx ControllerContext) (bool, error) {
	go podgc.NewPodGC(
		ctx.ClientBuilder.ClientOrDie("pod-garbage-collector"),
		ctx.InformerFactory.Core().V1().Pods(),
		int(ctx.Options.TerminatedPodGCThreshold),
	).Run(ctx.Stop)
	return true, nil
}

func startResourceQuotaController(ctx ControllerContext) (bool, error) {
	resourceQuotaControllerClient := ctx.ClientBuilder.ClientOrDie("resourcequota-controller")
	resourceQuotaRegistry := quotainstall.NewRegistry(resourceQuotaControllerClient, ctx.InformerFactory)
	groupKindsToReplenish := []schema.GroupKind{
		api.Kind("Pod"),
		api.Kind("Service"),
		api.Kind("ReplicationController"),
		api.Kind("PersistentVolumeClaim"),
		api.Kind("Secret"),
		api.Kind("ConfigMap"),
	}
	resourceQuotaControllerOptions := &resourcequotacontroller.ResourceQuotaControllerOptions{
		KubeClient:                resourceQuotaControllerClient,
		ResourceQuotaInformer:     ctx.InformerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.StaticResyncPeriodFunc(ctx.Options.ResourceQuotaSyncPeriod.Duration),
		Registry:                  resourceQuotaRegistry,
		ControllerFactory:         resourcequotacontroller.NewReplenishmentControllerFactory(ctx.InformerFactory),
		ReplenishmentResyncPeriod: ResyncPeriod(&ctx.Options),
		GroupKindsToReplenish:     groupKindsToReplenish,
	}
	go resourcequotacontroller.NewResourceQuotaController(
		resourceQuotaControllerOptions,
	).Run(int(ctx.Options.ConcurrentResourceQuotaSyncs), ctx.Stop)
	return true, nil
}

func startNamespaceController(ctx ControllerContext) (bool, error) {
	// TODO: should use a dynamic RESTMapper built from the discovery results.
	restMapper := api.Registry.RESTMapper()

	// the namespace cleanup controller is very chatty.  It makes lots of discovery calls and then it makes lots of delete calls
	// the ratelimiter negatively affects its speed.  Deleting 100 total items in a namespace (that's only a few of each resource
	// including events), takes ~10 seconds by default.
	nsKubeconfig := ctx.ClientBuilder.ConfigOrDie("namespace-controller")
	nsKubeconfig.QPS *= 10
	nsKubeconfig.Burst *= 10
	namespaceKubeClient := clientset.NewForConfigOrDie(nsKubeconfig)
	namespaceClientPool := dynamic.NewClientPool(nsKubeconfig, restMapper, dynamic.LegacyAPIPathResolverFunc)

	// Find the list of namespaced resources via discovery that the namespace controller must manage
	// TODO: consider using a list-watch + cache here rather than polling
	resources, err := namespaceKubeClient.Discovery().ServerResources()
	if err != nil {
		return true, fmt.Errorf("failed to get preferred server resources: %v", err)
	}
	gvrs, err := discovery.GroupVersionResources(resources)
	if err != nil {
		return true, fmt.Errorf("failed to parse preferred server resources: %v", err)
	}
	discoverResourcesFn := namespaceKubeClient.Discovery().ServerPreferredNamespacedResources
	if _, found := gvrs[extensions.SchemeGroupVersion.WithResource("thirdpartyresource")]; !found {
		// make discovery static
		snapshot, err := discoverResourcesFn()
		if err != nil {
			return true, fmt.Errorf("failed to get server resources: %v", err)
		}
		discoverResourcesFn = func() ([]*metav1.APIResourceList, error) {
			return snapshot, nil
		}
	}
	namespaceController := namespacecontroller.NewNamespaceController(
		namespaceKubeClient,
		namespaceClientPool,
		discoverResourcesFn,
		ctx.InformerFactory.Core().V1().Namespaces(),
		ctx.Options.NamespaceSyncPeriod.Duration,
		v1.FinalizerKubernetes,
	)
	go namespaceController.Run(int(ctx.Options.ConcurrentNamespaceSyncs), ctx.Stop)

	return true, nil

}

func startServiceAccountController(ctx ControllerContext) (bool, error) {
	go serviceaccountcontroller.NewServiceAccountsController(
		ctx.InformerFactory.Core().V1().ServiceAccounts(),
		ctx.InformerFactory.Core().V1().Namespaces(),
		ctx.ClientBuilder.ClientOrDie("service-account-controller"),
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	).Run(1, ctx.Stop)
	return true, nil
}

func startTTLController(ctx ControllerContext) (bool, error) {
	go ttlcontroller.NewTTLController(
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.ClientBuilder.ClientOrDie("ttl-controller"),
	).Run(5, ctx.Stop)
	return true, nil
}

func startGarbageCollectorController(ctx ControllerContext) (bool, error) {
	if !ctx.Options.EnableGarbageCollector {
		return false, nil
	}

	// TODO: should use a dynamic RESTMapper built from the discovery results.
	restMapper := api.Registry.RESTMapper()

	gcClientset := ctx.ClientBuilder.ClientOrDie("generic-garbage-collector")
	preferredResources, err := gcClientset.Discovery().ServerPreferredResources()
	if err != nil {
		return true, fmt.Errorf("failed to get supported resources from server: %v", err)
	}
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"get", "list", "watch", "patch", "update", "delete"}}, preferredResources)
	deletableGroupVersionResources, err := discovery.GroupVersionResources(deletableResources)
	if err != nil {
		return true, fmt.Errorf("Failed to parse resources from server: %v", err)
	}

	config := ctx.ClientBuilder.ConfigOrDie("generic-garbage-collector")
	config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
	metaOnlyClientPool := dynamic.NewClientPool(config, restMapper, dynamic.LegacyAPIPathResolverFunc)
	config.ContentConfig = dynamic.ContentConfig()
	clientPool := dynamic.NewClientPool(config, restMapper, dynamic.LegacyAPIPathResolverFunc)

	ignoredResources := make(map[schema.GroupResource]struct{})
	for _, r := range ctx.Options.GCIgnoredResources {
		ignoredResources[schema.GroupResource{Group: r.Group, Resource: r.Resource}] = struct{}{}
	}

	garbageCollector, err := garbagecollector.NewGarbageCollector(
		metaOnlyClientPool,
		clientPool,
		restMapper,
		deletableGroupVersionResources,
		ignoredResources,
		ctx.InformerFactory,
	)
	if err != nil {
		return true, fmt.Errorf("Failed to start the generic garbage collector: %v", err)
	}
	workers := int(ctx.Options.ConcurrentGCSyncs)
	go garbageCollector.Run(workers, ctx.Stop)

	return true, nil
}
