/*
Copyright 2025 The Kubernetes Authors.

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

package controller

import (
	"context"
	"fmt"
	"math/rand/v2"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	controllersmetrics "k8s.io/component-base/metrics/prometheus/controllers"
	genericcontrollermanager "k8s.io/controller-manager/app"
	"k8s.io/controller-manager/pkg/clientbuilder"
	"k8s.io/controller-manager/pkg/informerfactory"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
)

const (
	// kubeControllerManager defines variable used internally when referring to cloud-controller-manager component
	kubeControllerManager = "kube-controller-manager"
)

// ResyncPeriod returns a function which generates a duration each time it is
// invoked; this is because that multiple controllers don't get into lock-step.
func ResyncPeriod(c *config.CompletedConfig) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.Generic.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Context defines the context object for controller
type Context struct {
	// ClientBuilder will provide a client for this controller to use
	ClientBuilder clientbuilder.ControllerClientBuilder

	// InformerFactory gives access to informers for the controller.
	InformerFactory informers.SharedInformerFactory

	// ObjectOrMetadataInformerFactory gives access to informers for typed resources
	// and dynamic resources by their metadata. All generic controllers currently use
	// object metadata - if a future controller needs access to the full object this
	// would become GenericInformerFactory and take a dynamic client.
	ObjectOrMetadataInformerFactory informerfactory.InformerFactory

	// ComponentConfig provides access to init options for a given controller
	ComponentConfig kubectrlmgrconfig.KubeControllerManagerConfiguration

	// DeferredDiscoveryRESTMapper is a RESTMapper that will defer
	// initialization of the RESTMapper until the first mapping is
	// requested.
	RESTMapper *restmapper.DeferredDiscoveryRESTMapper

	// InformersStarted is closed after all of the controllers have been initialized and are running.  After this point it is safe,
	// for an individual controller to start the shared informers. Before it is closed, they should not.
	InformersStarted chan struct{}

	// ResyncPeriod generates a duration each time it is invoked; this is so that
	// multiple controllers don't get into lock-step and all hammer the apiserver
	// with list requests simultaneously.
	ResyncPeriod func() time.Duration

	// ControllerManagerMetrics provides a proxy to set controller manager specific metrics.
	ControllerManagerMetrics *controllersmetrics.ControllerManagerMetrics

	// GraphBuilder gives an access to dependencyGraphBuilder which keeps tracks of resources in the cluster
	GraphBuilder *garbagecollector.GraphBuilder
}

// IsControllerEnabled checks if the context's controllers enabled or not
func (c Context) IsControllerEnabled(controllerDescriptor *Descriptor) bool {
	controllersDisabledByDefault := sets.NewString()
	if controllerDescriptor.IsDisabledByDefault {
		controllersDisabledByDefault.Insert(controllerDescriptor.Name)
	}
	return genericcontrollermanager.IsControllerEnabled(controllerDescriptor.Name, controllersDisabledByDefault, c.ComponentConfig.Generic.Controllers)
}

// IsControllerEnabledByName checks whether the given controller is enabled,
// but it doesn't consult the descriptor, so it cannot check whether the controller is not disabled by default.
func (c Context) IsControllerEnabledByName(controllerName string) bool {
	return genericcontrollermanager.IsControllerEnabled(controllerName, sets.NewString(), c.ComponentConfig.Generic.Controllers)
}

// NewClientConfig is a shortcut for ClientBuilder.Config. It wraps the error with an additional message.
func (c Context) NewClientConfig(name string) (*restclient.Config, error) {
	config, err := c.ClientBuilder.Config(name)
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client config for %q: %w", name, err)
	}
	return config, nil
}

// NewClient is a shortcut for ClientBuilder.Client. It wraps the error with an additional message.
func (c Context) NewClient(name string) (kubernetes.Interface, error) {
	client, err := c.ClientBuilder.Client(name)
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client for %q: %w", name, err)
	}
	return client, nil
}

// CreateControllerContext creates a context struct containing references to resources needed by the
// controllers such as the cloud provider and clientBuilder. rootClientBuilder is only used for
// the shared-informers client and token controller.
func CreateControllerContext(ctx context.Context, s *config.CompletedConfig, rootClientBuilder, clientBuilder clientbuilder.ControllerClientBuilder) (Context, error) {
	// Informer transform to trim ManagedFields for memory efficiency.
	trim := func(obj interface{}) (interface{}, error) {
		if accessor, err := meta.Accessor(obj); err == nil {
			if accessor.GetManagedFields() != nil {
				accessor.SetManagedFields(nil)
			}
		}
		return obj, nil
	}

	versionedClient, err := rootClientBuilder.Client("shared-informers")
	if err != nil {
		return Context{}, fmt.Errorf("failed to create Kubernetes client for %q: %w", "shared-informers", err)
	}

	sharedInformers := informers.NewSharedInformerFactoryWithOptions(versionedClient, ResyncPeriod(s)(), informers.WithTransform(trim))

	metadataConfig, err := rootClientBuilder.Config("metadata-informers")
	if err != nil {
		return Context{}, fmt.Errorf("failed to create metadata client config: %w", err)
	}

	metadataClient, err := metadata.NewForConfig(metadataConfig)
	if err != nil {
		return Context{}, fmt.Errorf("failed to create metadata client: %w", err)
	}

	metadataInformers := metadatainformer.NewSharedInformerFactoryWithOptions(metadataClient, ResyncPeriod(s)(), metadatainformer.WithTransform(trim))

	// If apiserver is not running we should wait for some time and fail only then. This is particularly
	// important when we start apiserver and controller manager at the same time.
	if err := genericcontrollermanager.WaitForAPIServer(versionedClient, 10*time.Second); err != nil {
		return Context{}, fmt.Errorf("failed to wait for apiserver being healthy: %w", err)
	}

	// Use a discovery client capable of being refreshed.
	discoveryClient, err := rootClientBuilder.DiscoveryClient("controller-discovery")
	if err != nil {
		return Context{}, fmt.Errorf("failed to create discovery client: %w", err)
	}

	cachedClient := cacheddiscovery.NewMemCacheClient(discoveryClient)
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedClient)
	go wait.Until(func() {
		restMapper.Reset()
	}, 30*time.Second, ctx.Done())

	controllerContext := Context{
		ClientBuilder:                   clientBuilder,
		InformerFactory:                 sharedInformers,
		ObjectOrMetadataInformerFactory: informerfactory.NewInformerFactory(sharedInformers, metadataInformers),
		ComponentConfig:                 s.ComponentConfig,
		RESTMapper:                      restMapper,
		InformersStarted:                make(chan struct{}),
		ResyncPeriod:                    ResyncPeriod(s),
		ControllerManagerMetrics:        controllersmetrics.NewControllerManagerMetrics(kubeControllerManager),
	}

	if controllerContext.ComponentConfig.GarbageCollectorController.EnableGarbageCollector &&
		controllerContext.IsControllerEnabledByName(names.GarbageCollectorController) {
		ignoredResources := make(map[schema.GroupResource]struct{})
		for _, r := range controllerContext.ComponentConfig.GarbageCollectorController.GCIgnoredResources {
			ignoredResources[schema.GroupResource{Group: r.Group, Resource: r.Resource}] = struct{}{}
		}

		controllerContext.GraphBuilder = garbagecollector.NewDependencyGraphBuilder(
			ctx,
			metadataClient,
			controllerContext.RESTMapper,
			ignoredResources,
			controllerContext.ObjectOrMetadataInformerFactory,
			controllerContext.InformersStarted,
		)
	}

	controllersmetrics.Register()
	return controllerContext, nil
}
