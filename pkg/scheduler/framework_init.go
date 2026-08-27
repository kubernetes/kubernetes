/*
Copyright The Kubernetes Authors.

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

package scheduler

import (
	"context"
	"errors"
	"fmt"
	"time"

	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	frameworkplugins "k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodevolumelimits"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// FrameworkComponents holds the shared scheduler dependencies that span across
// all framework profiles, the scheduling queue, and event handlers.
//
// It decouples the initialization of cluster-state caches and DRA/extender
// managers from the per-profile framework wiring, allowing synthetic callers
// and simulation harnesses to construct profiles without redundant shared state.
type FrameworkComponents struct {
	// cache stores cluster state shared across framework profiles, scheduling queue,
	// and event handlers.
	cache internalcache.Cache

	extenders []fwk.Extender

	// APIDispatcher is non-nil when the SchedulerAsyncAPICalls feature gate is enabled.
	apiDispatcher   *apidispatcher.APIDispatcher
	metricsRecorder *metrics.MetricAsyncRecorder

	// DRA components required for registering event handlers. Nil unless
	// DynamicResourceAllocation feature gate is enabled.
	resourceClaimCache   *assumecache.AssumeCache
	resourceSliceTracker *resourceslicetracker.Tracker
	draManager           fwk.SharedDRAManager

	client          clientset.Interface
	informerFactory informers.SharedInformerFactory
	options         schedulerOptions
}

var defaultComponentsOptions = schedulerOptions{
	parallelism:         defaultSchedulerOptions.parallelism,
	applyDefaultProfile: defaultSchedulerOptions.applyDefaultProfile,
}

// NewFrameworkComponents initializes the shared dependencies required across
// scheduler framework profiles.
//
// Callers must invoke NewFrameworkComponents before starting the shared
// informer factory so that any required informers are registered prior to Start().
func NewFrameworkComponents(ctx context.Context, client clientset.Interface, informerFactory informers.SharedInformerFactory, opts ...Option) (*FrameworkComponents, error) {
	options := defaultComponentsOptions
	for _, opt := range opts {
		opt(&options)
	}

	return newFrameworkComponents(ctx, client, informerFactory, options)
}

func newFrameworkComponents(ctx context.Context,
	client clientset.Interface,
	informerFactory informers.SharedInformerFactory,
	options schedulerOptions,
) (*FrameworkComponents, error) {
	logger := klog.FromContext(ctx)
	stopEverything := ctx.Done()

	if options.applyDefaultProfile {
		var versionedCfg configv1.KubeSchedulerConfiguration
		scheme.Scheme.Default(&versionedCfg)
		cfg := schedulerapi.KubeSchedulerConfiguration{}
		if err := scheme.Scheme.Convert(&versionedCfg, &cfg, nil); err != nil {
			return nil, err
		}
		options.profiles = cfg.Profiles
	}

	extenders, err := buildExtenders(logger, options.extenders, options.profiles)
	if err != nil {
		return nil, fmt.Errorf("couldn't build extenders: %w", err)
	}

	metricsRecorder := metrics.NewMetricsAsyncRecorder(1000, time.Second, stopEverything)

	var resourceClaimCache *assumecache.AssumeCache
	var resourceSliceTracker *resourceslicetracker.Tracker
	var draManager fwk.SharedDRAManager
	if feature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		resourceClaimInformer := informerFactory.Resource().V1().ResourceClaims().Informer()
		resourceClaimCache = assumecache.NewAssumeCache(logger, resourceClaimInformer, "ResourceClaim", "", nil)
		resourceSliceTrackerOpts := resourceslicetracker.Options{
			EnableDeviceTaintRules:   feature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules),
			EnableConsumableCapacity: feature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity),
			SliceInformer:            informerFactory.Resource().V1().ResourceSlices(),
			KubeClient:               client,
		}
		// If device taint rules are disabled, the additional informers are not needed and
		// the tracker turns into a simple wrapper around the slice informer.
		if resourceSliceTrackerOpts.EnableDeviceTaintRules {
			resourceSliceTrackerOpts.TaintInformer = informerFactory.Resource().V1().DeviceTaintRules()
		}
		resourceSliceTracker, err = resourceslicetracker.StartTracker(ctx, resourceSliceTrackerOpts)
		if err != nil {
			return nil, fmt.Errorf("couldn't start resource slice tracker: %w", err)
		}
		draManager = dynamicresources.NewDRAManager(ctx, resourceClaimCache, resourceSliceTracker, informerFactory)
	}

	var apiDispatcher *apidispatcher.APIDispatcher
	if feature.DefaultFeatureGate.Enabled(features.SchedulerAsyncAPICalls) {
		apiDispatcher = apidispatcher.New(client, int(options.parallelism), apicalls.Relevances)
	}

	schedulerCache := internalcache.New(ctx, apiDispatcher, feature.DefaultFeatureGate.Enabled(features.GenericWorkload), feature.DefaultFeatureGate.Enabled(features.CompositePodGroup))

	return &FrameworkComponents{
		cache: schedulerCache,

		extenders:            extenders,
		apiDispatcher:        apiDispatcher,
		metricsRecorder:      metricsRecorder,
		resourceClaimCache:   resourceClaimCache,
		resourceSliceTracker: resourceSliceTracker,
		draManager:           draManager,
		client:               client,
		informerFactory:      informerFactory,
		options:              options,
	}, nil
}

// GetCache returns the shared scheduler cache instance.
func (c *FrameworkComponents) GetCache() internalcache.Cache {
	return c.cache
}

// NewFrameworkMap builds the map of scheduling framework profiles from the
// resolved configurations and shared components.
//
// It must run before the informer factory is started because it instantiates
// informers (such as CSINodes) that Start() would otherwise never run.
//
// snapshot is injected into every framework profile as its SharedLister;
// Scheduler passes the scheduler's internal cache snapshot, while library
// consumers may inject custom snapshots for testing or simulation.
func NewFrameworkMap(ctx context.Context, c *FrameworkComponents, recorderFactory profile.RecorderFactory, snapshot *internalcache.Snapshot) (profile.Map, error) {
	registry := frameworkplugins.NewInTreeRegistry()
	if err := registry.Merge(c.options.frameworkOutOfTreeRegistry); err != nil {
		return nil, err
	}
	csiManager := nodevolumelimits.NewCSIManager(
		c.informerFactory.Storage().V1().CSINodes().Lister())

	profiles, err := profile.NewMap(ctx, c.options.profiles, registry, recorderFactory,
		frameworkruntime.WithComponentConfigVersion(c.options.componentConfigVersion),
		frameworkruntime.WithClientSet(c.client),
		frameworkruntime.WithKubeConfig(c.options.kubeConfig),
		frameworkruntime.WithInformerFactory(c.informerFactory),
		frameworkruntime.WithSharedDRAManager(c.draManager),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithMutableSnapshotLister(snapshot),
		frameworkruntime.WithCaptureProfile(frameworkruntime.CaptureProfile(c.options.frameworkCapturer)),
		frameworkruntime.WithParallelism(int(c.options.parallelism)),
		frameworkruntime.WithExtenders(c.extenders),
		frameworkruntime.WithMetricsRecorder(c.metricsRecorder),
		frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
		frameworkruntime.WithPodsInPreBind(frameworkruntime.NewPodsInPreBindMap()),
		frameworkruntime.WithAPIDispatcher(c.apiDispatcher),
		frameworkruntime.WithSharedCSIManager(csiManager),
		frameworkruntime.WithPodGroupManager(c.cache),
	)
	if err != nil {
		return nil, fmt.Errorf("initializing profiles: %w", err)
	}
	if len(profiles) == 0 {
		return nil, errors.New("at least one profile is required")
	}

	return profiles, nil
}
