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

// Frameworks bundles the scheduling frameworks with the shared components
// constructed alongside them. It is created by NewFrameworks and exposed so
// out-of-tree consumers  can build frameworks dentical to kube-scheduler's
// without duplicating the wiring.
type Frameworks struct {
	Profiles  profile.Map
	Extenders []fwk.Extender

	// APIDispatcher is non-nil when the SchedulerAsyncAPICalls feature gate is enabled.
	APIDispatcher   *apidispatcher.APIDispatcher
	MetricsRecorder *metrics.MetricAsyncRecorder

	// DRA components required for registering event handlers. Nil unless
	// DynamicResourceAllocation feature gate is enabled.
	ResourceClaimCache   *assumecache.AssumeCache
	ResourceSliceTracker *resourceslicetracker.Tracker
	DraManager           fwk.SharedDRAManager

	// Cache stores cluster state shared across framework profiles, scheduling queue,
	// and event handlers.
	Cache internalcache.Cache

	// ProfileConfigs holds the resolved per-profile configuration, after default
	// profile defaulting has been applied.
	ProfileConfigs []schedulerapi.KubeSchedulerProfile
}

// NewFrameworks builds the scheduling frameworks and their shared
// dependencies exactly the way kube-scheduler does.
//
// snapshot is injected into every framework as its SharedLister; Scheduler passes
// the scheduler's internal cache snapshot, library consumers pass their own.
//
// NewFrameworks does not register metrics and does not set a PodNominator,
// PodActivator, or APICacher on the frameworks — callers wire those
// afterwards via the corresponding Framework setters.
func NewFrameworks(ctx context.Context,
	client clientset.Interface,
	informerFactory informers.SharedInformerFactory,
	recorderFactory profile.RecorderFactory,
	snapshot *internalcache.Snapshot,
	opts ...Option) (*Frameworks, error) {

	options := defaultSchedulerOptions
	for _, opt := range opts {
		opt(&options)
	}
	return buildFrameworks(ctx, client, informerFactory, recorderFactory, snapshot, options)
}

// buildFrameworks is the shared framework-construction phase of scheduler.New().
func buildFrameworks(
	ctx context.Context,
	client clientset.Interface,
	informerFactory informers.SharedInformerFactory,
	recorderFactory profile.RecorderFactory,
	snapshot *internalcache.Snapshot,
	options schedulerOptions,
) (*Frameworks, error) {

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

	registry := frameworkplugins.NewInTreeRegistry()
	if err := registry.Merge(options.frameworkOutOfTreeRegistry); err != nil {
		return nil, err
	}

	extenders, err := buildExtenders(logger, options.extenders, options.profiles)
	if err != nil {
		return nil, fmt.Errorf("couldn't build extenders: %w", err)
	}

	metricsRecorder := metrics.NewMetricsAsyncRecorder(1000, time.Second, stopEverything)
	// waitingPods holds all the pods that are in the scheduler and waiting in the permit stage
	waitingPods := frameworkruntime.NewWaitingPodsMap()

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
	sharedCSIManager := nodevolumelimits.NewCSIManager(informerFactory.Storage().V1().CSINodes().Lister())

	var apiDispatcher *apidispatcher.APIDispatcher
	if feature.DefaultFeatureGate.Enabled(features.SchedulerAsyncAPICalls) {
		apiDispatcher = apidispatcher.New(client, int(options.parallelism), apicalls.Relevances)
	}

	schedulerCache := internalcache.New(ctx, apiDispatcher, feature.DefaultFeatureGate.Enabled(features.GenericWorkload), feature.DefaultFeatureGate.Enabled(features.CompositePodGroup))
	podsInPreBind := frameworkruntime.NewPodsInPreBindMap()

	profiles, err := profile.NewMap(ctx, options.profiles, registry, recorderFactory,
		frameworkruntime.WithComponentConfigVersion(options.componentConfigVersion),
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithKubeConfig(options.kubeConfig),
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSharedDRAManager(draManager),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
		frameworkruntime.WithMutableSnapshotLister(snapshot),
		frameworkruntime.WithCaptureProfile(frameworkruntime.CaptureProfile(options.frameworkCapturer)),
		frameworkruntime.WithParallelism(int(options.parallelism)),
		frameworkruntime.WithExtenders(extenders),
		frameworkruntime.WithMetricsRecorder(metricsRecorder),
		frameworkruntime.WithWaitingPods(waitingPods),
		frameworkruntime.WithPodsInPreBind(podsInPreBind),
		frameworkruntime.WithAPIDispatcher(apiDispatcher),
		frameworkruntime.WithSharedCSIManager(sharedCSIManager),
		frameworkruntime.WithPodGroupManager(schedulerCache),
	)

	if err != nil {
		return nil, fmt.Errorf("initializing profiles: %v", err)
	}

	if len(profiles) == 0 {
		return nil, errors.New("at least one profile is required")
	}

	return &Frameworks{
		Profiles:             profiles,
		Extenders:            extenders,
		APIDispatcher:        apiDispatcher,
		MetricsRecorder:      metricsRecorder,
		ResourceClaimCache:   resourceClaimCache,
		ResourceSliceTracker: resourceSliceTracker,
		DraManager:           draManager,
		Cache:                schedulerCache,
		ProfileConfigs:       options.profiles,
	}, nil
}
