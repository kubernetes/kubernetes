/*
Copyright 2019 The Kubernetes Authors.

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

package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/util/slice"
)

const (
	// Specifies the maximum timeout a permit plugin can return.
	maxTimeout = 15 * time.Minute
)

// frameworkImpl is the component responsible for initializing and running scheduler
// plugins.
type frameworkImpl struct {
	registry             Registry
	snapshotSharedLister fwk.SharedLister
	waitingPods          *waitingPodsMap
	scorePluginWeight    map[string]int
	preEnqueuePlugins    []fwk.PreEnqueuePlugin
	enqueueExtensions    []fwk.EnqueueExtensions
	queueSortPlugins     []fwk.QueueSortPlugin
	preFilterPlugins     []fwk.PreFilterPlugin
	filterPlugins        []fwk.FilterPlugin
	postFilterPlugins    []fwk.PostFilterPlugin
	preScorePlugins      []fwk.PreScorePlugin
	scorePlugins         []fwk.ScorePlugin
	reservePlugins       []fwk.ReservePlugin
	preBindPlugins       []fwk.PreBindPlugin
	bindPlugins          []fwk.BindPlugin
	postBindPlugins      []fwk.PostBindPlugin
	permitPlugins        []fwk.PermitPlugin
	batchablePlugins     []fwk.SignPlugin

	// pluginsMap contains all plugins, by name.
	pluginsMap map[string]fwk.Plugin

	clientSet        clientset.Interface
	kubeConfig       *restclient.Config
	eventRecorder    events.EventRecorder
	informerFactory  informers.SharedInformerFactory
	sharedDRAManager fwk.SharedDRAManager
	workloadManager  fwk.WorkloadManager
	logger           klog.Logger

	sharedCSIManager fwk.CSIManager

	metricsRecorder          *metrics.MetricAsyncRecorder
	profileName              string
	percentageOfNodesToScore *int32

	extenders []fwk.Extender
	fwk.PodNominator
	fwk.PodActivator
	apiDispatcher *apidispatcher.APIDispatcher
	apiCacher     fwk.APICacher

	parallelizer fwk.Parallelizer

	batch *OpportunisticBatch

	enableSignatures bool
}

// extensionPoint encapsulates desired and applied set of plugins at a specific extension
// point. This is used to simplify iterating over all extension points supported by the
// frameworkImpl.
type extensionPoint struct {
	// the set of plugins to be configured at this extension point.
	plugins *config.PluginSet
	// a pointer to the slice storing plugins implementations that will run at this
	// extension point.
	slicePtr interface{}
}

func (f *frameworkImpl) getExtensionPoints(plugins *config.Plugins) []extensionPoint {
	return []extensionPoint{
		{&plugins.PreFilter, &f.preFilterPlugins},
		{&plugins.Filter, &f.filterPlugins},
		{&plugins.PostFilter, &f.postFilterPlugins},
		{&plugins.Reserve, &f.reservePlugins},
		{&plugins.PreScore, &f.preScorePlugins},
		{&plugins.Score, &f.scorePlugins},
		{&plugins.PreBind, &f.preBindPlugins},
		{&plugins.Bind, &f.bindPlugins},
		{&plugins.PostBind, &f.postBindPlugins},
		{&plugins.Permit, &f.permitPlugins},
		{&plugins.PreEnqueue, &f.preEnqueuePlugins},
		{&plugins.QueueSort, &f.queueSortPlugins},
	}
}

// Extenders returns the registered extenders.
func (f *frameworkImpl) Extenders() []fwk.Extender {
	return f.extenders
}

type frameworkOptions struct {
	componentConfigVersion string
	clientSet              clientset.Interface
	kubeConfig             *restclient.Config
	eventRecorder          events.EventRecorder
	informerFactory        informers.SharedInformerFactory
	sharedDRAManager       fwk.SharedDRAManager
	sharedCSIManager       fwk.CSIManager
	snapshotSharedLister   fwk.SharedLister
	metricsRecorder        *metrics.MetricAsyncRecorder
	podNominator           fwk.PodNominator
	podActivator           fwk.PodActivator
	extenders              []fwk.Extender
	captureProfile         CaptureProfile
	parallelizer           parallelize.Parallelizer
	waitingPods            *waitingPodsMap
	apiDispatcher          *apidispatcher.APIDispatcher
	workloadManager        fwk.WorkloadManager
	logger                 *klog.Logger
}

// Option for the frameworkImpl.
type Option func(*frameworkOptions)

// WithComponentConfigVersion sets the component config version to the
// KubeSchedulerConfiguration version used. The string should be the full
// scheme group/version of the external type we converted from (for example
// "kubescheduler.config.k8s.io/v1")
func WithComponentConfigVersion(componentConfigVersion string) Option {
	return func(o *frameworkOptions) {
		o.componentConfigVersion = componentConfigVersion
	}
}

// WithClientSet sets clientSet for the scheduling frameworkImpl.
func WithClientSet(clientSet clientset.Interface) Option {
	return func(o *frameworkOptions) {
		o.clientSet = clientSet
	}
}

// WithKubeConfig sets kubeConfig for the scheduling frameworkImpl.
func WithKubeConfig(kubeConfig *restclient.Config) Option {
	return func(o *frameworkOptions) {
		o.kubeConfig = kubeConfig
	}
}

// WithEventRecorder sets clientSet for the scheduling frameworkImpl.
func WithEventRecorder(recorder events.EventRecorder) Option {
	return func(o *frameworkOptions) {
		o.eventRecorder = recorder
	}
}

// WithInformerFactory sets informer factory for the scheduling frameworkImpl.
func WithInformerFactory(informerFactory informers.SharedInformerFactory) Option {
	return func(o *frameworkOptions) {
		o.informerFactory = informerFactory
	}
}

// WithSharedDRAManager sets SharedDRAManager for the framework.
func WithSharedDRAManager(sharedDRAManager fwk.SharedDRAManager) Option {
	return func(o *frameworkOptions) {
		o.sharedDRAManager = sharedDRAManager
	}
}

// WithSharedCSIManager sets SharedCSIManager for the framework.
func WithSharedCSIManager(sharedCSIManager fwk.CSIManager) Option {
	return func(o *frameworkOptions) {
		o.sharedCSIManager = sharedCSIManager
	}
}

// WithSnapshotSharedLister sets the SharedLister of the snapshot.
func WithSnapshotSharedLister(snapshotSharedLister fwk.SharedLister) Option {
	return func(o *frameworkOptions) {
		o.snapshotSharedLister = snapshotSharedLister
	}
}

// WithPodNominator sets podNominator for the scheduling frameworkImpl.
func WithPodNominator(nominator fwk.PodNominator) Option {
	return func(o *frameworkOptions) {
		o.podNominator = nominator
	}
}

func WithPodActivator(activator fwk.PodActivator) Option {
	return func(o *frameworkOptions) {
		o.podActivator = activator
	}
}

// WithExtenders sets extenders for the scheduling frameworkImpl.
func WithExtenders(extenders []fwk.Extender) Option {
	return func(o *frameworkOptions) {
		o.extenders = extenders
	}
}

// WithParallelism sets parallelism for the scheduling frameworkImpl.
func WithParallelism(parallelism int) Option {
	return func(o *frameworkOptions) {
		o.parallelizer = parallelize.NewParallelizer(parallelism)
	}
}

// WithAPIDispatcher sets API dispatcher for the scheduling frameworkImpl.
func WithAPIDispatcher(apiDispatcher *apidispatcher.APIDispatcher) Option {
	return func(o *frameworkOptions) {
		o.apiDispatcher = apiDispatcher
	}
}

// WithWorkloadManager sets Workload manager for the scheduling frameworkImpl.
func WithWorkloadManager(workloadManager fwk.WorkloadManager) Option {
	return func(o *frameworkOptions) {
		o.workloadManager = workloadManager
	}
}

// CaptureProfile is a callback to capture a finalized profile.
type CaptureProfile func(config.KubeSchedulerProfile)

// WithCaptureProfile sets a callback to capture the finalized profile.
func WithCaptureProfile(c CaptureProfile) Option {
	return func(o *frameworkOptions) {
		o.captureProfile = c
	}
}

// WithMetricsRecorder sets metrics recorder for the scheduling frameworkImpl.
func WithMetricsRecorder(r *metrics.MetricAsyncRecorder) Option {
	return func(o *frameworkOptions) {
		o.metricsRecorder = r
	}
}

// WithWaitingPods sets waitingPods for the scheduling frameworkImpl.
func WithWaitingPods(wp *waitingPodsMap) Option {
	return func(o *frameworkOptions) {
		o.waitingPods = wp
	}
}

// WithLogger overrides the default logger from k8s.io/klog.
func WithLogger(logger klog.Logger) Option {
	return func(o *frameworkOptions) {
		o.logger = &logger
	}
}

// defaultFrameworkOptions are applied when no option corresponding to those fields exist.
func defaultFrameworkOptions(stopCh <-chan struct{}) frameworkOptions {
	return frameworkOptions{
		metricsRecorder: metrics.NewMetricsAsyncRecorder(1000, time.Second, stopCh),
		parallelizer:    parallelize.NewParallelizer(parallelize.DefaultParallelism),
	}
}

var _ framework.Framework = &frameworkImpl{}

// NewFramework initializes plugins given the configuration and the registry.
func NewFramework(ctx context.Context, r Registry, profile *config.KubeSchedulerProfile, opts ...Option) (framework.Framework, error) {
	options := defaultFrameworkOptions(ctx.Done())
	for _, opt := range opts {
		opt(&options)
	}

	logger := klog.FromContext(ctx)
	if options.logger != nil {
		logger = *options.logger
	}
	f := &frameworkImpl{
		registry:             r,
		snapshotSharedLister: options.snapshotSharedLister,
		sharedCSIManager:     options.sharedCSIManager,
		scorePluginWeight:    make(map[string]int),
		waitingPods:          options.waitingPods,
		clientSet:            options.clientSet,
		kubeConfig:           options.kubeConfig,
		eventRecorder:        options.eventRecorder,
		informerFactory:      options.informerFactory,
		sharedDRAManager:     options.sharedDRAManager,
		metricsRecorder:      options.metricsRecorder,
		extenders:            options.extenders,
		PodNominator:         options.podNominator,
		PodActivator:         options.podActivator,
		apiDispatcher:        options.apiDispatcher,
		workloadManager:      options.workloadManager,
		parallelizer:         options.parallelizer,
		logger:               logger,
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.OpportunisticBatching) {
		f.batch = newOpportunisticBatch(f, signUsingFramework)
	}

	if len(f.extenders) > 0 {
		// Extender doesn't support any kind of requeueing feature like EnqueueExtensions in the scheduling framework.
		// We register a defaultEnqueueExtension to fwk.ExtenderName here.
		// And, in the scheduling cycle, when Extenders reject some Nodes and the pod ends up being unschedulable,
		// we put fwk.ExtenderName to pInfo.UnschedulablePlugins.
		f.enqueueExtensions = []fwk.EnqueueExtensions{&defaultEnqueueExtension{pluginName: framework.ExtenderName}}
	}

	if profile == nil {
		return f, nil
	}

	f.profileName = profile.SchedulerName
	f.percentageOfNodesToScore = profile.PercentageOfNodesToScore
	if profile.Plugins == nil {
		return f, nil
	}

	// get needed plugins from config
	pg := f.pluginsNeeded(profile.Plugins)

	pluginConfig := make(map[string]runtime.Object, len(profile.PluginConfig))
	for i := range profile.PluginConfig {
		name := profile.PluginConfig[i].Name
		if _, ok := pluginConfig[name]; ok {
			return nil, fmt.Errorf("repeated config for plugin %s", name)
		}
		pluginConfig[name] = profile.PluginConfig[i].Args
	}
	outputProfile := config.KubeSchedulerProfile{
		SchedulerName:            f.profileName,
		PercentageOfNodesToScore: f.percentageOfNodesToScore,
		Plugins:                  profile.Plugins,
		PluginConfig:             make([]config.PluginConfig, 0, len(pg)),
	}

	f.pluginsMap = make(map[string]fwk.Plugin)
	for name, factory := range r {
		// initialize only needed plugins.
		if !pg.Has(name) {
			continue
		}

		args := pluginConfig[name]
		if args != nil {
			outputProfile.PluginConfig = append(outputProfile.PluginConfig, config.PluginConfig{
				Name: name,
				Args: args,
			})
		}
		p, err := factory(ctx, args, f)
		if err != nil {
			return nil, fmt.Errorf("initializing plugin %q: %w", name, err)
		}
		f.pluginsMap[name] = p

		f.fillEnqueueExtensions(p)
	}

	// initialize plugins per individual extension points
	for _, e := range f.getExtensionPoints(profile.Plugins) {
		if err := updatePluginList(e.slicePtr, *e.plugins, f.pluginsMap); err != nil {
			return nil, err
		}
	}

	// initialize multiPoint plugins to their expanded extension points
	if len(profile.Plugins.MultiPoint.Enabled) > 0 {
		if err := f.expandMultiPointPlugins(logger, profile); err != nil {
			return nil, err
		}
	}

	if len(f.queueSortPlugins) != 1 {
		return nil, fmt.Errorf("only one queue sort plugin required for profile with scheduler name %q, but got %d", profile.SchedulerName, len(f.queueSortPlugins))
	}
	if len(f.bindPlugins) == 0 {
		return nil, fmt.Errorf("at least one bind plugin is needed for profile with scheduler name %q", profile.SchedulerName)
	}

	if err := getScoreWeights(f, append(profile.Plugins.Score.Enabled, profile.Plugins.MultiPoint.Enabled...)); err != nil {
		return nil, err
	}

	// Verifying the score weights again since Plugin.Name() could return a different
	// value from the one used in the configuration.
	for _, scorePlugin := range f.scorePlugins {
		if f.scorePluginWeight[scorePlugin.Name()] == 0 {
			return nil, fmt.Errorf("score plugin %q is not configured with weight", scorePlugin.Name())
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.OpportunisticBatching) {
		f.computeBatchablePlugins()
	}

	if options.captureProfile != nil {
		if len(outputProfile.PluginConfig) != 0 {
			sort.Slice(outputProfile.PluginConfig, func(i, j int) bool {
				return outputProfile.PluginConfig[i].Name < outputProfile.PluginConfig[j].Name
			})
		} else {
			outputProfile.PluginConfig = nil
		}
		options.captureProfile(outputProfile)
	}

	// Logs Enabled Plugins at each extension point, taking default plugins, given config, and multipoint into consideration
	logger.V(2).Info("the scheduler starts to work with those plugins", "Plugins", *f.ListPlugins())
	f.setInstrumentedPlugins()
	return f, nil
}

// setInstrumentedPlugins initializes instrumented plugins from current plugins that frameworkImpl has.
func (f *frameworkImpl) setInstrumentedPlugins() {
	// Cache metric streams for prefilter and filter plugins.
	for i, pl := range f.preFilterPlugins {
		f.preFilterPlugins[i] = &instrumentedPreFilterPlugin{
			PreFilterPlugin: f.preFilterPlugins[i],
			metric:          metrics.PluginEvaluationTotal.WithLabelValues(pl.Name(), metrics.PreFilter, f.profileName),
		}
	}
	for i, pl := range f.filterPlugins {
		f.filterPlugins[i] = &instrumentedFilterPlugin{
			FilterPlugin: f.filterPlugins[i],
			metric:       metrics.PluginEvaluationTotal.WithLabelValues(pl.Name(), metrics.Filter, f.profileName),
		}
	}

	// Cache metric streams for prescore and score plugins.
	for i, pl := range f.preScorePlugins {
		f.preScorePlugins[i] = &instrumentedPreScorePlugin{
			PreScorePlugin: f.preScorePlugins[i],
			metric:         metrics.PluginEvaluationTotal.WithLabelValues(pl.Name(), metrics.PreScore, f.profileName),
		}
	}
	for i, pl := range f.scorePlugins {
		f.scorePlugins[i] = &instrumentedScorePlugin{
			ScorePlugin: f.scorePlugins[i],
			metric:      metrics.PluginEvaluationTotal.WithLabelValues(pl.Name(), metrics.Score, f.profileName),
		}
	}
}

func (f *frameworkImpl) SetPodNominator(n fwk.PodNominator) {
	f.PodNominator = n
}

func (f *frameworkImpl) SetPodActivator(a fwk.PodActivator) {
	f.PodActivator = a
}

func (f *frameworkImpl) SetAPICacher(c fwk.APICacher) {
	f.apiCacher = c
}

// Close closes each plugin, when they implement io.Closer interface.
func (f *frameworkImpl) Close() error {
	var errs []error
	for name, plugin := range f.pluginsMap {
		if closer, ok := plugin.(io.Closer); ok {
			err := closer.Close()
			if err != nil {
				errs = append(errs, fmt.Errorf("%s failed to close: %w", name, err))
				// We try to close all plugins even if we got errors from some.
			}
		}
	}
	return errors.Join(errs...)
}

// getScoreWeights makes sure that, between MultiPoint-Score plugin weights and individual Score
// plugin weights there is not an overflow of MaxTotalScore.
func getScoreWeights(f *frameworkImpl, plugins []config.Plugin) error {
	var totalPriority int64
	scorePlugins := reflect.ValueOf(&f.scorePlugins).Elem()
	pluginType := scorePlugins.Type().Elem()
	for _, e := range plugins {
		pg := f.pluginsMap[e.Name]
		if !reflect.TypeOf(pg).Implements(pluginType) {
			continue
		}

		// We append MultiPoint plugins to the list of Score plugins. So if this plugin has already been
		// encountered, let the individual Score weight take precedence.
		if _, ok := f.scorePluginWeight[e.Name]; ok {
			continue
		}
		// a weight of zero is not permitted, plugins can be disabled explicitly
		// when configured.
		f.scorePluginWeight[e.Name] = int(e.Weight)
		if f.scorePluginWeight[e.Name] == 0 {
			f.scorePluginWeight[e.Name] = 1
		}

		// Checks totalPriority against MaxTotalScore to avoid overflow
		if int64(f.scorePluginWeight[e.Name])*fwk.MaxNodeScore > fwk.MaxTotalScore-totalPriority {
			return fmt.Errorf("total score of Score plugins could overflow")
		}
		totalPriority += int64(f.scorePluginWeight[e.Name]) * fwk.MaxNodeScore
	}
	return nil
}

type orderedSet struct {
	set         map[string]int
	list        []string
	deletionCnt int
}

func newOrderedSet() *orderedSet {
	return &orderedSet{set: make(map[string]int)}
}

func (os *orderedSet) insert(s string) {
	if os.has(s) {
		return
	}
	os.set[s] = len(os.list)
	os.list = append(os.list, s)
}

func (os *orderedSet) has(s string) bool {
	_, found := os.set[s]
	return found
}

func (os *orderedSet) delete(s string) {
	if i, found := os.set[s]; found {
		delete(os.set, s)
		os.list = append(os.list[:i-os.deletionCnt], os.list[i+1-os.deletionCnt:]...)
		os.deletionCnt++
	}
}

func (f *frameworkImpl) expandMultiPointPlugins(logger klog.Logger, profile *config.KubeSchedulerProfile) error {
	// initialize MultiPoint plugins
	for _, e := range f.getExtensionPoints(profile.Plugins) {
		plugins := reflect.ValueOf(e.slicePtr).Elem()
		pluginType := plugins.Type().Elem()
		// build enabledSet of plugins already registered via normal extension points
		// to check double registration
		enabledSet := newOrderedSet()
		for _, plugin := range e.plugins.Enabled {
			enabledSet.insert(plugin.Name)
		}

		disabledSet := sets.New[string]()
		for _, disabledPlugin := range e.plugins.Disabled {
			disabledSet.Insert(disabledPlugin.Name)
		}
		if disabledSet.Has("*") {
			logger.V(4).Info("Skipped MultiPoint expansion because all plugins are disabled for extension point", "extension", pluginType)
			continue
		}

		// track plugins enabled via multipoint separately from those enabled by specific extensions,
		// so that we can distinguish between double-registration and explicit overrides
		multiPointEnabled := newOrderedSet()
		overridePlugins := newOrderedSet()
		for _, ep := range profile.Plugins.MultiPoint.Enabled {
			pg, ok := f.pluginsMap[ep.Name]
			if !ok {
				return fmt.Errorf("%s %q does not exist", pluginType.Name(), ep.Name)
			}

			// if this plugin doesn't implement the type for the current extension we're trying to expand, skip
			if !reflect.TypeOf(pg).Implements(pluginType) {
				continue
			}

			// a plugin that's enabled via MultiPoint can still be disabled for specific extension points
			if disabledSet.Has(ep.Name) {
				logger.V(4).Info("Skipped disabled plugin for extension point", "plugin", ep.Name, "extension", pluginType)
				continue
			}

			// if this plugin has already been enabled by the specific extension point,
			// the user intent is to override the default plugin or make some other explicit setting.
			// Either way, discard the MultiPoint value for this plugin.
			// This maintains expected behavior for overriding default plugins (see https://github.com/kubernetes/kubernetes/pull/99582)
			if enabledSet.has(ep.Name) {
				overridePlugins.insert(ep.Name)
				logger.Info("MultiPoint plugin is explicitly re-configured; overriding", "plugin", ep.Name)
				continue
			}

			// if this plugin is already registered via MultiPoint, then this is
			// a double registration and an error in the config.
			if multiPointEnabled.has(ep.Name) {
				return fmt.Errorf("plugin %q already registered as %q", ep.Name, pluginType.Name())
			}

			// we only need to update the multipoint set, since we already have the specific extension set from above
			multiPointEnabled.insert(ep.Name)
		}

		// Reorder plugins. Here is the expected order:
		// - part 1: overridePlugins. Their order stay intact as how they're specified in regular extension point.
		// - part 2: multiPointEnabled - i.e., plugin defined in multipoint but not in regular extension point.
		// - part 3: other plugins (excluded by part 1 & 2) in regular extension point.
		newPlugins := reflect.New(reflect.TypeOf(e.slicePtr).Elem()).Elem()
		// part 1
		for _, name := range slice.CopyStrings(enabledSet.list) {
			if overridePlugins.has(name) {
				newPlugins = reflect.Append(newPlugins, reflect.ValueOf(f.pluginsMap[name]))
				enabledSet.delete(name)
			}
		}
		// part 2
		for _, name := range multiPointEnabled.list {
			newPlugins = reflect.Append(newPlugins, reflect.ValueOf(f.pluginsMap[name]))
		}
		// part 3
		for _, name := range enabledSet.list {
			newPlugins = reflect.Append(newPlugins, reflect.ValueOf(f.pluginsMap[name]))
		}
		plugins.Set(newPlugins)
	}
	return nil
}

func shouldHaveEnqueueExtensions(p fwk.Plugin) bool {
	switch p.(type) {
	// Only PreEnqueue, PreFilter, Filter, Reserve, and Permit plugins can (should) have EnqueueExtensions.
	// See the comment of EnqueueExtensions for more detailed reason here.
	case fwk.PreEnqueuePlugin, fwk.PreFilterPlugin, fwk.FilterPlugin, fwk.ReservePlugin, fwk.PermitPlugin:
		return true
	}
	return false
}

func (f *frameworkImpl) fillEnqueueExtensions(p fwk.Plugin) {
	if !shouldHaveEnqueueExtensions(p) {
		// Ignore EnqueueExtensions from plugin which isn't PreEnqueue, PreFilter, Filter, Reserve, and Permit.
		return
	}

	ext, ok := p.(fwk.EnqueueExtensions)
	if !ok {
		// If interface EnqueueExtensions is not implemented, register the default enqueue extensions
		// to the plugin because we don't know which events the plugin is interested in.
		// This is to ensure backward compatibility.
		f.enqueueExtensions = append(f.enqueueExtensions, &defaultEnqueueExtension{pluginName: p.Name()})
		return
	}

	f.enqueueExtensions = append(f.enqueueExtensions, ext)
}

// defaultEnqueueExtension is used when a plugin does not implement EnqueueExtensions interface.
type defaultEnqueueExtension struct {
	pluginName string
}

func (p *defaultEnqueueExtension) Name() string { return p.pluginName }
func (p *defaultEnqueueExtension) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	// need to return all specific cluster events with framework.All action instead of wildcard event
	// because the returning values are used to register event handlers.
	// If we return the wildcard here, it won't affect the event handlers registered by the plugin
	// and some events may not be registered in the event handlers.
	return framework.UnrollWildCardResource(), nil
}

func updatePluginList(pluginList interface{}, pluginSet config.PluginSet, pluginsMap map[string]fwk.Plugin) error {
	plugins := reflect.ValueOf(pluginList).Elem()
	pluginType := plugins.Type().Elem()
	set := sets.New[string]()
	for _, ep := range pluginSet.Enabled {
		pg, ok := pluginsMap[ep.Name]
		if !ok {
			return fmt.Errorf("%s %q does not exist", pluginType.Name(), ep.Name)
		}

		if !reflect.TypeOf(pg).Implements(pluginType) {
			return fmt.Errorf("plugin %q does not extend %s plugin", ep.Name, pluginType.Name())
		}

		if set.Has(ep.Name) {
			return fmt.Errorf("plugin %q already registered as %q", ep.Name, pluginType.Name())
		}

		set.Insert(ep.Name)

		newPlugins := reflect.Append(plugins, reflect.ValueOf(pg))
		plugins.Set(newPlugins)
	}
	return nil
}

// PreEnqueuePlugins returns the registered preEnqueue plugins.
func (f *frameworkImpl) PreEnqueuePlugins() []fwk.PreEnqueuePlugin {
	return f.preEnqueuePlugins
}

// EnqueueExtensions returns the registered reenqueue plugins.
func (f *frameworkImpl) EnqueueExtensions() []fwk.EnqueueExtensions {
	return f.enqueueExtensions
}

// QueueSortFunc returns the function to sort pods in scheduling queue
func (f *frameworkImpl) QueueSortFunc() fwk.LessFunc {
	if f == nil {
		// If frameworkImpl is nil, simply keep their order unchanged.
		// NOTE: this is primarily for tests.
		return func(_, _ fwk.QueuedPodInfo) bool { return false }
	}

	if len(f.queueSortPlugins) == 0 {
		panic("No QueueSort plugin is registered in the frameworkImpl.")
	}

	// Only one QueueSort plugin can be enabled.
	return f.queueSortPlugins[0].Less
}

// If any of our preFilter, filter, preScore or score plugins haven't
// implemented a signature, then disable the cache.
func (f *frameworkImpl) computeBatchablePlugins() {
	f.enableSignatures = true

	if len(f.extenders) > 0 {
		f.logger.Info("Disabling signatures for profile because it has extenders configured.",
			"profile", f.profileName)

		f.enableSignatures = false
	}

	// Get all plugins of compatible types.
	candidatePlugins := []fwk.Plugin{}
	for _, pl := range f.preFilterPlugins {
		candidatePlugins = append(candidatePlugins, pl)
	}
	for _, pl := range f.filterPlugins {
		candidatePlugins = append(candidatePlugins, pl)
	}
	for _, pl := range f.preScorePlugins {
		candidatePlugins = append(candidatePlugins, pl)
	}
	for _, pl := range f.scorePlugins {
		candidatePlugins = append(candidatePlugins, pl)
	}

	// Get signature elements from plugins.
	plugins := map[string]fwk.SignPlugin{}
	unsupportedPlugins := sets.New[string]()
	for _, pl := range candidatePlugins {
		if _, found := plugins[pl.Name()]; !found {
			if _, implements := pl.(fwk.SignPlugin); implements {
				f.batchablePlugins = append(f.batchablePlugins, pl.(fwk.SignPlugin))
				plugins[pl.Name()] = pl.(fwk.SignPlugin)
			} else {
				unsupportedPlugins.Insert(pl.Name())
				f.enableSignatures = false
			}
		}
	}

	if !f.enableSignatures {
		f.logger.Info("Disabling signatures for profile because plugins do not support it.",
			"profile", f.profileName, "plugins", unsupportedPlugins.UnsortedList())
	}
}

// SignPod returns a signature for a given pod. Any two pods with the same signature should get
// the same feasibility and scoring for the same set of nodes in the same state. If one or more plugins
// is unable to construct a signature for the pod, the result will be nil, which means
// there is no way to compare this pod against others, and will turn off a number of optimizations
// for this pod.
func (f *frameworkImpl) SignPod(ctx context.Context, pod *v1.Pod, recordPluginStats bool) fwk.PodSignature {
	logger := klog.FromContext(ctx)
	var status *fwk.Status

	if recordPluginStats {
		startTime := time.Now()
		defer func() {
			metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Sign, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
		}()
	}

	if !f.enableSignatures {
		return nil
	}

	sig := map[string]any{
		fwk.SchedulerNameSignerName: pod.Spec.SchedulerName,
	}

	for _, plugin := range f.batchablePlugins {
		startTime := time.Now()
		fragments, status := plugin.SignPod(ctx, pod)
		f.metricsRecorder.ObservePluginDurationAsync(metrics.Sign, plugin.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))

		if !status.IsSuccess() {
			if status.Code() == fwk.Error {
				logger.Error(status.AsError(), "SignPod failed for plugin", "plugin", plugin.Name())
			}
			logger.V(5).Info("SignPod can't sign pod due to plugin", "plugin", plugin.Name(), "status", status)
			return nil
		}

		for _, elem := range fragments {
			sig[elem.Key] = elem.Value
		}
	}

	sigBytes, err := json.Marshal(sig)
	if err != nil {
		logger.Error(err, "SignPod failed to marshal signature object")
		return nil
	}

	return sigBytes
}

// RunPreFilterPlugins runs the set of configured PreFilter plugins. It returns
// *Status and its code is set to non-success if any of the plugins returns
// anything but Success/Skip.
// When it returns Skip status, returned PreFilterResult and other fields in status are just ignored,
// and coupled Filter plugin/PreFilterExtensions() will be skipped in this scheduling cycle.
// If a non-success status is returned, then the scheduling cycle is aborted.
func (f *frameworkImpl) RunPreFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod) (_ *fwk.PreFilterResult, status *fwk.Status, _ sets.Set[string]) {
	startTime := time.Now()
	skipPlugins := sets.New[string]()
	defer func() {
		state.SetSkipFilterPlugins(skipPlugins)
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.PreFilter, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	nodes, err := f.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("getting all nodes: %w", err)), nil
	}
	var result *fwk.PreFilterResult
	pluginsWithNodes := sets.New[string]()
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PreFilter")
	}
	var returnStatus *fwk.Status
	for _, pl := range f.preFilterPlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		r, s := f.runPreFilterPlugin(ctx, pl, state, pod, nodes)
		if s.IsSkip() {
			skipPlugins.Insert(pl.Name())
			continue
		}
		if !s.IsSuccess() {
			s.SetPlugin(pl.Name())
			if s.Code() == fwk.UnschedulableAndUnresolvable {
				// In this case, the preemption shouldn't happen in this scheduling cycle.
				// So, no need to execute all PreFilter.
				return nil, s, nil
			}
			if s.Code() == fwk.Unschedulable {
				// In this case, the preemption should happen later in this scheduling cycle.
				// So we need to execute all PreFilter.
				// https://github.com/kubernetes/kubernetes/issues/119770
				returnStatus = s
				continue
			}
			return nil, fwk.AsStatus(fmt.Errorf("running PreFilter plugin %q: %w", pl.Name(), s.AsError())).WithPlugin(pl.Name()), nil
		}
		if !r.AllNodes() {
			pluginsWithNodes.Insert(pl.Name())
		}
		result = result.Merge(r)
		if !result.AllNodes() && len(result.NodeNames) == 0 {
			msg := fmt.Sprintf("node(s) didn't satisfy plugin(s) %v simultaneously", sets.List(pluginsWithNodes))
			if len(pluginsWithNodes) == 1 {
				msg = fmt.Sprintf("node(s) didn't satisfy plugin %v", sets.List(pluginsWithNodes)[0])
			}

			// When PreFilterResult filters out Nodes, the framework considers Nodes that are filtered out as getting "UnschedulableAndUnresolvable".
			return result, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, msg), pluginsWithNodes
		}
	}
	return result, returnStatus, pluginsWithNodes
}

func (f *frameworkImpl) runPreFilterPlugin(ctx context.Context, pl fwk.PreFilterPlugin, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PreFilter(ctx, state, pod, nodes)
	}
	startTime := time.Now()
	result, status := pl.PreFilter(ctx, state, pod, nodes)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PreFilter, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return result, status
}

// RunPreFilterExtensionAddPod calls the AddPod interface for the set of configured
// PreFilter plugins. It returns directly if any of the plugins return any
// status other than Success.
func (f *frameworkImpl) RunPreFilterExtensionAddPod(
	ctx context.Context,
	state fwk.CycleState,
	podToSchedule *v1.Pod,
	podInfoToAdd fwk.PodInfo,
	nodeInfo fwk.NodeInfo,
) (status *fwk.Status) {
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PreFilterExtension")
	}
	for _, pl := range f.preFilterPlugins {
		if pl.PreFilterExtensions() == nil || state.GetSkipFilterPlugins().Has(pl.Name()) {
			continue
		}
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runPreFilterExtensionAddPod(ctx, pl, state, podToSchedule, podInfoToAdd, nodeInfo)
		if !status.IsSuccess() {
			err := status.AsError()
			logger.Error(err, "Plugin failed", "pod", klog.KObj(podToSchedule), "node", klog.KObj(nodeInfo.Node()), "operation", "addPod", "plugin", pl.Name())
			return fwk.AsStatus(fmt.Errorf("running AddPod on PreFilter plugin %q: %w", pl.Name(), err))
		}
	}

	return nil
}

func (f *frameworkImpl) runPreFilterExtensionAddPod(ctx context.Context, pl fwk.PreFilterPlugin, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PreFilterExtensions().AddPod(ctx, state, podToSchedule, podInfoToAdd, nodeInfo)
	}
	startTime := time.Now()
	status := pl.PreFilterExtensions().AddPod(ctx, state, podToSchedule, podInfoToAdd, nodeInfo)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PreFilterExtensionAddPod, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunPreFilterExtensionRemovePod calls the RemovePod interface for the set of configured
// PreFilter plugins. It returns directly if any of the plugins return any
// status other than Success.
func (f *frameworkImpl) RunPreFilterExtensionRemovePod(
	ctx context.Context,
	state fwk.CycleState,
	podToSchedule *v1.Pod,
	podInfoToRemove fwk.PodInfo,
	nodeInfo fwk.NodeInfo,
) (status *fwk.Status) {
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PreFilterExtension")
	}
	for _, pl := range f.preFilterPlugins {
		if pl.PreFilterExtensions() == nil || state.GetSkipFilterPlugins().Has(pl.Name()) {
			continue
		}
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runPreFilterExtensionRemovePod(ctx, pl, state, podToSchedule, podInfoToRemove, nodeInfo)
		if !status.IsSuccess() {
			err := status.AsError()
			var node *v1.Node
			if nodeInfo != nil {
				node = nodeInfo.Node()
			}
			logger.Error(err, "Plugin failed", "node", klog.KObj(node), "operation", "removePod", "plugin", pl.Name(), "pod", klog.KObj(podToSchedule))
			return fwk.AsStatus(fmt.Errorf("running RemovePod on PreFilter plugin %q: %w", pl.Name(), err))
		}
	}

	return nil
}

func (f *frameworkImpl) runPreFilterExtensionRemovePod(ctx context.Context, pl fwk.PreFilterPlugin, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PreFilterExtensions().RemovePod(ctx, state, podToSchedule, podInfoToRemove, nodeInfo)
	}
	startTime := time.Now()
	status := pl.PreFilterExtensions().RemovePod(ctx, state, podToSchedule, podInfoToRemove, nodeInfo)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PreFilterExtensionRemovePod, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunFilterPlugins runs the set of configured Filter plugins for pod on
// the given node. If any of these plugins doesn't return "Success", the
// given node is not suitable for running pod.
// Meanwhile, the failure message and status are set for the given node.
func (f *frameworkImpl) RunFilterPlugins(
	ctx context.Context,
	state fwk.CycleState,
	pod *v1.Pod,
	nodeInfo fwk.NodeInfo,
) *fwk.Status {
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "Filter")
	}

	for _, pl := range f.filterPlugins {
		if state.GetSkipFilterPlugins().Has(pl.Name()) {
			continue
		}
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		if status := f.runFilterPlugin(ctx, pl, state, pod, nodeInfo); !status.IsSuccess() {
			if !status.IsRejected() {
				// Filter plugins are not supposed to return any status other than
				// Success or Unschedulable.
				status = fwk.AsStatus(fmt.Errorf("running %q filter plugin: %w", pl.Name(), status.AsError()))
			}
			status.SetPlugin(pl.Name())
			return status
		}
	}

	return nil
}

func (f *frameworkImpl) runFilterPlugin(ctx context.Context, pl fwk.FilterPlugin, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.Filter(ctx, state, pod, nodeInfo)
	}
	startTime := time.Now()
	status := pl.Filter(ctx, state, pod, nodeInfo)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.Filter, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunPostFilterPlugins runs the set of configured PostFilter plugins until the first
// Success, Error or UnschedulableAndUnresolvable is met; otherwise continues to execute all plugins.
func (f *frameworkImpl) RunPostFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap fwk.NodeToStatusReader) (_ *fwk.PostFilterResult, status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.PostFilter, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()

	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PostFilter")
	}

	// `result` records the last meaningful(non-noop) PostFilterResult.
	var result *fwk.PostFilterResult
	var reasons []string
	var rejectorPlugin string
	for _, pl := range f.postFilterPlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		r, s := f.runPostFilterPlugin(ctx, pl, state, pod, filteredNodeStatusMap)
		if s.IsSuccess() {
			return r, s
		} else if s.Code() == fwk.UnschedulableAndUnresolvable {
			return r, s.WithPlugin(pl.Name())
		} else if !s.IsRejected() {
			// Any status other than Success, Unschedulable or UnschedulableAndUnresolvable is Error.
			return nil, fwk.AsStatus(s.AsError()).WithPlugin(pl.Name())
		} else if r != nil && r.Mode() != fwk.ModeNoop {
			result = r
		}

		reasons = append(reasons, s.Reasons()...)
		// Record the first failed plugin unless we proved that
		// the latter is more relevant.
		if len(rejectorPlugin) == 0 {
			rejectorPlugin = pl.Name()
		}
	}

	return result, fwk.NewStatus(fwk.Unschedulable, reasons...).WithPlugin(rejectorPlugin)
}

func (f *frameworkImpl) runPostFilterPlugin(ctx context.Context, pl fwk.PostFilterPlugin, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PostFilter(ctx, state, pod, filteredNodeStatusMap)
	}
	startTime := time.Now()
	r, s := pl.PostFilter(ctx, state, pod, filteredNodeStatusMap)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PostFilter, pl.Name(), s.Code().String(), metrics.SinceInSeconds(startTime))
	return r, s
}

// RunFilterPluginsWithNominatedPods runs the set of configured filter plugins
// for nominated pod on the given node.
// This function is called from two different places: Schedule and Preempt.
// When it is called from Schedule, we want to test whether the pod is
// schedulable on the node with all the existing pods on the node plus higher
// and equal priority pods nominated to run on the node.
// When it is called from Preempt, we should remove the victims of preemption
// and add the nominated pods. Removal of the victims is done by
// SelectVictimsOnNode(). Preempt removes victims from PreFilter state and
// NodeInfo before calling this function.
func (f *frameworkImpl) RunFilterPluginsWithNominatedPods(ctx context.Context, state fwk.CycleState, pod *v1.Pod, info fwk.NodeInfo) *fwk.Status {
	var status *fwk.Status

	podsAdded := false
	// We run filters twice in some cases. If the node has greater or equal priority
	// nominated pods, we run them when those pods are added to PreFilter state and nodeInfo.
	// If all filters succeed in this pass, we run them again when these
	// nominated pods are not added. This second pass is necessary because some
	// filters such as inter-pod affinity may not pass without the nominated pods.
	// If there are no nominated pods for the node or if the first run of the
	// filters fail, we don't run the second pass.
	// We consider only equal or higher priority pods in the first pass, because
	// those are the current "pod" must yield to them and not take a space opened
	// for running them. It is ok if the current "pod" take resources freed for
	// lower priority pods.
	// Requiring that the new pod is schedulable in both circumstances ensures that
	// we are making a conservative decision: filters like resources and inter-pod
	// anti-affinity are more likely to fail when the nominated pods are treated
	// as running, while filters like pod affinity are more likely to fail when
	// the nominated pods are treated as not running. We can't just assume the
	// nominated pods are running because they are not running right now and in fact,
	// they may end up getting scheduled to a different node.
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithName(logger, "FilterWithNominatedPods")
	ctx = klog.NewContext(ctx, logger)
	for i := 0; i < 2; i++ {
		stateToUse := state
		nodeInfoToUse := info
		if i == 0 {
			var err error
			podsAdded, stateToUse, nodeInfoToUse, err = addGENominatedPods(ctx, f, pod, state, info)
			if err != nil {
				return fwk.AsStatus(err)
			}
		} else if !podsAdded || !status.IsSuccess() {
			break
		}

		status = f.RunFilterPlugins(ctx, stateToUse, pod, nodeInfoToUse)
		if !status.IsSuccess() && !status.IsRejected() {
			return status
		}
	}

	return status
}

// addGENominatedPods adds pods with equal or greater priority which are nominated
// to run on the node. It returns 1) whether any pod was added, 2) augmented cycleState,
// 3) augmented nodeInfo.
func addGENominatedPods(ctx context.Context, fh fwk.Handle, pod *v1.Pod, state fwk.CycleState, nodeInfo fwk.NodeInfo) (bool, fwk.CycleState, fwk.NodeInfo, error) {
	if fh == nil {
		// This may happen only in tests.
		return false, state, nodeInfo, nil
	}
	nominatedPodInfos := fh.NominatedPodsForNode(nodeInfo.Node().Name)
	if len(nominatedPodInfos) == 0 {
		return false, state, nodeInfo, nil
	}
	nodeInfoOut := nodeInfo.Snapshot()
	stateOut := state.Clone()
	podsAdded := false
	for _, pi := range nominatedPodInfos {
		if corev1.PodPriority(pi.GetPod()) >= corev1.PodPriority(pod) && pi.GetPod().UID != pod.UID {
			nodeInfoOut.AddPodInfo(pi)
			status := fh.RunPreFilterExtensionAddPod(ctx, stateOut, pod, pi, nodeInfoOut)
			if !status.IsSuccess() {
				return false, state, nodeInfo, status.AsError()
			}
			podsAdded = true
		}
	}
	return podsAdded, stateOut, nodeInfoOut, nil
}

// RunPreScorePlugins runs the set of configured pre-score plugins. If any
// of these plugins returns any status other than Success/Skip, the given pod is rejected.
// When it returns Skip status, other fields in status are just ignored,
// and coupled Score plugin will be skipped in this scheduling cycle.
func (f *frameworkImpl) RunPreScorePlugins(
	ctx context.Context,
	state fwk.CycleState,
	pod *v1.Pod,
	nodes []fwk.NodeInfo,
) (status *fwk.Status) {
	startTime := time.Now()
	skipPlugins := sets.New[string]()
	defer func() {
		state.SetSkipScorePlugins(skipPlugins)
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.PreScore, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PreScore")
	}
	for _, pl := range f.preScorePlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runPreScorePlugin(ctx, pl, state, pod, nodes)
		if status.IsSkip() {
			skipPlugins.Insert(pl.Name())
			continue
		}
		if !status.IsSuccess() {
			return fwk.AsStatus(fmt.Errorf("running PreScore plugin %q: %w", pl.Name(), status.AsError()))
		}
	}
	return nil
}

func (f *frameworkImpl) runPreScorePlugin(ctx context.Context, pl fwk.PreScorePlugin, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PreScore(ctx, state, pod, nodes)
	}
	startTime := time.Now()
	status := pl.PreScore(ctx, state, pod, nodes)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PreScore, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunScorePlugins runs the set of configured scoring plugins.
// It returns a list that stores scores from each plugin and total score for each Node.
// It also returns *Status, which is set to non-success if any of the plugins returns
// a non-success status.
func (f *frameworkImpl) RunScorePlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (ns []fwk.NodePluginScores, status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Score, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	allNodePluginScores := make([]fwk.NodePluginScores, len(nodes))
	numPlugins := len(f.scorePlugins)
	plugins := make([]fwk.ScorePlugin, 0, numPlugins)
	pluginToNodeScores := make(map[string]fwk.NodeScoreList, numPlugins)
	for _, pl := range f.scorePlugins {
		if state.GetSkipScorePlugins().Has(pl.Name()) {
			continue
		}
		plugins = append(plugins, pl)
		pluginToNodeScores[pl.Name()] = make(fwk.NodeScoreList, len(nodes))
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errCh := parallelize.NewErrorChannel()

	if len(plugins) > 0 {
		logger := klog.FromContext(ctx)
		verboseLogs := logger.V(4).Enabled()
		if verboseLogs {
			logger = klog.LoggerWithName(logger, "Score")
		}
		// Run Score method for each node in parallel.
		f.Parallelizer().Until(ctx, len(nodes), func(index int) {
			nodeInfo := nodes[index]
			nodeName := nodeInfo.Node().Name
			logger := logger
			if verboseLogs {
				logger = klog.LoggerWithValues(logger, "node", klog.ObjectRef{Name: nodeName})
			}
			for _, pl := range plugins {
				ctx := ctx
				if verboseLogs {
					logger := klog.LoggerWithName(logger, pl.Name())
					ctx = klog.NewContext(ctx, logger)
				}
				s, status := f.runScorePlugin(ctx, pl, state, pod, nodeInfo)
				if !status.IsSuccess() {
					err := fmt.Errorf("plugin %q failed with: %w", pl.Name(), status.AsError())
					errCh.SendErrorWithCancel(err, cancel)
					return
				}
				pluginToNodeScores[pl.Name()][index] = fwk.NodeScore{
					Name:  nodeName,
					Score: s,
				}
			}
		}, metrics.Score)
		if err := errCh.ReceiveError(); err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("running Score plugins: %w", err))
		}
	}

	// Run NormalizeScore method for each ScorePlugin in parallel.
	f.Parallelizer().Until(ctx, len(plugins), func(index int) {
		pl := plugins[index]
		if pl.ScoreExtensions() == nil {
			return
		}
		nodeScoreList := pluginToNodeScores[pl.Name()]
		status := f.runScoreExtension(ctx, pl, state, pod, nodeScoreList)
		if !status.IsSuccess() {
			err := fmt.Errorf("plugin %q failed with: %w", pl.Name(), status.AsError())
			errCh.SendErrorWithCancel(err, cancel)
			return
		}
	}, metrics.Score)
	if err := errCh.ReceiveError(); err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("running Normalize on Score plugins: %w", err))
	}

	// Apply score weight for each ScorePlugin in parallel,
	// and then, build allNodePluginScores.
	f.Parallelizer().Until(ctx, len(nodes), func(index int) {
		nodePluginScores := fwk.NodePluginScores{
			Name:   nodes[index].Node().Name,
			Scores: make([]fwk.PluginScore, len(plugins)),
		}

		for i, pl := range plugins {
			weight := f.scorePluginWeight[pl.Name()]
			nodeScoreList := pluginToNodeScores[pl.Name()]
			score := nodeScoreList[index].Score

			if score > fwk.MaxNodeScore || score < fwk.MinNodeScore {
				err := fmt.Errorf("plugin %q returns an invalid score %v, it should in the range of [%v, %v] after normalizing", pl.Name(), score, fwk.MinNodeScore, fwk.MaxNodeScore)
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
			weightedScore := score * int64(weight)
			nodePluginScores.Scores[i] = fwk.PluginScore{
				Name:  pl.Name(),
				Score: weightedScore,
			}
			nodePluginScores.TotalScore += weightedScore
		}
		allNodePluginScores[index] = nodePluginScores
	}, metrics.Score)
	if err := errCh.ReceiveError(); err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("applying score defaultWeights on Score plugins: %w", err))
	}

	return allNodePluginScores, nil
}

func (f *frameworkImpl) runScorePlugin(ctx context.Context, pl fwk.ScorePlugin, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	if !state.ShouldRecordPluginMetrics() {
		return pl.Score(ctx, state, pod, nodeInfo)
	}
	startTime := time.Now()
	s, status := pl.Score(ctx, state, pod, nodeInfo)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.Score, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return s, status
}

func (f *frameworkImpl) runScoreExtension(ctx context.Context, pl fwk.ScorePlugin, state fwk.CycleState, pod *v1.Pod, nodeScoreList fwk.NodeScoreList) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.ScoreExtensions().NormalizeScore(ctx, state, pod, nodeScoreList)
	}
	startTime := time.Now()
	status := pl.ScoreExtensions().NormalizeScore(ctx, state, pod, nodeScoreList)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.ScoreExtensionNormalize, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

func (f *frameworkImpl) GetNodeHint(ctx context.Context, pod *v1.Pod, state fwk.CycleState, cycleCount int64) (hint string, signature fwk.PodSignature) {
	return f.batch.GetNodeHint(ctx, pod, state, cycleCount)
}

func (f *frameworkImpl) StoreScheduleResults(ctx context.Context, signature fwk.PodSignature, hintedNode, chosenNode string, otherNodes framework.SortedScoredNodes, cycleCount int64) {
	f.batch.StoreScheduleResults(ctx, signature, hintedNode, chosenNode, otherNodes, cycleCount)
}

// RunPreBindPlugins runs the set of configured prebind plugins. It returns a
// failure (bool) if any of the plugins returns an error. It also returns an
// error containing the rejection message or the error occurred in the plugin.
func (f *frameworkImpl) RunPreBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.PreBind, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PreBind")
		logger = klog.LoggerWithValues(logger, "node", klog.ObjectRef{Name: nodeName})
	}
	for _, pl := range f.preBindPlugins {
		if state.GetSkipPreBindPlugins().Has(pl.Name()) {
			continue
		}

		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runPreBindPlugin(ctx, pl, state, pod, nodeName)
		if !status.IsSuccess() {
			if status.IsRejected() {
				logger.V(4).Info("Pod rejected by PreBind plugin", "pod", klog.KObj(pod), "node", nodeName, "plugin", pl.Name(), "status", status.Message())
				status.SetPlugin(pl.Name())
				return status
			}
			err := status.AsError()
			logger.Error(err, "Plugin failed", "plugin", pl.Name(), "pod", klog.KObj(pod), "node", nodeName)
			return fwk.AsStatus(fmt.Errorf("running PreBind plugin %q: %w", pl.Name(), err))
		}
	}
	return nil
}

func (f *frameworkImpl) runPreBindPlugin(ctx context.Context, pl fwk.PreBindPlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PreBind(ctx, state, pod, nodeName)
	}
	startTime := time.Now()
	status := pl.PreBind(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PreBind, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunPreBindPreFlights runs the set of configured PreBindPreFlight functions from PreBind plugins.
// The returning value is:
// - Success: one or more plugins return success, meaning, some PreBind plugins will work for this pod.
// - Skip: all plugins return skip.
// - Error: any plugin return error.
func (f *frameworkImpl) RunPreBindPreFlights(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.PreBindPreFlight, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PreBindPreFlight")
		logger = klog.LoggerWithValues(logger, "node", klog.ObjectRef{Name: nodeName})
	}
	skipPlugins := sets.New[string]()
	returningStatus := fwk.NewStatus(fwk.Skip)
	for _, pl := range f.preBindPlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runPreBindPreFlight(ctx, pl, state, pod, nodeName)
		switch {
		case status.Code() == fwk.Error:
			err := status.AsError()
			logger.Error(err, "Plugin failed", "plugin", pl.Name(), "pod", klog.KObj(pod), "node", nodeName)
			return fwk.AsStatus(fmt.Errorf("running PreBindPreFlight %q: %w", pl.Name(), err))
		case status.IsSuccess():
			// We return success when one or more plugins return success.
			returningStatus = nil
		case status.IsSkip():
			skipPlugins.Insert(pl.Name())
		default:
			// Other statuses are unexpected
			return fwk.AsStatus(fmt.Errorf("PreBindPreFlight %s returned %q, which is unsupported. It is supposed to return Success, Skip, or Error status", pl.Name(), status.Code()))
		}
	}
	state.SetSkipPreBindPlugins(skipPlugins)
	return returningStatus
}

func (f *frameworkImpl) runPreBindPreFlight(ctx context.Context, pl fwk.PreBindPlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.PreBindPreFlight(ctx, state, pod, nodeName)
	}
	startTime := time.Now()
	status := pl.PreBindPreFlight(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PreBindPreFlight, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunBindPlugins runs the set of configured bind plugins until one returns a non `Skip` status.
func (f *frameworkImpl) RunBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Bind, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	if len(f.bindPlugins) == 0 {
		return fwk.NewStatus(fwk.Skip, "")
	}
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "Bind")
	}
	for _, pl := range f.bindPlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runBindPlugin(ctx, pl, state, pod, nodeName)
		if status.IsSkip() {
			continue
		}
		if !status.IsSuccess() {
			if status.IsRejected() {
				logger.V(4).Info("Pod rejected by Bind plugin", "pod", klog.KObj(pod), "node", nodeName, "plugin", pl.Name(), "status", status.Message())
				status.SetPlugin(pl.Name())
				return status
			}
			err := status.AsError()
			logger.Error(err, "Plugin Failed", "plugin", pl.Name(), "pod", klog.KObj(pod), "node", nodeName)
			return fwk.AsStatus(fmt.Errorf("running Bind plugin %q: %w", pl.Name(), err))
		}
		return status
	}
	return status
}

func (f *frameworkImpl) runBindPlugin(ctx context.Context, bp fwk.BindPlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return bp.Bind(ctx, state, pod, nodeName)
	}
	startTime := time.Now()
	status := bp.Bind(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.Bind, bp.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunPostBindPlugins runs the set of configured postbind plugins.
func (f *frameworkImpl) RunPostBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.PostBind, fwk.Success.String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "PostBind")
	}
	for _, pl := range f.postBindPlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		f.runPostBindPlugin(ctx, pl, state, pod, nodeName)
	}
}

func (f *frameworkImpl) runPostBindPlugin(ctx context.Context, pl fwk.PostBindPlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	if !state.ShouldRecordPluginMetrics() {
		pl.PostBind(ctx, state, pod, nodeName)
		return
	}
	startTime := time.Now()
	pl.PostBind(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.PostBind, pl.Name(), fwk.Success.String(), metrics.SinceInSeconds(startTime))
}

// RunReservePluginsReserve runs the Reserve method in the set of configured
// reserve plugins. If any of these plugins returns an error, it does not
// continue running the remaining ones and returns the error. In such a case,
// the pod will not be scheduled and the caller will be expected to call
// RunReservePluginsUnreserve.
func (f *frameworkImpl) RunReservePluginsReserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Reserve, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "Reserve")
		logger = klog.LoggerWithValues(logger, "node", klog.ObjectRef{Name: nodeName})
	}
	for _, pl := range f.reservePlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status = f.runReservePluginReserve(ctx, pl, state, pod, nodeName)
		if !status.IsSuccess() {
			if status.IsRejected() {
				logger.V(4).Info("Pod rejected by plugin", "pod", klog.KObj(pod), "plugin", pl.Name(), "status", status.Message())
				status.SetPlugin(pl.Name())
				return status
			}
			err := status.AsError()
			logger.Error(err, "Plugin failed", "plugin", pl.Name(), "pod", klog.KObj(pod))
			return fwk.AsStatus(fmt.Errorf("running Reserve plugin %q: %w", pl.Name(), err))
		}
	}
	return nil
}

func (f *frameworkImpl) runReservePluginReserve(ctx context.Context, pl fwk.ReservePlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !state.ShouldRecordPluginMetrics() {
		return pl.Reserve(ctx, state, pod, nodeName)
	}
	startTime := time.Now()
	status := pl.Reserve(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.Reserve, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status
}

// RunReservePluginsUnreserve runs the Unreserve method in the set of
// configured reserve plugins.
func (f *frameworkImpl) RunReservePluginsUnreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Unreserve, fwk.Success.String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	// Execute the Unreserve operation of each reserve plugin in the
	// *reverse* order in which the Reserve operation was executed.
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "Unreserve")
		logger = klog.LoggerWithValues(logger, "node", klog.ObjectRef{Name: nodeName})
	}
	for i := len(f.reservePlugins) - 1; i >= 0; i-- {
		pl := f.reservePlugins[i]
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		f.runReservePluginUnreserve(ctx, pl, state, pod, nodeName)
	}
}

func (f *frameworkImpl) runReservePluginUnreserve(ctx context.Context, pl fwk.ReservePlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	if !state.ShouldRecordPluginMetrics() {
		pl.Unreserve(ctx, state, pod, nodeName)
		return
	}
	startTime := time.Now()
	pl.Unreserve(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.Unreserve, pl.Name(), fwk.Success.String(), metrics.SinceInSeconds(startTime))
}

// RunPermitPlugins runs the set of configured permit plugins. If any of these
// plugins returns a status other than "Success" or "Wait", it does not continue
// running the remaining plugins and returns an error. Otherwise, if any of the
// plugins returns "Wait", then this function will create and add waiting pod
// to a map of currently waiting pods and return status with "Wait" code.
// Pod will remain waiting pod for the minimum duration returned by the permit plugins.
func (f *frameworkImpl) RunPermitPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Permit, status.Code().String(), f.profileName).Observe(metrics.SinceInSeconds(startTime))
	}()
	pluginsWaitTime := make(map[string]time.Duration)
	statusCode := fwk.Success
	logger := klog.FromContext(ctx)
	verboseLogs := logger.V(4).Enabled()
	if verboseLogs {
		logger = klog.LoggerWithName(logger, "Permit")
		logger = klog.LoggerWithValues(logger, "node", klog.ObjectRef{Name: nodeName})
	}
	for _, pl := range f.permitPlugins {
		ctx := ctx
		if verboseLogs {
			logger := klog.LoggerWithName(logger, pl.Name())
			ctx = klog.NewContext(ctx, logger)
		}
		status, timeout := f.runPermitPlugin(ctx, pl, state, pod, nodeName)
		if !status.IsSuccess() {
			if status.IsRejected() {
				logger.V(4).Info("Pod rejected by plugin", "pod", klog.KObj(pod), "plugin", pl.Name(), "status", status.Message())
				return status.WithPlugin(pl.Name())
			}
			if status.IsWait() {
				// Not allowed to be greater than maxTimeout.
				if timeout > maxTimeout {
					timeout = maxTimeout
				}
				pluginsWaitTime[pl.Name()] = timeout
				statusCode = fwk.Wait
			} else {
				err := status.AsError()
				logger.Error(err, "Plugin failed", "plugin", pl.Name(), "pod", klog.KObj(pod))
				return fwk.AsStatus(fmt.Errorf("running Permit plugin %q: %w", pl.Name(), err)).WithPlugin(pl.Name())
			}
		}
	}
	if statusCode == fwk.Wait {
		waitingPod := newWaitingPod(pod, pluginsWaitTime)
		f.waitingPods.add(waitingPod)
		msg := fmt.Sprintf("one or more plugins asked to wait and no plugin rejected pod %q", pod.Name)
		logger.V(4).Info("One or more plugins asked to wait and no plugin rejected pod", "pod", klog.KObj(pod))
		return fwk.NewStatus(fwk.Wait, msg)
	}
	return nil
}

func (f *frameworkImpl) runPermitPlugin(ctx context.Context, pl fwk.PermitPlugin, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if !state.ShouldRecordPluginMetrics() {
		return pl.Permit(ctx, state, pod, nodeName)
	}
	startTime := time.Now()
	status, timeout := pl.Permit(ctx, state, pod, nodeName)
	f.metricsRecorder.ObservePluginDurationAsync(metrics.Permit, pl.Name(), status.Code().String(), metrics.SinceInSeconds(startTime))
	return status, timeout
}

func (f *frameworkImpl) WillWaitOnPermit(ctx context.Context, pod *v1.Pod) bool {
	return f.waitingPods.get(pod.UID) != nil
}

// WaitOnPermit will block, if the pod is a waiting pod, until the waiting pod is rejected or allowed.
func (f *frameworkImpl) WaitOnPermit(ctx context.Context, pod *v1.Pod) *fwk.Status {
	waitingPod := f.waitingPods.get(pod.UID)
	if waitingPod == nil {
		return nil
	}
	defer f.waitingPods.remove(pod.UID)

	logger := klog.FromContext(ctx)
	logger.V(4).Info("Pod waiting on permit", "pod", klog.KObj(pod))

	startTime := time.Now()
	s := <-waitingPod.s
	metrics.PermitWaitDuration.WithLabelValues(s.Code().String()).Observe(metrics.SinceInSeconds(startTime))

	if !s.IsSuccess() {
		if s.IsRejected() {
			logger.V(4).Info("Pod rejected while waiting on permit", "pod", klog.KObj(pod), "status", s.Message())
			return s
		}
		err := s.AsError()
		logger.Error(err, "Failed waiting on permit for pod", "pod", klog.KObj(pod))
		return fwk.AsStatus(fmt.Errorf("waiting on permit for pod: %w", err)).WithPlugin(s.Plugin())
	}
	return nil
}

// SnapshotSharedLister returns the scheduler's SharedLister of the latest NodeInfo
// snapshot. The snapshot is taken at the beginning of a scheduling cycle and remains
// unchanged until a pod finishes "Reserve". There is no guarantee that the information
// remains unchanged after "Reserve".
func (f *frameworkImpl) SnapshotSharedLister() fwk.SharedLister {
	return f.snapshotSharedLister
}

// IterateOverWaitingPods acquires a read lock and iterates over the WaitingPods map.
func (f *frameworkImpl) IterateOverWaitingPods(callback func(fwk.WaitingPod)) {
	f.waitingPods.iterate(callback)
}

// GetWaitingPod returns a reference to a WaitingPod given its UID.
func (f *frameworkImpl) GetWaitingPod(uid types.UID) fwk.WaitingPod {
	if wp := f.waitingPods.get(uid); wp != nil {
		return wp
	}
	return nil // Returning nil instead of *waitingPod(nil).
}

// RejectWaitingPod rejects a WaitingPod given its UID.
// The returned value indicates if the given pod is waiting or not.
func (f *frameworkImpl) RejectWaitingPod(uid types.UID) bool {
	if waitingPod := f.waitingPods.get(uid); waitingPod != nil {
		waitingPod.Reject("", "removed")
		return true
	}
	return false
}

// HasFilterPlugins returns true if at least one filter plugin is defined.
func (f *frameworkImpl) HasFilterPlugins() bool {
	return len(f.filterPlugins) > 0
}

// HasPostFilterPlugins returns true if at least one postFilter plugin is defined.
func (f *frameworkImpl) HasPostFilterPlugins() bool {
	return len(f.postFilterPlugins) > 0
}

// HasScorePlugins returns true if at least one score plugin is defined.
func (f *frameworkImpl) HasScorePlugins() bool {
	return len(f.scorePlugins) > 0
}

// ListPlugins returns a map of extension point name to plugin names configured at each extension
// point. Returns nil if no plugins where configured.
func (f *frameworkImpl) ListPlugins() *config.Plugins {
	m := config.Plugins{}

	for _, e := range f.getExtensionPoints(&m) {
		plugins := reflect.ValueOf(e.slicePtr).Elem()
		extName := plugins.Type().Elem().Name()
		var cfgs []config.Plugin
		for i := 0; i < plugins.Len(); i++ {
			name := plugins.Index(i).Interface().(fwk.Plugin).Name()
			p := config.Plugin{Name: name}
			if extName == "ScorePlugin" {
				// Weights apply only to score plugins.
				p.Weight = int32(f.scorePluginWeight[name])
			}
			cfgs = append(cfgs, p)
		}
		if len(cfgs) > 0 {
			e.plugins.Enabled = cfgs
		}
	}
	return &m
}

// ClientSet returns a kubernetes clientset.
func (f *frameworkImpl) ClientSet() clientset.Interface {
	return f.clientSet
}

// KubeConfig returns a kubernetes config.
func (f *frameworkImpl) KubeConfig() *restclient.Config {
	return f.kubeConfig
}

// EventRecorder returns an event recorder.
func (f *frameworkImpl) EventRecorder() events.EventRecorder {
	return f.eventRecorder
}

// SharedInformerFactory returns a shared informer factory.
func (f *frameworkImpl) SharedInformerFactory() informers.SharedInformerFactory {
	return f.informerFactory
}

// SharedDRAManager returns the SharedDRAManager of the framework.
func (f *frameworkImpl) SharedDRAManager() fwk.SharedDRAManager {
	return f.sharedDRAManager
}

// SharedCSIManager returns the SharedCSIManager of the framework.
func (f *frameworkImpl) SharedCSIManager() fwk.CSIManager {
	return f.sharedCSIManager
}

// WorkloadManager returns the WorkloadManager of the framework.
func (f *frameworkImpl) WorkloadManager() fwk.WorkloadManager {
	return f.workloadManager
}

func (f *frameworkImpl) pluginsNeeded(plugins *config.Plugins) sets.Set[string] {
	pgSet := sets.Set[string]{}

	if plugins == nil {
		return pgSet
	}

	find := func(pgs *config.PluginSet) {
		for _, pg := range pgs.Enabled {
			pgSet.Insert(pg.Name)
		}
	}

	for _, e := range f.getExtensionPoints(plugins) {
		find(e.plugins)
	}
	// Parse MultiPoint separately since they are not returned by f.getExtensionPoints()
	find(&plugins.MultiPoint)

	return pgSet
}

// ProfileName returns the profile name associated to this framework.
func (f *frameworkImpl) ProfileName() string {
	return f.profileName
}

// PercentageOfNodesToScore returns percentageOfNodesToScore associated to a profile.
func (f *frameworkImpl) PercentageOfNodesToScore() *int32 {
	return f.percentageOfNodesToScore
}

// Parallelizer returns a parallelizer holding parallelism for scheduler.
func (f *frameworkImpl) Parallelizer() fwk.Parallelizer {
	return f.parallelizer
}

// APIDispatcher returns an apiDispatcher that can be used to dispatch API calls.
// This requires SchedulerAsyncAPICalls feature gate to be enabled.
func (f *frameworkImpl) APIDispatcher() fwk.APIDispatcher {
	if f.apiDispatcher == nil {
		return nil
	}
	return f.apiDispatcher
}

// APICacher returns an apiCacher that can be used to dispatch API calls through scheduler's cache
// instead of directly using APIDispatcher().
// This requires SchedulerAsyncAPICalls feature gate to be enabled.
func (f *frameworkImpl) APICacher() fwk.APICacher {
	if f.apiCacher == nil {
		return nil
	}
	return f.apiCacher
}

// Used only for tests
func (f *frameworkImpl) TotalBatchedPods() int64 {
	return f.batch.batchedPods
}
