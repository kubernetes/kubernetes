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

package v1alpha1

import (
	"context"
	"fmt"
	"reflect"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// Specifies the maximum timeout a permit plugin can return.
	maxTimeout                  time.Duration = 15 * time.Minute
	preFilter                                 = "PreFilter"
	preFilterExtensionAddPod                  = "PreFilterExtensionAddPod"
	preFilterExtensionRemovePod               = "PreFilterExtensionRemovePod"
	filter                                    = "Filter"
	postFilter                                = "PostFilter"
	score                                     = "Score"
	preBind                                   = "PreBind"
	bind                                      = "Bind"
	postBind                                  = "PostBind"
	reserve                                   = "Reserve"
	unreserve                                 = "Unreserve"
	permit                                    = "Permit"
)

// framework is the component responsible for initializing and running scheduler
// plugins.
type framework struct {
	registry              Registry
	nodeInfoSnapshot      *schedulernodeinfo.Snapshot
	waitingPods           *waitingPodsMap
	pluginNameToWeightMap map[string]int
	queueSortPlugins      []QueueSortPlugin
	preFilterPlugins      []PreFilterPlugin
	filterPlugins         []FilterPlugin
	postFilterPlugins     []PostFilterPlugin
	scorePlugins          []ScorePlugin
	reservePlugins        []ReservePlugin
	preBindPlugins        []PreBindPlugin
	bindPlugins           []BindPlugin
	postBindPlugins       []PostBindPlugin
	unreservePlugins      []UnreservePlugin
	permitPlugins         []PermitPlugin

	clientSet       clientset.Interface
	informerFactory informers.SharedInformerFactory
}

// extensionPoint encapsulates desired and applied set of plugins at a specific extension
// point. This is used to simplify iterating over all extension points supported by the
// framework.
type extensionPoint struct {
	// the set of plugins to be configured at this extension point.
	plugins *config.PluginSet
	// a pointer to the slice storing plugins implementations that will run at this
	// extenstion point.
	slicePtr interface{}
}

func (f *framework) getExtensionPoints(plugins *config.Plugins) []extensionPoint {
	return []extensionPoint{
		{plugins.PreFilter, &f.preFilterPlugins},
		{plugins.Filter, &f.filterPlugins},
		{plugins.Reserve, &f.reservePlugins},
		{plugins.PostFilter, &f.postFilterPlugins},
		{plugins.Score, &f.scorePlugins},
		{plugins.PreBind, &f.preBindPlugins},
		{plugins.Bind, &f.bindPlugins},
		{plugins.PostBind, &f.postBindPlugins},
		{plugins.Unreserve, &f.unreservePlugins},
		{plugins.Permit, &f.permitPlugins},
		{plugins.QueueSort, &f.queueSortPlugins},
	}
}

type frameworkOptions struct {
	clientSet       clientset.Interface
	informerFactory informers.SharedInformerFactory
}

// Option for the framework.
type Option func(*frameworkOptions)

// WithClientSet sets clientSet for the scheduling framework.
func WithClientSet(clientSet clientset.Interface) Option {
	return func(o *frameworkOptions) {
		o.clientSet = clientSet
	}
}

// WithInformerFactory sets informer factory for the scheduling framework.
func WithInformerFactory(informerFactory informers.SharedInformerFactory) Option {
	return func(o *frameworkOptions) {
		o.informerFactory = informerFactory
	}
}

var defaultFrameworkOptions = frameworkOptions{}

var _ = Framework(&framework{})

// NewFramework initializes plugins given the configuration and the registry.
func NewFramework(r Registry, plugins *config.Plugins, args []config.PluginConfig, opts ...Option) (Framework, error) {
	options := defaultFrameworkOptions
	for _, opt := range opts {
		opt(&options)
	}

	f := &framework{
		registry:              r,
		nodeInfoSnapshot:      schedulernodeinfo.NewSnapshot(),
		pluginNameToWeightMap: make(map[string]int),
		waitingPods:           newWaitingPodsMap(),
		clientSet:             options.clientSet,
		informerFactory:       options.informerFactory,
	}
	if plugins == nil {
		return f, nil
	}

	// get needed plugins from config
	pg := f.pluginsNeeded(plugins)
	if len(pg) == 0 {
		return f, nil
	}

	pluginConfig := make(map[string]*runtime.Unknown, 0)
	for i := range args {
		pluginConfig[args[i].Name] = &args[i].Args
	}

	pluginsMap := make(map[string]Plugin)
	for name, factory := range r {
		// initialize only needed plugins.
		if _, ok := pg[name]; !ok {
			continue
		}

		p, err := factory(pluginConfig[name], f)
		if err != nil {
			return nil, fmt.Errorf("error initializing plugin %q: %v", name, err)
		}
		pluginsMap[name] = p

		// a weight of zero is not permitted, plugins can be disabled explicitly
		// when configured.
		f.pluginNameToWeightMap[name] = int(pg[name].Weight)
		if f.pluginNameToWeightMap[name] == 0 {
			f.pluginNameToWeightMap[name] = 1
		}
	}

	for _, e := range f.getExtensionPoints(plugins) {
		if err := updatePluginList(e.slicePtr, e.plugins, pluginsMap); err != nil {
			return nil, err
		}
	}

	// Verifying the score weights again since Plugin.Name() could return a different
	// value from the one used in the configuration.
	for _, scorePlugin := range f.scorePlugins {
		if f.pluginNameToWeightMap[scorePlugin.Name()] == 0 {
			return nil, fmt.Errorf("score plugin %q is not configured with weight", scorePlugin.Name())
		}
	}

	if len(f.queueSortPlugins) > 1 {
		return nil, fmt.Errorf("only one queue sort plugin can be enabled")
	}

	return f, nil
}

func updatePluginList(pluginList interface{}, pluginSet *config.PluginSet, pluginsMap map[string]Plugin) error {
	if pluginSet == nil {
		return nil
	}

	plugins := reflect.ValueOf(pluginList).Elem()
	pluginType := plugins.Type().Elem()
	set := sets.NewString()
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

// QueueSortFunc returns the function to sort pods in scheduling queue
func (f *framework) QueueSortFunc() LessFunc {
	if len(f.queueSortPlugins) == 0 {
		return nil
	}

	// Only one QueueSort plugin can be enabled.
	return f.queueSortPlugins[0].Less
}

// RunPreFilterPlugins runs the set of configured PreFilter plugins. It returns
// *Status and its code is set to non-success if any of the plugins returns
// anything but Success. If a non-success status is returned, then the scheduling
// cycle is aborted.
func (f *framework) RunPreFilterPlugins(
	state *CycleState, pod *v1.Pod) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, preFilter, status) }()
	for _, pl := range f.preFilterPlugins {
		status := pl.PreFilter(state, pod)
		if !status.IsSuccess() {
			if status.IsUnschedulable() {
				msg := fmt.Sprintf("rejected by %q at prefilter: %v", pl.Name(), status.Message())
				klog.V(4).Infof(msg)
				return NewStatus(status.Code(), msg)
			}
			msg := fmt.Sprintf("error while running %q prefilter plugin for pod %q: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}

	return nil
}

// RunPreFilterExtensionAddPod calls the AddPod interface for the set of configured
// PreFilter plugins. It returns directly if any of the plugins return any
// status other than Success.
func (f *framework) RunPreFilterExtensionAddPod(state *CycleState, podToSchedule *v1.Pod,
	podToAdd *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, preFilterExtensionAddPod, status) }()
	for _, pl := range f.preFilterPlugins {
		if pl.PreFilterExtensions() == nil {
			continue
		}
		if status := pl.PreFilterExtensions().AddPod(state, podToSchedule, podToAdd, nodeInfo); !status.IsSuccess() {
			msg := fmt.Sprintf("error while running AddPod for plugin %q while scheduling pod %q: %v",
				pl.Name(), podToSchedule.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}

	return nil
}

// RunPreFilterExtensionRemovePod calls the RemovePod interface for the set of configured
// PreFilter plugins. It returns directly if any of the plugins return any
// status other than Success.
func (f *framework) RunPreFilterExtensionRemovePod(state *CycleState, podToSchedule *v1.Pod,
	podToRemove *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, preFilterExtensionRemovePod, status) }()
	for _, pl := range f.preFilterPlugins {
		if pl.PreFilterExtensions() == nil {
			continue
		}
		if status := pl.PreFilterExtensions().RemovePod(state, podToSchedule, podToRemove, nodeInfo); !status.IsSuccess() {
			msg := fmt.Sprintf("error while running RemovePod for plugin %q while scheduling pod %q: %v",
				pl.Name(), podToSchedule.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}

	return nil
}

// RunFilterPlugins runs the set of configured Filter plugins for pod on
// the given node. If any of these plugins doesn't return "Success", the
// given node is not suitable for running pod.
// Meanwhile, the failure message and status are set for the given node.
func (f *framework) RunFilterPlugins(state *CycleState,
	pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, filter, status) }()
	for _, pl := range f.filterPlugins {
		status := pl.Filter(state, pod, nodeInfo)
		if !status.IsSuccess() {
			if !status.IsUnschedulable() {
				errMsg := fmt.Sprintf("error while running %q filter plugin for pod %q: %v",
					pl.Name(), pod.Name, status.Message())
				klog.Error(errMsg)
				return NewStatus(Error, errMsg)
			}
			return status
		}
	}

	return nil
}

// RunPostFilterPlugins runs the set of configured post-filter plugins. If any
// of these plugins returns any status other than "Success", the given node is
// rejected. The filteredNodeStatuses is the set of filtered nodes and their statuses.
func (f *framework) RunPostFilterPlugins(
	state *CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
	filteredNodesStatuses NodeToStatusMap,
) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, postFilter, status) }()
	for _, pl := range f.postFilterPlugins {
		status := pl.PostFilter(state, pod, nodes, filteredNodesStatuses)
		if !status.IsSuccess() {
			msg := fmt.Sprintf("error while running %q postfilter plugin for pod %q: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}

	return nil
}

// RunScorePlugins runs the set of configured scoring plugins. It returns a list that
// stores for each scoring plugin name the corresponding NodeScoreList(s).
// It also returns *Status, which is set to non-success if any of the plugins returns
// a non-success status.
func (f *framework) RunScorePlugins(state *CycleState, pod *v1.Pod, nodes []*v1.Node) (ps PluginToNodeScores, status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, score, status) }()
	pluginToNodeScores := make(PluginToNodeScores, len(f.scorePlugins))
	for _, pl := range f.scorePlugins {
		pluginToNodeScores[pl.Name()] = make(NodeScoreList, len(nodes))
	}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := schedutil.NewErrorChannel()

	// Run Score method for each node in parallel.
	workqueue.ParallelizeUntil(ctx, 16, len(nodes), func(index int) {
		for _, pl := range f.scorePlugins {
			nodeName := nodes[index].Name
			score, status := pl.Score(state, pod, nodeName)
			if !status.IsSuccess() {
				errCh.SendErrorWithCancel(fmt.Errorf(status.Message()), cancel)
				return
			}
			pluginToNodeScores[pl.Name()][index] = NodeScore{
				Name:  nodeName,
				Score: int64(score),
			}
		}
	})
	if err := errCh.ReceiveError(); err != nil {
		msg := fmt.Sprintf("error while running score plugin for pod %q: %v", pod.Name, err)
		klog.Error(msg)
		return nil, NewStatus(Error, msg)
	}

	// Run NormalizeScore method for each ScorePlugin in parallel.
	workqueue.ParallelizeUntil(ctx, 16, len(f.scorePlugins), func(index int) {
		pl := f.scorePlugins[index]
		nodeScoreList := pluginToNodeScores[pl.Name()]
		if pl.ScoreExtensions() == nil {
			return
		}
		status := pl.ScoreExtensions().NormalizeScore(state, pod, nodeScoreList)
		if !status.IsSuccess() {
			err := fmt.Errorf("normalize score plugin %q failed with error %v", pl.Name(), status.Message())
			errCh.SendErrorWithCancel(err, cancel)
			return
		}
	})
	if err := errCh.ReceiveError(); err != nil {
		msg := fmt.Sprintf("error while running normalize score plugin for pod %q: %v", pod.Name, err)
		klog.Error(msg)
		return nil, NewStatus(Error, msg)
	}

	// Apply score defaultWeights for each ScorePlugin in parallel.
	workqueue.ParallelizeUntil(ctx, 16, len(f.scorePlugins), func(index int) {
		pl := f.scorePlugins[index]
		// Score plugins' weight has been checked when they are initialized.
		weight := f.pluginNameToWeightMap[pl.Name()]
		nodeScoreList := pluginToNodeScores[pl.Name()]

		for i, nodeScore := range nodeScoreList {
			// return error if score plugin returns invalid score.
			if nodeScore.Score > int64(MaxNodeScore) || nodeScore.Score < int64(MinNodeScore) {
				err := fmt.Errorf("score plugin %q returns an invalid score %v, it should in the range of [%v, %v] after normalizing", pl.Name(), nodeScore.Score, MinNodeScore, MaxNodeScore)
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
			nodeScoreList[i].Score = nodeScore.Score * int64(weight)
		}
	})
	if err := errCh.ReceiveError(); err != nil {
		msg := fmt.Sprintf("error while applying score defaultWeights for pod %q: %v", pod.Name, err)
		klog.Error(msg)
		return nil, NewStatus(Error, msg)
	}

	return pluginToNodeScores, nil
}

// RunPreBindPlugins runs the set of configured prebind plugins. It returns a
// failure (bool) if any of the plugins returns an error. It also returns an
// error containing the rejection message or the error occurred in the plugin.
func (f *framework) RunPreBindPlugins(
	state *CycleState, pod *v1.Pod, nodeName string) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, preBind, status) }()
	for _, pl := range f.preBindPlugins {
		status := pl.PreBind(state, pod, nodeName)
		if !status.IsSuccess() {
			msg := fmt.Sprintf("error while running %q prebind plugin for pod %q: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}
	return nil
}

// RunBindPlugins runs the set of configured bind plugins until one returns a non `Skip` status.
func (f *framework) RunBindPlugins(state *CycleState, pod *v1.Pod, nodeName string) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, bind, status) }()
	if len(f.bindPlugins) == 0 {
		return NewStatus(Skip, "")
	}
	for _, bp := range f.bindPlugins {
		status = bp.Bind(state, pod, nodeName)
		if status != nil && status.Code() == Skip {
			continue
		}
		if !status.IsSuccess() {
			msg := fmt.Sprintf("bind plugin %q failed to bind pod \"%v/%v\": %v", bp.Name(), pod.Namespace, pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
		return status
	}
	return status
}

// RunPostBindPlugins runs the set of configured postbind plugins.
func (f *framework) RunPostBindPlugins(
	state *CycleState, pod *v1.Pod, nodeName string) {
	startTime := time.Now()
	defer recordExtensionPointDuration(startTime, postBind, nil)
	for _, pl := range f.postBindPlugins {
		pl.PostBind(state, pod, nodeName)
	}
}

// RunReservePlugins runs the set of configured reserve plugins. If any of these
// plugins returns an error, it does not continue running the remaining ones and
// returns the error. In such case, pod will not be scheduled.
func (f *framework) RunReservePlugins(
	state *CycleState, pod *v1.Pod, nodeName string) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, reserve, status) }()
	for _, pl := range f.reservePlugins {
		status := pl.Reserve(state, pod, nodeName)
		if !status.IsSuccess() {
			msg := fmt.Sprintf("error while running %q reserve plugin for pod %q: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}
	return nil
}

// RunUnreservePlugins runs the set of configured unreserve plugins.
func (f *framework) RunUnreservePlugins(
	state *CycleState, pod *v1.Pod, nodeName string) {
	startTime := time.Now()
	defer recordExtensionPointDuration(startTime, unreserve, nil)
	for _, pl := range f.unreservePlugins {
		pl.Unreserve(state, pod, nodeName)
	}
}

// RunPermitPlugins runs the set of configured permit plugins. If any of these
// plugins returns a status other than "Success" or "Wait", it does not continue
// running the remaining plugins and returns an error. Otherwise, if any of the
// plugins returns "Wait", then this function will block for the timeout period
// returned by the plugin, if the time expires, then it will return an error.
// Note that if multiple plugins asked to wait, then we wait for the minimum
// timeout duration.
func (f *framework) RunPermitPlugins(
	state *CycleState, pod *v1.Pod, nodeName string) (status *Status) {
	startTime := time.Now()
	defer func() { recordExtensionPointDuration(startTime, permit, status) }()
	timeout := maxTimeout
	statusCode := Success
	for _, pl := range f.permitPlugins {
		status, d := pl.Permit(state, pod, nodeName)
		if !status.IsSuccess() {
			if status.IsUnschedulable() {
				msg := fmt.Sprintf("rejected by %q at permit: %v", pl.Name(), status.Message())
				klog.V(4).Infof(msg)
				return NewStatus(status.Code(), msg)
			}
			if status.Code() == Wait {
				// Use the minimum timeout duration.
				if timeout > d {
					timeout = d
				}
				statusCode = Wait
			} else {
				msg := fmt.Sprintf("error while running %q permit plugin for pod %q: %v", pl.Name(), pod.Name, status.Message())
				klog.Error(msg)
				return NewStatus(Error, msg)
			}
		}
	}

	// We now wait for the minimum duration if at least one plugin asked to
	// wait (and no plugin rejected the pod)
	if statusCode == Wait {
		w := newWaitingPod(pod)
		f.waitingPods.add(w)
		defer f.waitingPods.remove(pod.UID)
		timer := time.NewTimer(timeout)
		klog.V(4).Infof("waiting for %v for pod %q at permit", timeout, pod.Name)
		select {
		case <-timer.C:
			msg := fmt.Sprintf("pod %q rejected due to timeout after waiting %v at permit", pod.Name, timeout)
			klog.V(4).Infof(msg)
			return NewStatus(Unschedulable, msg)
		case s := <-w.s:
			if !s.IsSuccess() {
				if s.IsUnschedulable() {
					msg := fmt.Sprintf("rejected while waiting at permit: %v", s.Message())
					klog.V(4).Infof(msg)
					return NewStatus(s.Code(), msg)
				}
				msg := fmt.Sprintf("error received while waiting at permit for pod %q: %v", pod.Name, s.Message())
				klog.Error(msg)
				return NewStatus(Error, msg)
			}
		}
	}

	return nil
}

// NodeInfoSnapshot returns the latest NodeInfo snapshot. The snapshot
// is taken at the beginning of a scheduling cycle and remains unchanged until a
// pod finishes "Reserve". There is no guarantee that the information remains
// unchanged after "Reserve".
func (f *framework) NodeInfoSnapshot() *schedulernodeinfo.Snapshot {
	return f.nodeInfoSnapshot
}

// IterateOverWaitingPods acquires a read lock and iterates over the WaitingPods map.
func (f *framework) IterateOverWaitingPods(callback func(WaitingPod)) {
	f.waitingPods.iterate(callback)
}

// GetWaitingPod returns a reference to a WaitingPod given its UID.
func (f *framework) GetWaitingPod(uid types.UID) WaitingPod {
	return f.waitingPods.get(uid)
}

// HasFilterPlugins returns true if at least one filter plugin is defined.
func (f *framework) HasFilterPlugins() bool {
	return len(f.filterPlugins) > 0
}

// ListPlugins returns a map of extension point name to plugin names configured at each extension
// point. Returns nil if no plugins where configred.
func (f *framework) ListPlugins() map[string][]config.Plugin {
	m := make(map[string][]config.Plugin)

	for _, e := range f.getExtensionPoints(&config.Plugins{}) {
		plugins := reflect.ValueOf(e.slicePtr).Elem()
		extName := plugins.Type().Elem().Name()
		var cfgs []config.Plugin
		for i := 0; i < plugins.Len(); i++ {
			name := plugins.Index(i).Interface().(Plugin).Name()
			p := config.Plugin{Name: name}
			if extName == "ScorePlugin" {
				// Weights apply only to score plugins.
				p.Weight = int32(f.pluginNameToWeightMap[name])
			}
			cfgs = append(cfgs, p)
		}
		if len(cfgs) > 0 {
			m[extName] = cfgs
		}
	}
	if len(m) > 0 {
		return m
	}
	return nil
}

// ClientSet returns a kubernetes clientset.
func (f *framework) ClientSet() clientset.Interface {
	return f.clientSet
}

// SharedInformerFactory returns a shared informer factory.
func (f *framework) SharedInformerFactory() informers.SharedInformerFactory {
	return f.informerFactory
}

func (f *framework) pluginsNeeded(plugins *config.Plugins) map[string]config.Plugin {
	pgMap := make(map[string]config.Plugin)

	if plugins == nil {
		return pgMap
	}

	find := func(pgs *config.PluginSet) {
		if pgs == nil {
			return
		}
		for _, pg := range pgs.Enabled {
			pgMap[pg.Name] = pg
		}
	}
	for _, e := range f.getExtensionPoints(plugins) {
		find(e.plugins)
	}
	return pgMap
}

func recordExtensionPointDuration(start time.Time, extensionPoint string, status *Status) {
	statusCode := Success.String()
	if status != nil {
		statusCode = status.Code().String()
	}
	metrics.FrameworkExtensionPointDuration.WithLabelValues(extensionPoint, statusCode).Observe(metrics.SinceInSeconds(start))
}
