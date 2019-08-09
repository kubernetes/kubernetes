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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// framework is the component responsible for initializing and running scheduler
// plugins.
type framework struct {
	registry                  Registry
	nodeInfoSnapshot          *cache.NodeInfoSnapshot
	waitingPods               *waitingPodsMap
	pluginNameToWeightMap     map[string]int
	queueSortPlugins          []QueueSortPlugin
	prefilterPlugins          []PrefilterPlugin
	filterPlugins             []FilterPlugin
	postFilterPlugins         []PostFilterPlugin
	scorePlugins              []ScorePlugin
	scoreWithNormalizePlugins []ScoreWithNormalizePlugin
	reservePlugins            []ReservePlugin
	prebindPlugins            []PrebindPlugin
	bindPlugins               []BindPlugin
	postbindPlugins           []PostbindPlugin
	unreservePlugins          []UnreservePlugin
	permitPlugins             []PermitPlugin
}

const (
	// Specifies the maximum timeout a permit plugin can return.
	maxTimeout time.Duration = 15 * time.Minute
)

var _ = Framework(&framework{})

// NewFramework initializes plugins given the configuration and the registry.
func NewFramework(r Registry, plugins *config.Plugins, args []config.PluginConfig) (Framework, error) {
	f := &framework{
		registry:              r,
		nodeInfoSnapshot:      cache.NewNodeInfoSnapshot(),
		pluginNameToWeightMap: make(map[string]int),
		waitingPods:           newWaitingPodsMap(),
	}
	if plugins == nil {
		return f, nil
	}

	// get needed plugins from config
	pg := pluginsNeeded(plugins)
	if len(pg) == 0 {
		return f, nil
	}

	pluginConfig := pluginNameToConfig(args)
	pluginsMap := make(map[string]Plugin)
	for name, factory := range r {
		// initialize only needed plugins
		if _, ok := pg[name]; !ok {
			continue
		}

		// find the config args of a plugin
		pc := pluginConfig[name]

		p, err := factory(pc, f)
		if err != nil {
			return nil, fmt.Errorf("error initializing plugin %q: %v", name, err)
		}
		pluginsMap[name] = p

		// A weight of zero is not permitted, plugins can be disabled explicitly
		// when configured.
		f.pluginNameToWeightMap[name] = int(pg[name].Weight)
		if f.pluginNameToWeightMap[name] == 0 {
			f.pluginNameToWeightMap[name] = 1
		}
	}

	if plugins.PreFilter != nil {
		for _, pf := range plugins.PreFilter.Enabled {
			if pg, ok := pluginsMap[pf.Name]; ok {
				p, ok := pg.(PrefilterPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend prefilter plugin", pf.Name)
				}
				f.prefilterPlugins = append(f.prefilterPlugins, p)
			} else {
				return nil, fmt.Errorf("prefilter plugin %q does not exist", pf.Name)
			}
		}
	}

	if plugins.Filter != nil {
		for _, r := range plugins.Filter.Enabled {
			if pg, ok := pluginsMap[r.Name]; ok {
				p, ok := pg.(FilterPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend filter plugin", r.Name)
				}
				f.filterPlugins = append(f.filterPlugins, p)
			} else {
				return nil, fmt.Errorf("filter plugin %q does not exist", r.Name)
			}
		}
	}

	if plugins.Score != nil {
		for _, sc := range plugins.Score.Enabled {
			if pg, ok := pluginsMap[sc.Name]; ok {
				// First, make sure the plugin implements ScorePlugin interface.
				p, ok := pg.(ScorePlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend score plugin", sc.Name)
				}
				if f.pluginNameToWeightMap[p.Name()] == 0 {
					return nil, fmt.Errorf("score plugin %q is not configured with weight", p.Name())
				}
				f.scorePlugins = append(f.scorePlugins, p)

				// Next, if the plugin also implements ScoreWithNormalizePlugin interface,
				// add it to the normalizeScore plugin list.
				np, ok := pg.(ScoreWithNormalizePlugin)
				if ok {
					f.scoreWithNormalizePlugins = append(f.scoreWithNormalizePlugins, np)
				}
			} else {
				return nil, fmt.Errorf("score plugin %q does not exist", sc.Name)
			}
		}
	}

	if plugins.Reserve != nil {
		for _, r := range plugins.Reserve.Enabled {
			if pg, ok := pluginsMap[r.Name]; ok {
				p, ok := pg.(ReservePlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend reserve plugin", r.Name)
				}
				f.reservePlugins = append(f.reservePlugins, p)
			} else {
				return nil, fmt.Errorf("reserve plugin %q does not exist", r.Name)
			}
		}
	}

	if plugins.PostFilter != nil {
		for _, r := range plugins.PostFilter.Enabled {
			if pg, ok := pluginsMap[r.Name]; ok {
				p, ok := pg.(PostFilterPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend post-filter plugin", r.Name)
				}
				f.postFilterPlugins = append(f.postFilterPlugins, p)
			} else {
				return nil, fmt.Errorf("post-filter plugin %q does not exist", r.Name)
			}
		}
	}

	if plugins.PreBind != nil {
		for _, pb := range plugins.PreBind.Enabled {
			if pg, ok := pluginsMap[pb.Name]; ok {
				p, ok := pg.(PrebindPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend prebind plugin", pb.Name)
				}
				f.prebindPlugins = append(f.prebindPlugins, p)
			} else {
				return nil, fmt.Errorf("prebind plugin %q does not exist", pb.Name)
			}
		}
	}

	if plugins.Bind != nil {
		for _, pb := range plugins.Bind.Enabled {
			if pg, ok := pluginsMap[pb.Name]; ok {
				p, ok := pg.(BindPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend bind plugin", pb.Name)
				}
				f.bindPlugins = append(f.bindPlugins, p)
			} else {
				return nil, fmt.Errorf("bind plugin %q does not exist", pb.Name)
			}
		}
	}

	if plugins.PostBind != nil {
		for _, pb := range plugins.PostBind.Enabled {
			if pg, ok := pluginsMap[pb.Name]; ok {
				p, ok := pg.(PostbindPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend postbind plugin", pb.Name)
				}
				f.postbindPlugins = append(f.postbindPlugins, p)
			} else {
				return nil, fmt.Errorf("postbind plugin %q does not exist", pb.Name)
			}
		}
	}

	if plugins.Unreserve != nil {
		for _, ur := range plugins.Unreserve.Enabled {
			if pg, ok := pluginsMap[ur.Name]; ok {
				p, ok := pg.(UnreservePlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend unreserve plugin", ur.Name)
				}
				f.unreservePlugins = append(f.unreservePlugins, p)
			} else {
				return nil, fmt.Errorf("unreserve plugin %q does not exist", ur.Name)
			}
		}
	}

	if plugins.Permit != nil {
		for _, pr := range plugins.Permit.Enabled {
			if pg, ok := pluginsMap[pr.Name]; ok {
				p, ok := pg.(PermitPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend permit plugin", pr.Name)
				}
				f.permitPlugins = append(f.permitPlugins, p)
			} else {
				return nil, fmt.Errorf("permit plugin %q does not exist", pr.Name)
			}
		}
	}

	if plugins.QueueSort != nil {
		for _, qs := range plugins.QueueSort.Enabled {
			if pg, ok := pluginsMap[qs.Name]; ok {
				p, ok := pg.(QueueSortPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %q does not extend queue sort plugin", qs.Name)
				}
				f.queueSortPlugins = append(f.queueSortPlugins, p)
				if len(f.queueSortPlugins) > 1 {
					return nil, fmt.Errorf("only one queue sort plugin can be enabled")
				}
			} else {
				return nil, fmt.Errorf("queue sort plugin %q does not exist", qs.Name)
			}
		}
	}

	return f, nil
}

// QueueSortFunc returns the function to sort pods in scheduling queue
func (f *framework) QueueSortFunc() LessFunc {
	if len(f.queueSortPlugins) == 0 {
		return nil
	}

	// Only one QueueSort plugin can be enabled.
	return f.queueSortPlugins[0].Less
}

// RunPrefilterPlugins runs the set of configured prefilter plugins. It returns
// *Status and its code is set to non-success if any of the plugins returns
// anything but Success. If a non-success status is returned, then the scheduling
// cycle is aborted.
func (f *framework) RunPrefilterPlugins(
	pc *PluginContext, pod *v1.Pod) *Status {
	for _, pl := range f.prefilterPlugins {
		status := pl.Prefilter(pc, pod)
		if !status.IsSuccess() {
			if status.Code() == Unschedulable {
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

// RunFilterPlugins runs the set of configured Filter plugins for pod on
// the given node. If any of these plugins doesn't return "Success", the
// given node is not suitable for running pod.
// Meanwhile, the failure message and status are set for the given node.
func (f *framework) RunFilterPlugins(pc *PluginContext,
	pod *v1.Pod, nodeName string) *Status {
	for _, pl := range f.filterPlugins {
		status := pl.Filter(pc, pod, nodeName)
		if !status.IsSuccess() {
			if status.Code() != Unschedulable {
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
	pc *PluginContext,
	pod *v1.Pod,
	nodes []*v1.Node,
	filteredNodesStatuses NodeToStatusMap,
) *Status {
	for _, pl := range f.postFilterPlugins {
		status := pl.PostFilter(pc, pod, nodes, filteredNodesStatuses)
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
func (f *framework) RunScorePlugins(pc *PluginContext, pod *v1.Pod, nodes []*v1.Node) (PluginToNodeScores, *Status) {
	pluginToNodeScores := make(PluginToNodeScores, len(f.scorePlugins))
	for _, pl := range f.scorePlugins {
		pluginToNodeScores[pl.Name()] = make(NodeScoreList, len(nodes))
	}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := schedutil.NewErrorChannel()
	workqueue.ParallelizeUntil(ctx, 16, len(nodes), func(index int) {
		for _, pl := range f.scorePlugins {
			nodeName := nodes[index].Name
			score, status := pl.Score(pc, pod, nodeName)
			if !status.IsSuccess() {
				errCh.SendErrorWithCancel(fmt.Errorf(status.Message()), cancel)
				return
			}
			pluginToNodeScores[pl.Name()][index] = NodeScore{
				Name:  nodeName,
				Score: score,
			}
		}
	})

	if err := errCh.ReceiveError(); err != nil {
		msg := fmt.Sprintf("error while running score plugin for pod %q: %v", pod.Name, err)
		klog.Error(msg)
		return nil, NewStatus(Error, msg)
	}

	return pluginToNodeScores, nil
}

// RunNormalizeScorePlugins runs the NormalizeScore function of Score plugins.
// It should be called after RunScorePlugins with the PluginToNodeScores result.
// It then modifies the list with normalized scores. It returns a non-success Status
// if any of the NormalizeScore functions returns a non-success status.
func (f *framework) RunNormalizeScorePlugins(pc *PluginContext, pod *v1.Pod, scores PluginToNodeScores) *Status {
	ctx, cancel := context.WithCancel(context.Background())
	errCh := schedutil.NewErrorChannel()
	workqueue.ParallelizeUntil(ctx, 16, len(f.scoreWithNormalizePlugins), func(index int) {
		pl := f.scoreWithNormalizePlugins[index]
		nodeScoreList, ok := scores[pl.Name()]
		if !ok {
			err := fmt.Errorf("normalize score plugin %q has no corresponding scores in the PluginToNodeScores", pl.Name())
			errCh.SendErrorWithCancel(err, cancel)
			return
		}
		status := pl.NormalizeScore(pc, pod, nodeScoreList)
		if !status.IsSuccess() {
			err := fmt.Errorf("normalize score plugin %q failed with error %v", pl.Name(), status.Message())
			errCh.SendErrorWithCancel(err, cancel)
			return
		}
	})

	if err := errCh.ReceiveError(); err != nil {
		msg := fmt.Sprintf("error while running normalize score plugin for pod %q: %v", pod.Name, err)
		klog.Error(msg)
		return NewStatus(Error, msg)
	}

	return nil
}

// ApplyScoreWeights applies weights to the score results. It should be called after
// RunNormalizeScorePlugins.
func (f *framework) ApplyScoreWeights(pc *PluginContext, pod *v1.Pod, scores PluginToNodeScores) *Status {
	ctx, cancel := context.WithCancel(context.Background())
	errCh := schedutil.NewErrorChannel()
	workqueue.ParallelizeUntil(ctx, 16, len(f.scorePlugins), func(index int) {
		pl := f.scorePlugins[index]
		// Score plugins' weight has been checked when they are initialized.
		weight := f.pluginNameToWeightMap[pl.Name()]
		nodeScoreList, ok := scores[pl.Name()]
		if !ok {
			err := fmt.Errorf("score plugin %q has no corresponding scores in the PluginToNodeScores", pl.Name())
			errCh.SendErrorWithCancel(err, cancel)
			return
		}

		for i, nodeScore := range nodeScoreList {
			// return error if score plugin returns invalid score.
			if nodeScore.Score > MaxNodeScore || nodeScore.Score < MinNodeScore {
				err := fmt.Errorf("score plugin %q returns an invalid score %q, it should in the range of [MinNodeScore, MaxNodeScore] after normalizing", pl.Name(), nodeScore.Score)
				errCh.SendErrorWithCancel(err, cancel)
				return
			}

			nodeScoreList[i].Score = nodeScore.Score * weight
		}
	})

	if err := errCh.ReceiveError(); err != nil {
		msg := fmt.Sprintf("error while applying score weights for pod %q: %v", pod.Name, err)
		klog.Error(msg)
		return NewStatus(Error, msg)
	}

	return nil
}

// RunPrebindPlugins runs the set of configured prebind plugins. It returns a
// failure (bool) if any of the plugins returns an error. It also returns an
// error containing the rejection message or the error occurred in the plugin.
func (f *framework) RunPrebindPlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	for _, pl := range f.prebindPlugins {
		status := pl.Prebind(pc, pod, nodeName)
		if !status.IsSuccess() {
			if status.Code() == Unschedulable {
				msg := fmt.Sprintf("rejected by %q at prebind: %v", pl.Name(), status.Message())
				klog.V(4).Infof(msg)
				return NewStatus(status.Code(), msg)
			}
			msg := fmt.Sprintf("error while running %q prebind plugin for pod %q: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}
	return nil
}

// RunBindPlugins runs the set of configured bind plugins until one returns a non `Skip` status.
func (f *framework) RunBindPlugins(pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	if len(f.bindPlugins) == 0 {
		return NewStatus(Skip, "")
	}
	var status *Status
	for _, bp := range f.bindPlugins {
		status = bp.Bind(pc, pod, nodeName)
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

// RunPostbindPlugins runs the set of configured postbind plugins.
func (f *framework) RunPostbindPlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) {
	for _, pl := range f.postbindPlugins {
		pl.Postbind(pc, pod, nodeName)
	}
}

// RunReservePlugins runs the set of configured reserve plugins. If any of these
// plugins returns an error, it does not continue running the remaining ones and
// returns the error. In such case, pod will not be scheduled.
func (f *framework) RunReservePlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	for _, pl := range f.reservePlugins {
		status := pl.Reserve(pc, pod, nodeName)
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
	pc *PluginContext, pod *v1.Pod, nodeName string) {
	for _, pl := range f.unreservePlugins {
		pl.Unreserve(pc, pod, nodeName)
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
	pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	timeout := maxTimeout
	statusCode := Success
	for _, pl := range f.permitPlugins {
		status, d := pl.Permit(pc, pod, nodeName)
		if !status.IsSuccess() {
			if status.Code() == Unschedulable {
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
				if s.Code() == Unschedulable {
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
func (f *framework) NodeInfoSnapshot() *cache.NodeInfoSnapshot {
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

func pluginNameToConfig(args []config.PluginConfig) map[string]*runtime.Unknown {
	pc := make(map[string]*runtime.Unknown, 0)
	for _, p := range args {
		pc[p.Name] = &p.Args
	}
	return pc
}

func pluginsNeeded(plugins *config.Plugins) map[string]config.Plugin {
	pgMap := make(map[string]config.Plugin, 0)

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
	find(plugins.QueueSort)
	find(plugins.PreFilter)
	find(plugins.Filter)
	find(plugins.PostFilter)
	find(plugins.Score)
	find(plugins.Reserve)
	find(plugins.Permit)
	find(plugins.PreBind)
	find(plugins.Bind)
	find(plugins.PostBind)
	find(plugins.Unreserve)

	return pgMap
}
