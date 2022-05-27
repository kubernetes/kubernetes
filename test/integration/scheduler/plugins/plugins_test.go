/*
Copyright 2018 The Kubernetes Authors.

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

package plugins

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/kube-scheduler/config/v1beta3"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"
)

// imported from testutils
var (
	createPausePod                  = testutils.CreatePausePod
	initPausePod                    = testutils.InitPausePod
	getPod                          = testutils.GetPod
	deletePod                       = testutils.DeletePod
	podUnschedulable                = testutils.PodUnschedulable
	podSchedulingError              = testutils.PodSchedulingError
	createAndWaitForNodesInCache    = testutils.CreateAndWaitForNodesInCache
	waitForPodUnschedulable         = testutils.WaitForPodUnschedulable
	waitForPodToScheduleWithTimeout = testutils.WaitForPodToScheduleWithTimeout
)

type PreFilterPlugin struct {
	numPreFilterCalled int
	failPreFilter      bool
	rejectPreFilter    bool
}

type ScorePlugin struct {
	failScore      bool
	numScoreCalled int32
	highScoreNode  string
}

type ScoreWithNormalizePlugin struct {
	numScoreCalled          int
	numNormalizeScoreCalled int
}

type FilterPlugin struct {
	numFilterCalled int32
	failFilter      bool
	rejectFilter    bool

	numCalledPerPod map[string]int
	sync.RWMutex
}

type PostFilterPlugin struct {
	fh                  framework.Handle
	numPostFilterCalled int
	failPostFilter      bool
	rejectPostFilter    bool
}

type ReservePlugin struct {
	name                  string
	numReserveCalled      int
	failReserve           bool
	numUnreserveCalled    int
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PreScorePlugin struct {
	numPreScoreCalled int
	failPreScore      bool
}

type PreBindPlugin struct {
	numPreBindCalled int
	failPreBind      bool
	rejectPreBind    bool
	// If set to true, always succeed on non-first scheduling attempt.
	succeedOnRetry bool
	// Record the pod UIDs that have been tried scheduling.
	podUIDs map[types.UID]struct{}
}

type BindPlugin struct {
	name                  string
	numBindCalled         int
	bindStatus            *framework.Status
	client                clientset.Interface
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PostBindPlugin struct {
	name                  string
	numPostBindCalled     int
	pluginInvokeEventChan chan pluginInvokeEvent
}

type PermitPlugin struct {
	name                string
	numPermitCalled     int
	failPermit          bool
	rejectPermit        bool
	timeoutPermit       bool
	waitAndRejectPermit bool
	waitAndAllowPermit  bool
	cancelled           bool
	waitingPod          string
	rejectingPod        string
	allowingPod         string
	fh                  framework.Handle
}

const (
	prefilterPluginName          = "prefilter-plugin"
	postfilterPluginName         = "postfilter-plugin"
	scorePluginName              = "score-plugin"
	scoreWithNormalizePluginName = "score-with-normalize-plugin"
	filterPluginName             = "filter-plugin"
	preScorePluginName           = "prescore-plugin"
	reservePluginName            = "reserve-plugin"
	preBindPluginName            = "prebind-plugin"
	postBindPluginName           = "postbind-plugin"
	permitPluginName             = "permit-plugin"
)

var _ framework.PreFilterPlugin = &PreFilterPlugin{}
var _ framework.PostFilterPlugin = &PostFilterPlugin{}
var _ framework.ScorePlugin = &ScorePlugin{}
var _ framework.FilterPlugin = &FilterPlugin{}
var _ framework.ScorePlugin = &ScorePlugin{}
var _ framework.ScorePlugin = &ScoreWithNormalizePlugin{}
var _ framework.ReservePlugin = &ReservePlugin{}
var _ framework.PreScorePlugin = &PreScorePlugin{}
var _ framework.PreBindPlugin = &PreBindPlugin{}
var _ framework.BindPlugin = &BindPlugin{}
var _ framework.PostBindPlugin = &PostBindPlugin{}
var _ framework.PermitPlugin = &PermitPlugin{}

// newPlugin returns a plugin factory with specified Plugin.
func newPlugin(plugin framework.Plugin) frameworkruntime.PluginFactory {
	return func(_ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
		switch pl := plugin.(type) {
		case *PermitPlugin:
			pl.fh = fh
		case *PostFilterPlugin:
			pl.fh = fh
		}
		return plugin, nil
	}
}

// Name returns name of the score plugin.
func (sp *ScorePlugin) Name() string {
	return scorePluginName
}

// reset returns name of the score plugin.
func (sp *ScorePlugin) reset() {
	sp.failScore = false
	sp.numScoreCalled = 0
	sp.highScoreNode = ""
}

// Score returns the score of scheduling a pod on a specific node.
func (sp *ScorePlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	curCalled := atomic.AddInt32(&sp.numScoreCalled, 1)
	if sp.failScore {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", p.Name))
	}

	score := int64(1)
	if curCalled == 1 {
		// The first node is scored the highest, the rest is scored lower.
		sp.highScoreNode = nodeName
		score = framework.MaxNodeScore
	}
	return score, nil
}

func (sp *ScorePlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// Name returns name of the score plugin.
func (sp *ScoreWithNormalizePlugin) Name() string {
	return scoreWithNormalizePluginName
}

// reset returns name of the score plugin.
func (sp *ScoreWithNormalizePlugin) reset() {
	sp.numScoreCalled = 0
	sp.numNormalizeScoreCalled = 0
}

// Score returns the score of scheduling a pod on a specific node.
func (sp *ScoreWithNormalizePlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	sp.numScoreCalled++
	score := int64(10)
	return score, nil
}

func (sp *ScoreWithNormalizePlugin) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	sp.numNormalizeScoreCalled++
	return nil
}

func (sp *ScoreWithNormalizePlugin) ScoreExtensions() framework.ScoreExtensions {
	return sp
}

// Name returns name of the plugin.
func (fp *FilterPlugin) Name() string {
	return filterPluginName
}

// reset is used to reset filter plugin.
func (fp *FilterPlugin) reset() {
	fp.numFilterCalled = 0
	fp.failFilter = false
	if fp.numCalledPerPod != nil {
		fp.numCalledPerPod = make(map[string]int)
	}
}

// Filter is a test function that returns an error or nil, depending on the
// value of "failFilter".
func (fp *FilterPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	atomic.AddInt32(&fp.numFilterCalled, 1)

	if fp.numCalledPerPod != nil {
		fp.Lock()
		fp.numCalledPerPod[fmt.Sprintf("%v/%v", pod.Namespace, pod.Name)]++
		fp.Unlock()
	}

	if fp.failFilter {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if fp.rejectFilter {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}

	return nil
}

// Name returns name of the plugin.
func (rp *ReservePlugin) Name() string {
	return rp.name
}

// Reserve is a test function that increments an intenral counter and returns
// an error or nil, depending on the value of "failReserve".
func (rp *ReservePlugin) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	rp.numReserveCalled++
	if rp.failReserve {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	return nil
}

// Unreserve is a test function that increments an internal counter and emits
// an event to a channel. While Unreserve implementations should normally be
// idempotent, we relax that requirement here for testing purposes.
func (rp *ReservePlugin) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	rp.numUnreserveCalled++
	if rp.pluginInvokeEventChan != nil {
		rp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: rp.Name(), val: rp.numUnreserveCalled}
	}
}

// reset used to reset internal counters.
func (rp *ReservePlugin) reset() {
	rp.numReserveCalled = 0
	rp.numUnreserveCalled = 0
	rp.failReserve = false
}

// Name returns name of the plugin.
func (*PreScorePlugin) Name() string {
	return preScorePluginName
}

// PreScore is a test function.
func (pfp *PreScorePlugin) PreScore(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, _ []*v1.Node) *framework.Status {
	pfp.numPreScoreCalled++
	if pfp.failPreScore {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// reset used to reset prescore plugin.
func (pfp *PreScorePlugin) reset() {
	pfp.numPreScoreCalled = 0
	pfp.failPreScore = false
}

// Name returns name of the plugin.
func (pp *PreBindPlugin) Name() string {
	return preBindPluginName
}

// PreBind is a test function that returns (true, nil) or errors for testing.
func (pp *PreBindPlugin) PreBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	pp.numPreBindCalled++
	if _, tried := pp.podUIDs[pod.UID]; tried && pp.succeedOnRetry {
		return nil
	}
	pp.podUIDs[pod.UID] = struct{}{}
	if pp.failPreBind {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPreBind {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil
}

// reset used to reset prebind plugin.
func (pp *PreBindPlugin) reset() {
	pp.numPreBindCalled = 0
	pp.failPreBind = false
	pp.rejectPreBind = false
	pp.succeedOnRetry = false
	pp.podUIDs = make(map[types.UID]struct{})
}

const bindPluginAnnotation = "bindPluginName"

func (bp *BindPlugin) Name() string {
	return bp.name
}

func (bp *BindPlugin) Bind(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) *framework.Status {
	bp.numBindCalled++
	if bp.pluginInvokeEventChan != nil {
		bp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: bp.Name(), val: bp.numBindCalled}
	}
	if bp.bindStatus.IsSuccess() {
		if err := bp.client.CoreV1().Pods(p.Namespace).Bind(context.TODO(), &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Namespace: p.Namespace, Name: p.Name, UID: p.UID, Annotations: map[string]string{bindPluginAnnotation: bp.Name()}},
			Target: v1.ObjectReference{
				Kind: "Node",
				Name: nodeName,
			},
		}, metav1.CreateOptions{}); err != nil {
			return framework.NewStatus(framework.Error, fmt.Sprintf("bind failed: %v", err))
		}
	}
	return bp.bindStatus
}

// reset used to reset numBindCalled.
func (bp *BindPlugin) reset() {
	bp.numBindCalled = 0
}

// Name returns name of the plugin.
func (pp *PostBindPlugin) Name() string {
	return pp.name
}

// PostBind is a test function, which counts the number of times called.
func (pp *PostBindPlugin) PostBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	pp.numPostBindCalled++
	if pp.pluginInvokeEventChan != nil {
		pp.pluginInvokeEventChan <- pluginInvokeEvent{pluginName: pp.Name(), val: pp.numPostBindCalled}
	}
}

// reset used to reset postbind plugin.
func (pp *PostBindPlugin) reset() {
	pp.numPostBindCalled = 0
}

// Name returns name of the plugin.
func (pp *PreFilterPlugin) Name() string {
	return prefilterPluginName
}

// Extensions returns the PreFilterExtensions interface.
func (pp *PreFilterPlugin) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// PreFilter is a test function that returns (true, nil) or errors for testing.
func (pp *PreFilterPlugin) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	pp.numPreFilterCalled++
	if pp.failPreFilter {
		return nil, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPreFilter {
		return nil, framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil, nil
}

// reset used to reset prefilter plugin.
func (pp *PreFilterPlugin) reset() {
	pp.numPreFilterCalled = 0
	pp.failPreFilter = false
	pp.rejectPreFilter = false
}

// Name returns name of the plugin.
func (pp *PostFilterPlugin) Name() string {
	return postfilterPluginName
}

func (pp *PostFilterPlugin) PostFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, _ framework.NodeToStatusMap) (*framework.PostFilterResult, *framework.Status) {
	pp.numPostFilterCalled++
	nodeInfos, err := pp.fh.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return nil, framework.NewStatus(framework.Error, err.Error())
	}

	for _, nodeInfo := range nodeInfos {
		pp.fh.RunFilterPlugins(ctx, state, pod, nodeInfo)
	}
	var nodes []*v1.Node
	for _, nodeInfo := range nodeInfos {
		nodes = append(nodes, nodeInfo.Node())
	}
	pp.fh.RunScorePlugins(ctx, state, pod, nodes)

	if pp.failPostFilter {
		return nil, framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}
	if pp.rejectPostFilter {
		return nil, framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name))
	}
	return nil, framework.NewStatus(framework.Success, fmt.Sprintf("make room for pod %v to be schedulable", pod.Name))
}

// Name returns name of the plugin.
func (pp *PermitPlugin) Name() string {
	return pp.name
}

// Permit implements the permit test plugin.
func (pp *PermitPlugin) Permit(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (*framework.Status, time.Duration) {
	pp.numPermitCalled++
	if pp.failPermit {
		return framework.NewStatus(framework.Error, fmt.Sprintf("injecting failure for pod %v", pod.Name)), 0
	}
	if pp.rejectPermit {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name)), 0
	}
	if pp.timeoutPermit {
		go func() {
			select {
			case <-ctx.Done():
				pp.cancelled = true
			}
		}()
		return framework.NewStatus(framework.Wait, ""), 3 * time.Second
	}
	if pp.waitAndRejectPermit || pp.waitAndAllowPermit {
		if pp.waitingPod == "" || pp.waitingPod == pod.Name {
			pp.waitingPod = pod.Name
			return framework.NewStatus(framework.Wait, ""), 30 * time.Second
		}
		if pp.waitAndRejectPermit {
			pp.rejectingPod = pod.Name
			pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) {
				wp.Reject(pp.name, fmt.Sprintf("reject pod %v", wp.GetPod().Name))
			})
			return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("reject pod %v", pod.Name)), 0
		}
		if pp.waitAndAllowPermit {
			pp.allowingPod = pod.Name
			pp.allowAllPods()
			return nil, 0
		}
	}
	return nil, 0
}

// allowAllPods allows all waiting pods.
func (pp *PermitPlugin) allowAllPods() {
	pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { wp.Allow(pp.name) })
}

// rejectAllPods rejects all waiting pods.
func (pp *PermitPlugin) rejectAllPods() {
	pp.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { wp.Reject(pp.name, "rejectAllPods") })
}

// reset used to reset permit plugin.
func (pp *PermitPlugin) reset() {
	pp.numPermitCalled = 0
	pp.failPermit = false
	pp.rejectPermit = false
	pp.timeoutPermit = false
	pp.waitAndRejectPermit = false
	pp.waitAndAllowPermit = false
	pp.cancelled = false
	pp.waitingPod = ""
	pp.allowingPod = ""
	pp.rejectingPod = ""
}

// TestPreFilterPlugin tests invocation of prefilter plugins.
func TestPreFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a pre-filter plugin.
	preFilterPlugin := &PreFilterPlugin{}
	registry, prof := initRegistryAndConfig(t, preFilterPlugin)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "prefilter-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name   string
		fail   bool
		reject bool
	}{
		{
			name:   "disable fail and reject flags",
			fail:   false,
			reject: false,
		},
		{
			name:   "enable fail and disable reject flags",
			fail:   true,
			reject: false,
		},
		{
			name:   "disable fail and enable reject flags",
			fail:   false,
			reject: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			preFilterPlugin.failPreFilter = test.fail
			preFilterPlugin.rejectPreFilter = test.reject
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.reject || test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
			}

			if preFilterPlugin.numPreFilterCalled == 0 {
				t.Errorf("Expected the prefilter plugin to be called.")
			}

			preFilterPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPostFilterPlugin tests invocation of postfilter plugins.
func TestPostFilterPlugin(t *testing.T) {
	var numNodes int32 = 1
	tests := []struct {
		name                      string
		numNodes                  int32
		rejectFilter              bool
		failScore                 bool
		rejectPostFilter          bool
		expectFilterNumCalled     int32
		expectScoreNumCalled      int32
		expectPostFilterNumCalled int
	}{
		{
			name:                      "Filter passed and Score success",
			numNodes:                  30,
			rejectFilter:              false,
			failScore:                 false,
			rejectPostFilter:          false,
			expectFilterNumCalled:     30,
			expectScoreNumCalled:      30,
			expectPostFilterNumCalled: 0,
		},
		{
			name:                      "Filter failed and PostFilter passed",
			numNodes:                  numNodes,
			rejectFilter:              true,
			failScore:                 false,
			rejectPostFilter:          false,
			expectFilterNumCalled:     numNodes * 2,
			expectScoreNumCalled:      1,
			expectPostFilterNumCalled: 1,
		},
		{
			name:                      "Filter failed and PostFilter failed",
			numNodes:                  numNodes,
			rejectFilter:              true,
			failScore:                 false,
			rejectPostFilter:          true,
			expectFilterNumCalled:     numNodes * 2,
			expectScoreNumCalled:      1,
			expectPostFilterNumCalled: 1,
		},
		{
			name:                      "Score failed and PostFilter failed",
			numNodes:                  numNodes,
			rejectFilter:              true,
			failScore:                 true,
			rejectPostFilter:          true,
			expectFilterNumCalled:     numNodes * 2,
			expectScoreNumCalled:      1,
			expectPostFilterNumCalled: 1,
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a plugin registry for testing. Register a combination of filter and postFilter plugin.
			var (
				filterPlugin     = &FilterPlugin{}
				scorePlugin      = &ScorePlugin{}
				postFilterPlugin = &PostFilterPlugin{}
			)
			filterPlugin.rejectFilter = tt.rejectFilter
			scorePlugin.failScore = tt.failScore
			postFilterPlugin.rejectPostFilter = tt.rejectPostFilter
			registry := frameworkruntime.Registry{
				filterPluginName:     newPlugin(filterPlugin),
				scorePluginName:      newPlugin(scorePlugin),
				postfilterPluginName: newPlugin(postFilterPlugin),
			}

			// Setup plugins for testing.
			cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
				Profiles: []v1beta3.KubeSchedulerProfile{{
					SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
					Plugins: &v1beta3.Plugins{
						Filter: v1beta3.PluginSet{
							Enabled: []v1beta3.Plugin{
								{Name: filterPluginName},
							},
						},
						Score: v1beta3.PluginSet{
							Enabled: []v1beta3.Plugin{
								{Name: scorePluginName},
							},
							// disable default in-tree Score plugins
							// to make it easy to control configured ScorePlugins failure
							Disabled: []v1beta3.Plugin{
								{Name: "*"},
							},
						},
						PostFilter: v1beta3.PluginSet{
							Enabled: []v1beta3.Plugin{
								{Name: postfilterPluginName},
							},
							// Need to disable default in-tree PostFilter plugins, as they will
							// call RunFilterPlugins and hence impact the "numFilterCalled".
							Disabled: []v1beta3.Plugin{
								{Name: "*"},
							},
						},
					},
				}}})

			// Create the API server and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(
				t,
				testutils.InitTestAPIServer(t, fmt.Sprintf("postfilter%v-", i), nil),
				int(tt.numNodes),
				scheduler.WithProfiles(cfg.Profiles...),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
			)
			defer testutils.CleanupTest(t, testCtx)

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet, initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if tt.rejectFilter {
				if err = wait.Poll(10*time.Millisecond, 10*time.Second, podUnschedulable(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled.")
				}

				if numFilterCalled := atomic.LoadInt32(&filterPlugin.numFilterCalled); numFilterCalled < tt.expectFilterNumCalled {
					t.Errorf("Expected the filter plugin to be called at least %v times, but got %v.", tt.expectFilterNumCalled, numFilterCalled)
				}
				if numScoreCalled := atomic.LoadInt32(&scorePlugin.numScoreCalled); numScoreCalled < tt.expectScoreNumCalled {
					t.Errorf("Expected the score plugin to be called at least %v times, but got %v.", tt.expectScoreNumCalled, numScoreCalled)
				}
				if postFilterPlugin.numPostFilterCalled < tt.expectPostFilterNumCalled {
					t.Errorf("Expected the postfilter plugin to be called at least %v times, but got %v.", tt.expectPostFilterNumCalled, postFilterPlugin.numPostFilterCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
				if numFilterCalled := atomic.LoadInt32(&filterPlugin.numFilterCalled); numFilterCalled != tt.expectFilterNumCalled {
					t.Errorf("Expected the filter plugin to be called %v times, but got %v.", tt.expectFilterNumCalled, numFilterCalled)
				}
				if numScoreCalled := atomic.LoadInt32(&scorePlugin.numScoreCalled); numScoreCalled != tt.expectScoreNumCalled {
					t.Errorf("Expected the score plugin to be called %v times, but got %v.", tt.expectScoreNumCalled, numScoreCalled)
				}
				if postFilterPlugin.numPostFilterCalled != tt.expectPostFilterNumCalled {
					t.Errorf("Expected the postfilter plugin to be called %v times, but got %v.", tt.expectPostFilterNumCalled, postFilterPlugin.numPostFilterCalled)
				}
			}
		})
	}
}

// TestScorePlugin tests invocation of score plugins.
func TestScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a score plugin.
	scorePlugin := &ScorePlugin{}
	registry, prof := initRegistryAndConfig(t, scorePlugin)

	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "score-plugin", nil), 10,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail score plugin",
			fail: true,
		},
		{
			name: "do not fail score plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scorePlugin.failScore = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Fatalf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				} else {
					p, err := getPod(testCtx.ClientSet, pod.Name, pod.Namespace)
					if err != nil {
						t.Errorf("Failed to retrieve the pod. error: %v", err)
					} else if p.Spec.NodeName != scorePlugin.highScoreNode {
						t.Errorf("Expected the pod to be scheduled on node %q, got %q", scorePlugin.highScoreNode, p.Spec.NodeName)
					}
				}
			}

			if numScoreCalled := atomic.LoadInt32(&scorePlugin.numScoreCalled); numScoreCalled == 0 {
				t.Errorf("Expected the score plugin to be called.")
			}

			scorePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestNormalizeScorePlugin tests invocation of normalize score plugins.
func TestNormalizeScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a normalize score plugin.
	scoreWithNormalizePlugin := &ScoreWithNormalizePlugin{}
	registry, prof := initRegistryAndConfig(t, scoreWithNormalizePlugin)

	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "score-plugin", nil), 10,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))

	defer testutils.CleanupTest(t, testCtx)

	// Create a best effort pod.
	pod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
	if err != nil {
		t.Fatalf("Error while creating a test pod: %v", err)
	}

	if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
		t.Errorf("Expected the pod to be scheduled. error: %v", err)
	}

	if scoreWithNormalizePlugin.numScoreCalled == 0 {
		t.Errorf("Expected the score plugin to be called.")
	}
	if scoreWithNormalizePlugin.numNormalizeScoreCalled == 0 {
		t.Error("Expected the normalize score plugin to be called")
	}

	scoreWithNormalizePlugin.reset()
}

// TestReservePlugin tests invocation of reserve plugins.
func TestReservePluginReserve(t *testing.T) {
	// Create a plugin registry for testing. Register only a reserve plugin.
	reservePlugin := &ReservePlugin{}
	registry, prof := initRegistryAndConfig(t, reservePlugin)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "reserve-plugin-reserve", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail reserve plugin",
			fail: true,
		},
		{
			name: "do not fail reserve plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reservePlugin.failReserve = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second,
					podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
			}

			if reservePlugin.numReserveCalled == 0 {
				t.Errorf("Expected the reserve plugin to be called.")
			}

			reservePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPrebindPlugin tests invocation of prebind plugins.
func TestPrebindPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register a prebind and a filter plugin.
	preBindPlugin := &PreBindPlugin{podUIDs: make(map[types.UID]struct{})}
	filterPlugin := &FilterPlugin{}
	registry := frameworkruntime.Registry{
		preBindPluginName: newPlugin(preBindPlugin),
		filterPluginName:  newPlugin(filterPlugin),
	}

	// Setup initial prebind and filter plugin in different profiles.
	// The second profile ensures the embedded filter plugin is exclusively called, and hence
	// we can use its internal `numFilterCalled` to perform some precise checking logic.
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{
			{
				SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
				Plugins: &v1beta3.Plugins{
					PreBind: v1beta3.PluginSet{
						Enabled: []v1beta3.Plugin{
							{Name: preBindPluginName},
						},
					},
				},
			},
			{
				SchedulerName: pointer.StringPtr("2nd-scheduler"),
				Plugins: &v1beta3.Plugins{
					Filter: v1beta3.PluginSet{
						Enabled: []v1beta3.Plugin{
							{Name: filterPluginName},
						},
					},
				},
			},
		},
	})

	// Create the API server and the scheduler with the test plugin set.
	nodesNum := 2
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "prebind-plugin", nil), nodesNum,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name             string
		fail             bool
		reject           bool
		succeedOnRetry   bool
		unschedulablePod *v1.Pod
	}{
		{
			name:   "disable fail and reject flags",
			fail:   false,
			reject: false,
		},
		{
			name:   "enable fail and disable reject flags",
			fail:   true,
			reject: false,
		},
		{
			name:   "disable fail and enable reject flags",
			fail:   false,
			reject: true,
		},
		{
			name:   "enable fail and reject flags",
			fail:   true,
			reject: true,
		},
		{
			name:           "fail on 1st try but succeed on retry",
			fail:           true,
			reject:         false,
			succeedOnRetry: true,
		},
		{
			name:           "reject on 1st try but succeed on retry",
			fail:           false,
			reject:         true,
			succeedOnRetry: true,
		},
		{
			name:             "failure on preBind moves unschedulable pods",
			fail:             true,
			unschedulablePod: st.MakePod().Name("unschedulable-pod").Namespace(testCtx.NS.Name).Container(imageutils.GetPauseImageName()).Obj(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if p := test.unschedulablePod; p != nil {
				p.Spec.SchedulerName = "2nd-scheduler"
				filterPlugin.rejectFilter = true
				if _, err := createPausePod(testCtx.ClientSet, p); err != nil {
					t.Fatalf("Error while creating an unschedulable pod: %v", err)
				}
				defer filterPlugin.reset()
			}

			preBindPlugin.failPreBind = test.fail
			preBindPlugin.rejectPreBind = test.reject
			preBindPlugin.succeedOnRetry = test.succeedOnRetry
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail || test.reject {
				if test.succeedOnRetry {
					if err = testutils.WaitForPodToScheduleWithTimeout(testCtx.ClientSet, pod, 10*time.Second); err != nil {
						t.Errorf("Expected the pod to be schedulable on retry, but got an error: %v", err)
					}
				} else if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
			} else if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
				t.Errorf("Expected the pod to be scheduled. error: %v", err)
			}

			if preBindPlugin.numPreBindCalled == 0 {
				t.Errorf("Expected the prebind plugin to be called.")
			}

			if test.unschedulablePod != nil {
				if err := wait.Poll(10*time.Millisecond, 15*time.Second, func() (bool, error) {
					// 2 means the unschedulable pod is expected to be retried at least twice.
					// (one initial attempt plus the one moved by the preBind pod)
					return int(filterPlugin.numFilterCalled) >= 2*nodesNum, nil
				}); err != nil {
					t.Errorf("Timed out waiting for the unschedulable Pod to be retried at least twice.")
				}
			}

			preBindPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestUnReserveReservePlugins tests invocation of the Unreserve operation in
// reserve plugins through failures in execution points such as pre-bind. Also
// tests that the order of invocation of Unreserve operation is executed in the
// reverse order of invocation of the Reserve operation.
func TestUnReserveReservePlugins(t *testing.T) {
	tests := []struct {
		name          string
		plugins       []*ReservePlugin
		fail          bool
		failPluginIdx int
	}{
		{
			name:          "The first Reserve plugin fails",
			failPluginIdx: 0,
			plugins: []*ReservePlugin{
				{
					name:        "failedReservePlugin1",
					failReserve: true,
				},
				{
					name:        "succeedReservePlugin",
					failReserve: false,
				},
				{
					name:        "failedReservePlugin2",
					failReserve: true,
				},
			},
			fail: true,
		},
		{
			name:          "The middle Reserve plugin fails",
			failPluginIdx: 1,
			plugins: []*ReservePlugin{
				{
					name:        "succeedReservePlugin1",
					failReserve: false,
				},
				{
					name:        "failedReservePlugin1",
					failReserve: true,
				},
				{
					name:        "succeedReservePlugin2",
					failReserve: false,
				},
			},
			fail: true,
		},
		{
			name:          "The last Reserve plugin fails",
			failPluginIdx: 2,
			plugins: []*ReservePlugin{
				{
					name:        "succeedReservePlugin1",
					failReserve: false,
				},
				{
					name:        "succeedReservePlugin2",
					failReserve: false,
				},
				{
					name:        "failedReservePlugin1",
					failReserve: true,
				},
			},
			fail: true,
		},
		{
			name:          "The Reserve plugins succeed",
			failPluginIdx: -1,
			plugins: []*ReservePlugin{
				{
					name:        "succeedReservePlugin1",
					failReserve: false,
				},
				{
					name:        "succeedReservePlugin2",
					failReserve: false,
				},
				{
					name:        "succeedReservePlugin3",
					failReserve: false,
				},
			},
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var pls []framework.Plugin
			for _, pl := range test.plugins {
				pls = append(pls, pl)
			}
			registry, prof := initRegistryAndConfig(t, pls...)

			// Create the API server and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(
				t,
				testutils.InitTestAPIServer(t, "unreserve-reserve-plugin", nil),
				2,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			// Create a best effort pod.
			podName := "test-pod"
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a reasons other than Unschedulable, but got: %v", err)
				}

				for i, pl := range test.plugins {
					if i <= test.failPluginIdx {
						if pl.numReserveCalled != 1 {
							t.Errorf("Reserve Plugins %s numReserveCalled = %d, want 1.", pl.name, pl.numReserveCalled)
						}
					} else {
						if pl.numReserveCalled != 0 {
							t.Errorf("Reserve Plugins %s numReserveCalled = %d, want 0.", pl.name, pl.numReserveCalled)
						}
					}

					if pl.numUnreserveCalled != 1 {
						t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 1.", pl.name, pl.numUnreserveCalled)
					}
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}

				for _, pl := range test.plugins {
					if pl.numReserveCalled != 1 {
						t.Errorf("Reserve Plugin %s numReserveCalled = %d, want 1.", pl.name, pl.numReserveCalled)
					}
					if pl.numUnreserveCalled != 0 {
						t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 0.", pl.name, pl.numUnreserveCalled)
					}
				}
			}
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestUnReservePermitPlugins tests unreserve of Permit plugins.
func TestUnReservePermitPlugins(t *testing.T) {
	tests := []struct {
		name   string
		plugin *PermitPlugin
		fail   bool
	}{
		{
			name: "All Reserve plugins passed, but a Permit plugin was rejected",
			fail: true,
			plugin: &PermitPlugin{
				name:         "rejectedPermitPlugin",
				rejectPermit: true,
			},
		},
		{
			name: "All Reserve plugins passed, but a Permit plugin timeout in waiting",
			fail: true,
			plugin: &PermitPlugin{
				name:          "timeoutPermitPlugin",
				timeoutPermit: true,
			},
		},
		{
			name: "The Permit plugin succeed",
			fail: false,
			plugin: &PermitPlugin{
				name: "succeedPermitPlugin",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reservePlugin := &ReservePlugin{
				name:        "reservePlugin",
				failReserve: false,
			}
			registry, profile := initRegistryAndConfig(t, []framework.Plugin{test.plugin, reservePlugin}...)

			// Create the API server and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(
				t,
				testutils.InitTestAPIServer(t, "unreserve-reserve-plugin", nil),
				2,
				scheduler.WithProfiles(profile),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			// Create a best effort pod.
			podName := "test-pod"
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}

				// Verify the Reserve Plugins
				if reservePlugin.numUnreserveCalled != 1 {
					t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 1.", reservePlugin.name, reservePlugin.numUnreserveCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}

				// Verify the Reserve Plugins
				if reservePlugin.numUnreserveCalled != 0 {
					t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 0.", reservePlugin.name, reservePlugin.numUnreserveCalled)
				}
			}

			if test.plugin.numPermitCalled != 1 {
				t.Errorf("Expected the Permit plugin to be called.")
			}

			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestUnReservePreBindPlugins tests unreserve of Prebind plugins.
func TestUnReservePreBindPlugins(t *testing.T) {
	tests := []struct {
		name   string
		plugin *PreBindPlugin
		fail   bool
	}{
		{
			name: "All Reserve plugins passed, but a PreBind plugin failed",
			fail: true,
			plugin: &PreBindPlugin{
				podUIDs:       make(map[types.UID]struct{}),
				rejectPreBind: true,
			},
		},
		{
			name:   "All Reserve plugins passed, and PreBind plugin succeed",
			fail:   false,
			plugin: &PreBindPlugin{podUIDs: make(map[types.UID]struct{})},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reservePlugin := &ReservePlugin{
				name:        "reservePlugin",
				failReserve: false,
			}
			registry, profile := initRegistryAndConfig(t, []framework.Plugin{test.plugin, reservePlugin}...)

			// Create the API server and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(
				t,
				testutils.InitTestAPIServer(t, "unreserve-prebind-plugin", nil),
				2,
				scheduler.WithProfiles(profile),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			// Create a pause pod.
			podName := "test-pod"
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a reasons other than Unschedulable, but got: %v", err)
				}

				// Verify the Reserve Plugins
				if reservePlugin.numUnreserveCalled != 1 {
					t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 1.", reservePlugin.name, reservePlugin.numUnreserveCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}

				// Verify the Reserve Plugins
				if reservePlugin.numUnreserveCalled != 0 {
					t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 0.", reservePlugin.name, reservePlugin.numUnreserveCalled)
				}
			}

			if test.plugin.numPreBindCalled != 1 {
				t.Errorf("Expected the Prebind plugin to be called.")
			}

			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestUnReserveBindPlugins tests unreserve of Bind plugins.
func TestUnReserveBindPlugins(t *testing.T) {
	tests := []struct {
		name   string
		plugin *BindPlugin
		fail   bool
	}{
		{
			name:   "All Reserve plugins passed, and Bind plugin succeed",
			fail:   false,
			plugin: &BindPlugin{name: "SucceedBindPlugin"},
		},
		{
			name:   "All Reserve plugins passed, but a Bind plugin failed",
			fail:   false,
			plugin: &BindPlugin{name: "FailedBindPlugin"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reservePlugin := &ReservePlugin{
				name:        "reservePlugin",
				failReserve: false,
			}
			registry, profile := initRegistryAndConfig(t, []framework.Plugin{test.plugin, reservePlugin}...)

			apiCtx := testutils.InitTestAPIServer(t, "unreserve-bind-plugin", nil)
			test.plugin.client = apiCtx.ClientSet

			// Create the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(
				t,
				apiCtx,
				2,
				scheduler.WithProfiles(profile),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			// Create a pause pod.
			podName := "test-pod"
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a reasons other than Unschedulable, but got: %v", err)
				}

				// Verify the Reserve Plugins
				if reservePlugin.numUnreserveCalled != 1 {
					t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 1.", reservePlugin.name, reservePlugin.numUnreserveCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}

				// Verify the Reserve Plugins
				if reservePlugin.numUnreserveCalled != 0 {
					t.Errorf("Reserve Plugin %s numUnreserveCalled = %d, want 0.", reservePlugin.name, reservePlugin.numUnreserveCalled)
				}
			}

			if test.plugin.numBindCalled != 1 {
				t.Errorf("Expected the Bind plugin to be called.")
			}

			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

type pluginInvokeEvent struct {
	pluginName string
	val        int
}

// TestBindPlugin tests invocation of bind plugins.
func TestBindPlugin(t *testing.T) {
	testContext := testutils.InitTestAPIServer(t, "bind-plugin", nil)
	bindPlugin1 := &BindPlugin{name: "bind-plugin-1", client: testContext.ClientSet}
	bindPlugin2 := &BindPlugin{name: "bind-plugin-2", client: testContext.ClientSet}
	reservePlugin := &ReservePlugin{name: "mock-reserve-plugin"}
	postBindPlugin := &PostBindPlugin{name: "mock-post-bind-plugin"}
	// Create a plugin registry for testing. Register reserve, bind, and
	// postBind plugins.

	registry := frameworkruntime.Registry{
		reservePlugin.Name():  newPlugin(reservePlugin),
		bindPlugin1.Name():    newPlugin(bindPlugin1),
		bindPlugin2.Name():    newPlugin(bindPlugin2),
		postBindPlugin.Name(): newPlugin(postBindPlugin),
	}

	// Setup initial unreserve and bind plugins for testing.
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &v1beta3.Plugins{
				MultiPoint: v1beta3.PluginSet{
					Disabled: []v1beta3.Plugin{
						{Name: defaultbinder.Name},
					},
				},
				Reserve: v1beta3.PluginSet{
					Enabled: []v1beta3.Plugin{{Name: reservePlugin.Name()}},
				},
				Bind: v1beta3.PluginSet{
					// Put DefaultBinder last.
					Enabled:  []v1beta3.Plugin{{Name: bindPlugin1.Name()}, {Name: bindPlugin2.Name()}, {Name: defaultbinder.Name}},
					Disabled: []v1beta3.Plugin{{Name: defaultbinder.Name}},
				},
				PostBind: v1beta3.PluginSet{
					Enabled: []v1beta3.Plugin{{Name: postBindPlugin.Name()}},
				},
			},
		}},
	})

	// Create the scheduler with the test plugin set.
	testCtx := testutils.InitTestSchedulerWithOptions(t, testContext, 0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	defer testutils.CleanupTest(t, testCtx)

	// Add a few nodes.
	_, err := createAndWaitForNodesInCache(testCtx, "test-node", st.MakeNode(), 2)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                   string
		bindPluginStatuses     []*framework.Status
		expectBoundByScheduler bool   // true means this test case expecting scheduler would bind pods
		expectBoundByPlugin    bool   // true means this test case expecting a plugin would bind pods
		expectBindPluginName   string // expecting plugin name to bind pods
		expectInvokeEvents     []pluginInvokeEvent
	}{
		{
			name:                   "bind plugins skipped to bind the pod and scheduler bond the pod",
			bindPluginStatuses:     []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Skip, "")},
			expectBoundByScheduler: true,
			expectInvokeEvents:     []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		{
			name:                 "bindplugin2 succeeded to bind the pod",
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Skip, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin2.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: bindPlugin2.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		{
			name:                 "bindplugin1 succeeded to bind the pod",
			bindPluginStatuses:   []*framework.Status{framework.NewStatus(framework.Success, ""), framework.NewStatus(framework.Success, "")},
			expectBoundByPlugin:  true,
			expectBindPluginName: bindPlugin1.Name(),
			expectInvokeEvents:   []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: postBindPlugin.Name(), val: 1}},
		},
		{
			name:               "bind plugin fails to bind the pod",
			bindPluginStatuses: []*framework.Status{framework.NewStatus(framework.Error, "failed to bind"), framework.NewStatus(framework.Success, "")},
			expectInvokeEvents: []pluginInvokeEvent{{pluginName: bindPlugin1.Name(), val: 1}, {pluginName: reservePlugin.Name(), val: 1}},
		},
	}

	var pluginInvokeEventChan chan pluginInvokeEvent
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pluginInvokeEventChan = make(chan pluginInvokeEvent, 10)

			bindPlugin1.bindStatus = test.bindPluginStatuses[0]
			bindPlugin2.bindStatus = test.bindPluginStatuses[1]

			bindPlugin1.pluginInvokeEventChan = pluginInvokeEventChan
			bindPlugin2.pluginInvokeEventChan = pluginInvokeEventChan
			reservePlugin.pluginInvokeEventChan = pluginInvokeEventChan
			postBindPlugin.pluginInvokeEventChan = pluginInvokeEventChan

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.expectBoundByScheduler || test.expectBoundByPlugin {
				// bind plugins skipped to bind the pod
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Fatalf("Expected the pod to be scheduled. error: %v", err)
				}
				pod, err = testCtx.ClientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
				if err != nil {
					t.Errorf("can't get pod: %v", err)
				}
				if test.expectBoundByScheduler {
					if pod.Annotations[bindPluginAnnotation] != "" {
						t.Errorf("Expected the pod to be bound by scheduler instead of by bindplugin %s", pod.Annotations[bindPluginAnnotation])
					}
					if bindPlugin1.numBindCalled != 1 || bindPlugin2.numBindCalled != 1 {
						t.Errorf("Expected each bind plugin to be called once, was called %d and %d times.", bindPlugin1.numBindCalled, bindPlugin2.numBindCalled)
					}
				} else {
					if pod.Annotations[bindPluginAnnotation] != test.expectBindPluginName {
						t.Errorf("Expected the pod to be bound by bindplugin %s instead of by bindplugin %s", test.expectBindPluginName, pod.Annotations[bindPluginAnnotation])
					}
					if bindPlugin1.numBindCalled != 1 {
						t.Errorf("Expected %s to be called once, was called %d times.", bindPlugin1.Name(), bindPlugin1.numBindCalled)
					}
					if test.expectBindPluginName == bindPlugin1.Name() && bindPlugin2.numBindCalled > 0 {
						// expect bindplugin1 succeeded to bind the pod and bindplugin2 should not be called.
						t.Errorf("Expected %s not to be called, was called %d times.", bindPlugin2.Name(), bindPlugin1.numBindCalled)
					}
				}
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, func() (done bool, err error) {
					return postBindPlugin.numPostBindCalled == 1, nil
				}); err != nil {
					t.Errorf("Expected the postbind plugin to be called once, was called %d times.", postBindPlugin.numPostBindCalled)
				}
				if reservePlugin.numUnreserveCalled != 0 {
					t.Errorf("Expected unreserve to not be called, was called %d times.", reservePlugin.numUnreserveCalled)
				}
			} else {
				// bind plugin fails to bind the pod
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
				if postBindPlugin.numPostBindCalled > 0 {
					t.Errorf("Didn't expect the postbind plugin to be called %d times.", postBindPlugin.numPostBindCalled)
				}
			}
			for j := range test.expectInvokeEvents {
				expectEvent := test.expectInvokeEvents[j]
				select {
				case event := <-pluginInvokeEventChan:
					if event.pluginName != expectEvent.pluginName {
						t.Errorf("Expect invoke event %d from plugin %s instead of %s", j, expectEvent.pluginName, event.pluginName)
					}
					if event.val != expectEvent.val {
						t.Errorf("Expect val of invoke event %d to be %d instead of %d", j, expectEvent.val, event.val)
					}
				case <-time.After(time.Second * 30):
					t.Errorf("Waiting for invoke event %d timeout.", j)
				}
			}
			postBindPlugin.reset()
			bindPlugin1.reset()
			bindPlugin2.reset()
			reservePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPostBindPlugin tests invocation of postbind plugins.
func TestPostBindPlugin(t *testing.T) {
	tests := []struct {
		name        string
		preBindFail bool
	}{
		{
			name:        "plugin preBind fail",
			preBindFail: true,
		},
		{
			name:        "plugin preBind do not fail",
			preBindFail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a plugin registry for testing. Register a prebind and a postbind plugin.
			preBindPlugin := &PreBindPlugin{
				failPreBind: test.preBindFail,
				podUIDs:     make(map[types.UID]struct{}),
			}
			postBindPlugin := &PostBindPlugin{
				name:                  postBindPluginName,
				pluginInvokeEventChan: make(chan pluginInvokeEvent, 1),
			}

			registry, prof := initRegistryAndConfig(t, preBindPlugin, postBindPlugin)
			// Create the API server and the scheduler with the test plugin set.
			testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "postbind-plugin", nil), 2,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer testutils.CleanupTest(t, testCtx)

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.preBindFail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
				if postBindPlugin.numPostBindCalled > 0 {
					t.Errorf("Didn't expect the postbind plugin to be called %d times.", postBindPlugin.numPostBindCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
				select {
				case <-postBindPlugin.pluginInvokeEventChan:
				case <-time.After(time.Second * 15):
					t.Errorf("pluginInvokeEventChan timed out")
				}
				if postBindPlugin.numPostBindCalled == 0 {
					t.Errorf("Expected the postbind plugin to be called, was called %d times.", postBindPlugin.numPostBindCalled)
				}
			}

			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPermitPlugin tests invocation of permit plugins.
func TestPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	perPlugin := &PermitPlugin{name: permitPluginName}
	registry, prof := initRegistryAndConfig(t, perPlugin)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "permit-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name    string
		fail    bool
		reject  bool
		timeout bool
	}{
		{
			name:    "disable fail, reject and timeout flags",
			fail:    false,
			reject:  false,
			timeout: false,
		},
		{
			name:    "enable fail, disable reject and timeout flags",
			fail:    true,
			reject:  false,
			timeout: false,
		},
		{
			name:    "disable fail and timeout, enable reject flags",
			fail:    false,
			reject:  true,
			timeout: false,
		},
		{
			name:    "enable fail and reject, disable timeout flags",
			fail:    true,
			reject:  true,
			timeout: false,
		},
		{
			name:    "disable fail and reject, disable timeout flags",
			fail:    false,
			reject:  false,
			timeout: true,
		},
		{
			name:    "disable fail and reject, enable timeout flags",
			fail:    false,
			reject:  false,
			timeout: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			perPlugin.failPermit = test.fail
			perPlugin.rejectPermit = test.reject
			perPlugin.timeoutPermit = test.timeout
			perPlugin.waitAndRejectPermit = false
			perPlugin.waitAndAllowPermit = false

			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}
			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podSchedulingError(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Expected a scheduling error, but didn't get it. error: %v", err)
				}
			} else {
				if test.reject || test.timeout {
					if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
						t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
					}
				} else {
					if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
						t.Errorf("Expected the pod to be scheduled. error: %v", err)
					}
				}
			}

			if perPlugin.numPermitCalled == 0 {
				t.Errorf("Expected the permit plugin to be called.")
			}

			perPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestMultiplePermitPlugins tests multiple permit plugins returning wait for a same pod.
func TestMultiplePermitPlugins(t *testing.T) {
	// Create a plugin registry for testing.
	perPlugin1 := &PermitPlugin{name: "permit-plugin-1"}
	perPlugin2 := &PermitPlugin{name: "permit-plugin-2"}
	registry, prof := initRegistryAndConfig(t, perPlugin1, perPlugin2)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "multi-permit-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	// Both permit plugins will return Wait for permitting
	perPlugin1.timeoutPermit = true
	perPlugin2.timeoutPermit = true

	// Create a test pod.
	podName := "test-pod"
	pod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&testutils.PausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
	if err != nil {
		t.Errorf("Error while creating a test pod: %v", err)
	}

	var waitingPod framework.WaitingPod
	// Wait until the test pod is actually waiting.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		waitingPod = perPlugin1.fh.GetWaitingPod(pod.UID)
		return waitingPod != nil, nil
	})

	// Check the number of pending permits
	if l := len(waitingPod.GetPendingPlugins()); l != 2 {
		t.Errorf("Expected the number of pending plugins is 2, but got %d", l)
	}

	perPlugin1.allowAllPods()
	// Check the number of pending permits
	if l := len(waitingPod.GetPendingPlugins()); l != 1 {
		t.Errorf("Expected the number of pending plugins is 1, but got %d", l)
	}

	perPlugin2.allowAllPods()
	if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
		t.Errorf("Expected the pod to be scheduled. error: %v", err)
	}

	if perPlugin1.numPermitCalled == 0 || perPlugin2.numPermitCalled == 0 {
		t.Errorf("Expected the permit plugin to be called.")
	}

	testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
}

// TestPermitPluginsCancelled tests whether all permit plugins are cancelled when pod is rejected.
func TestPermitPluginsCancelled(t *testing.T) {
	// Create a plugin registry for testing.
	perPlugin1 := &PermitPlugin{name: "permit-plugin-1"}
	perPlugin2 := &PermitPlugin{name: "permit-plugin-2"}
	registry, prof := initRegistryAndConfig(t, perPlugin1, perPlugin2)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "permit-plugins", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	// Both permit plugins will return Wait for permitting
	perPlugin1.timeoutPermit = true
	perPlugin2.timeoutPermit = true

	// Create a test pod.
	podName := "test-pod"
	pod, err := createPausePod(testCtx.ClientSet,
		initPausePod(&testutils.PausePodConfig{Name: podName, Namespace: testCtx.NS.Name}))
	if err != nil {
		t.Errorf("Error while creating a test pod: %v", err)
	}

	var waitingPod framework.WaitingPod
	// Wait until the test pod is actually waiting.
	wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		waitingPod = perPlugin1.fh.GetWaitingPod(pod.UID)
		return waitingPod != nil, nil
	})

	perPlugin1.rejectAllPods()
	// Wait some time for the permit plugins to be cancelled
	err = wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
		return perPlugin1.cancelled && perPlugin2.cancelled, nil
	})
	if err != nil {
		t.Errorf("Expected all permit plugins to be cancelled")
	}
}

// TestCoSchedulingWithPermitPlugin tests invocation of permit plugins.
func TestCoSchedulingWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a permit plugin.
	permitPlugin := &PermitPlugin{name: permitPluginName}
	registry, prof := initRegistryAndConfig(t, permitPlugin)

	// Create the API server and the scheduler with the test plugin set.
	// TODO Make the subtests not share scheduler instances.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "permit-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name       string
		waitReject bool
		waitAllow  bool
	}{
		{
			name:       "having wait reject true and wait allow false",
			waitReject: true,
			waitAllow:  false,
		},
		{
			name:       "having wait reject false and wait allow true",
			waitReject: false,
			waitAllow:  true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			permitPlugin.failPermit = false
			permitPlugin.rejectPermit = false
			permitPlugin.timeoutPermit = false
			permitPlugin.waitAndRejectPermit = test.waitReject
			permitPlugin.waitAndAllowPermit = test.waitAllow

			// Create two pods. First pod to enter Permit() will wait and a second one will either
			// reject or allow first one.
			podA, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "pod-a", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating the first pod: %v", err)
			}
			podB, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "pod-b", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating the second pod: %v", err)
			}

			if test.waitReject {
				if err = waitForPodUnschedulable(testCtx.ClientSet, podA); err != nil {
					t.Errorf("Didn't expect the first pod to be scheduled. error: %v", err)
				}
				if err = waitForPodUnschedulable(testCtx.ClientSet, podB); err != nil {
					t.Errorf("Didn't expect the second pod to be scheduled. error: %v", err)
				}
				if !((permitPlugin.waitingPod == podA.Name && permitPlugin.rejectingPod == podB.Name) ||
					(permitPlugin.waitingPod == podB.Name && permitPlugin.rejectingPod == podA.Name)) {
					t.Errorf("Expect one pod to wait and another pod to reject instead %s waited and %s rejected.",
						permitPlugin.waitingPod, permitPlugin.rejectingPod)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, podA); err != nil {
					t.Errorf("Expected the first pod to be scheduled. error: %v", err)
				}
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, podB); err != nil {
					t.Errorf("Expected the second pod to be scheduled. error: %v", err)
				}
				if !((permitPlugin.waitingPod == podA.Name && permitPlugin.allowingPod == podB.Name) ||
					(permitPlugin.waitingPod == podB.Name && permitPlugin.allowingPod == podA.Name)) {
					t.Errorf("Expect one pod to wait and another pod to allow instead %s waited and %s allowed.",
						permitPlugin.waitingPod, permitPlugin.allowingPod)
				}
			}

			if permitPlugin.numPermitCalled == 0 {
				t.Errorf("Expected the permit plugin to be called.")
			}

			permitPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{podA, podB})
		})
	}
}

// TestFilterPlugin tests invocation of filter plugins.
func TestFilterPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a filter plugin.
	filterPlugin := &FilterPlugin{}
	registry, prof := initRegistryAndConfig(t, filterPlugin)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "filter-plugin", nil), 1,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail filter plugin",
			fail: true,
		},
		{
			name: "do not fail filter plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filterPlugin.failFilter = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = wait.Poll(10*time.Millisecond, 30*time.Second, podUnschedulable(testCtx.ClientSet, pod.Namespace, pod.Name)); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled.")
				}
				if filterPlugin.numFilterCalled < 1 {
					t.Errorf("Expected the filter plugin to be called at least 1 time, but got %v.", filterPlugin.numFilterCalled)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
				if filterPlugin.numFilterCalled != 1 {
					t.Errorf("Expected the filter plugin to be called 1 time, but got %v.", filterPlugin.numFilterCalled)
				}
			}

			filterPlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPreScorePlugin tests invocation of pre-score plugins.
func TestPreScorePlugin(t *testing.T) {
	// Create a plugin registry for testing. Register only a pre-score plugin.
	preScorePlugin := &PreScorePlugin{}
	registry, prof := initRegistryAndConfig(t, preScorePlugin)

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "pre-score-plugin", nil), 2,
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	tests := []struct {
		name string
		fail bool
	}{
		{
			name: "fail preScore plugin",
			fail: true,
		},
		{
			name: "do not fail preScore plugin",
			fail: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			preScorePlugin.failPreScore = test.fail
			// Create a best effort pod.
			pod, err := createPausePod(testCtx.ClientSet,
				initPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Errorf("Error while creating a test pod: %v", err)
			}

			if test.fail {
				if err = waitForPodUnschedulable(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Didn't expect the pod to be scheduled. error: %v", err)
				}
			} else {
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
					t.Errorf("Expected the pod to be scheduled. error: %v", err)
				}
			}

			if preScorePlugin.numPreScoreCalled == 0 {
				t.Errorf("Expected the pre-score plugin to be called.")
			}

			preScorePlugin.reset()
			testutils.CleanupPods(testCtx.ClientSet, t, []*v1.Pod{pod})
		})
	}
}

// TestPreemptWithPermitPlugin tests preempt with permit plugins.
// It verifies how waitingPods behave in different scenarios:
// - when waitingPods get preempted
//   - they should be removed from internal waitingPods map, but not physically deleted
//   - it'd trigger moving unschedulable Pods, but not the waitingPods themselves
// - when waitingPods get deleted externally, it'd trigger moving unschedulable Pods
func TestPreemptWithPermitPlugin(t *testing.T) {
	// Create a plugin registry for testing. Register a permit and a filter plugin.
	permitPlugin := &PermitPlugin{}
	// Inject a fake filter plugin to use its internal `numFilterCalled` to verify
	// how many times a Pod gets tried scheduling.
	filterPlugin := &FilterPlugin{numCalledPerPod: make(map[string]int)}
	registry := frameworkruntime.Registry{
		permitPluginName: newPlugin(permitPlugin),
		filterPluginName: newPlugin(filterPlugin),
	}

	// Setup initial permit and filter plugins in the profile.
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{
			{
				SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
				Plugins: &v1beta3.Plugins{
					Permit: v1beta3.PluginSet{
						Enabled: []v1beta3.Plugin{
							{Name: permitPluginName},
						},
					},
					Filter: v1beta3.PluginSet{
						// Ensure the fake filter plugin is always called; otherwise noderesources
						// would fail first and exit the Filter phase.
						Enabled: []v1beta3.Plugin{
							{Name: filterPluginName},
							{Name: noderesources.Name},
						},
						Disabled: []v1beta3.Plugin{
							{Name: noderesources.Name},
						},
					},
				},
			},
		},
	})

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "preempt-with-permit-plugin", nil), 0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	// Add one node.
	nodeRes := map[v1.ResourceName]string{
		v1.ResourcePods:   "32",
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
	_, err := createAndWaitForNodesInCache(testCtx, "test-node", st.MakeNode().Capacity(nodeRes), 1)
	if err != nil {
		t.Fatal(err)
	}

	ns := testCtx.NS.Name
	lowPriority, highPriority := int32(100), int32(300)
	resReq := map[v1.ResourceName]string{
		v1.ResourceCPU:    "200m",
		v1.ResourceMemory: "200",
	}
	preemptorReq := map[v1.ResourceName]string{
		v1.ResourceCPU:    "400m",
		v1.ResourceMemory: "400",
	}

	tests := []struct {
		name                   string
		deleteWaitingPod       bool
		maxNumWaitingPodCalled int
		runningPod             *v1.Pod
		waitingPod             *v1.Pod
		preemptor              *v1.Pod
	}{
		{
			name:                   "waiting pod is not physically deleted upon preemption",
			maxNumWaitingPodCalled: 2,
			runningPod:             st.MakePod().Name("running-pod").Namespace(ns).Priority(lowPriority).Req(resReq).ZeroTerminationGracePeriod().Obj(),
			waitingPod:             st.MakePod().Name("waiting-pod").Namespace(ns).Priority(lowPriority).Req(resReq).ZeroTerminationGracePeriod().Obj(),
			preemptor:              st.MakePod().Name("preemptor-pod").Namespace(ns).Priority(highPriority).Req(preemptorReq).ZeroTerminationGracePeriod().Obj(),
		},
		{
			name:                   "rejecting a waiting pod to trigger retrying unschedulable pods immediately, but the waiting pod itself won't be retried",
			maxNumWaitingPodCalled: 1,
			waitingPod:             st.MakePod().Name("waiting-pod").Namespace(ns).Priority(lowPriority).Req(resReq).ZeroTerminationGracePeriod().Obj(),
			preemptor:              st.MakePod().Name("preemptor-pod").Namespace(ns).Priority(highPriority).Req(preemptorReq).ZeroTerminationGracePeriod().Obj(),
		},
		{
			name:                   "deleting a waiting pod to trigger retrying unschedulable pods immediately",
			deleteWaitingPod:       true,
			maxNumWaitingPodCalled: 1,
			waitingPod:             st.MakePod().Name("waiting-pod").Namespace(ns).Priority(lowPriority).Req(resReq).ZeroTerminationGracePeriod().Obj(),
			preemptor:              st.MakePod().Name("preemptor-pod").Namespace(ns).Priority(lowPriority).Req(preemptorReq).ZeroTerminationGracePeriod().Obj(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				permitPlugin.reset()
				filterPlugin.reset()
				var pods []*v1.Pod
				for _, p := range []*v1.Pod{tt.runningPod, tt.waitingPod, tt.preemptor} {
					if p != nil {
						pods = append(pods, p)
					}
				}
				testutils.CleanupPods(testCtx.ClientSet, t, pods)
			}()

			permitPlugin.waitAndAllowPermit = true
			permitPlugin.waitingPod = "waiting-pod"

			if r := tt.runningPod; r != nil {
				if _, err := createPausePod(testCtx.ClientSet, r); err != nil {
					t.Fatalf("Error while creating the running pod: %v", err)
				}
				// Wait until the pod to be scheduled.
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, r); err != nil {
					t.Fatalf("The running pod is expected to be scheduled: %v", err)
				}
			}

			if w := tt.waitingPod; w != nil {
				if _, err := createPausePod(testCtx.ClientSet, w); err != nil {
					t.Fatalf("Error while creating the waiting pod: %v", err)
				}
				// Wait until the waiting-pod is actually waiting.
				if err := wait.Poll(10*time.Millisecond, 30*time.Second, func() (bool, error) {
					w := false
					permitPlugin.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { w = true })
					return w, nil
				}); err != nil {
					t.Fatalf("The waiting pod is expected to be waiting: %v", err)
				}
			}

			if p := tt.preemptor; p != nil {
				if _, err := createPausePod(testCtx.ClientSet, p); err != nil {
					t.Fatalf("Error while creating the preemptor pod: %v", err)
				}
				// Delete the waiting pod if specified.
				if w := tt.waitingPod; w != nil && tt.deleteWaitingPod {
					if err := deletePod(testCtx.ClientSet, w.Name, w.Namespace); err != nil {
						t.Fatalf("Error while deleting the waiting pod: %v", err)
					}
				}
				if err = testutils.WaitForPodToSchedule(testCtx.ClientSet, p); err != nil {
					t.Fatalf("Expected the preemptor pod to be scheduled. error: %v", err)
				}
			}

			if w := tt.waitingPod; w != nil {
				if err := wait.Poll(200*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					w := false
					permitPlugin.fh.IterateOverWaitingPods(func(wp framework.WaitingPod) { w = true })
					return !w, nil
				}); err != nil {
					t.Fatalf("Expected the waiting pod to get preempted.")
				}

				filterPlugin.RLock()
				waitingPodCalled := filterPlugin.numCalledPerPod[fmt.Sprintf("%v/%v", w.Namespace, w.Name)]
				filterPlugin.RUnlock()
				if waitingPodCalled > tt.maxNumWaitingPodCalled {
					t.Fatalf("Expected the waiting pod to be called %v times at most, but got %v", tt.maxNumWaitingPodCalled, waitingPodCalled)
				}

				if !tt.deleteWaitingPod {
					// Expect the waitingPod to be still present.
					if _, err := getPod(testCtx.ClientSet, w.Name, w.Namespace); err != nil {
						t.Error("Get waiting pod in waiting pod failed.")
					}
				}

				if permitPlugin.numPermitCalled == 0 {
					t.Errorf("Expected the permit plugin to be called.")
				}
			}

			if r := tt.runningPod; r != nil {
				// Expect the runningPod to be deleted physically.
				if _, err = getPod(testCtx.ClientSet, r.Name, r.Namespace); err == nil {
					t.Error("The running pod still exists.")
				} else if !errors.IsNotFound(err) {
					t.Errorf("Get running pod failed: %v", err)
				}
			}
		})
	}
}

const (
	jobPluginName = "job plugin"
)

var _ framework.PreFilterPlugin = &JobPlugin{}
var _ framework.PostBindPlugin = &PostBindPlugin{}

type JobPlugin struct {
	podLister     listersv1.PodLister
	podsActivated bool
}

func (j *JobPlugin) Name() string {
	return jobPluginName
}

func (j *JobPlugin) PreFilter(_ context.Context, _ *framework.CycleState, p *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	labelSelector := labels.SelectorFromSet(labels.Set{"driver": ""})
	driverPods, err := j.podLister.Pods(p.Namespace).List(labelSelector)
	if err != nil {
		return nil, framework.AsStatus(err)
	}
	if len(driverPods) == 0 {
		return nil, framework.NewStatus(framework.UnschedulableAndUnresolvable, "unable to find driver pod")
	}
	return nil, nil
}

func (j *JobPlugin) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func (j *JobPlugin) PostBind(_ context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) {
	if _, ok := p.Labels["driver"]; !ok {
		return
	}

	// If it's a driver pod, move other executor pods proactively to accelerating the scheduling.
	labelSelector := labels.SelectorFromSet(labels.Set{"executor": ""})
	podsToActivate, err := j.podLister.Pods(p.Namespace).List(labelSelector)
	if err == nil && len(podsToActivate) != 0 {
		c, err := state.Read(framework.PodsToActivateKey)
		if err == nil {
			if s, ok := c.(*framework.PodsToActivate); ok {
				s.Lock()
				for _, pod := range podsToActivate {
					namespacedName := fmt.Sprintf("%v/%v", pod.Namespace, pod.Name)
					s.Map[namespacedName] = pod
				}
				s.Unlock()
				j.podsActivated = true
			}
		}
	}
}

// This test simulates a typical spark job workflow.
// - N executor pods are created, but kept pending due to missing the driver pod
// - when the driver pod gets created and scheduled, proactively move the executors to activeQ
//   and thus accelerate the entire job workflow.
func TestActivatePods(t *testing.T) {
	var jobPlugin *JobPlugin
	// Create a plugin registry for testing. Register a Job plugin.
	registry := frameworkruntime.Registry{jobPluginName: func(_ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
		jobPlugin = &JobPlugin{podLister: fh.SharedInformerFactory().Core().V1().Pods().Lister()}
		return jobPlugin, nil
	}}

	// Setup initial filter plugin for testing.
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &v1beta3.Plugins{
				PreFilter: v1beta3.PluginSet{
					Enabled: []v1beta3.Plugin{
						{Name: jobPluginName},
					},
				},
				PostBind: v1beta3.PluginSet{
					Enabled: []v1beta3.Plugin{
						{Name: jobPluginName},
					},
				},
			},
		}},
	})

	// Create the API server and the scheduler with the test plugin set.
	testCtx := initTestSchedulerForFrameworkTest(t, testutils.InitTestAPIServer(t, "job-plugin", nil), 1,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry))
	defer testutils.CleanupTest(t, testCtx)

	cs := testCtx.ClientSet
	ns := testCtx.NS.Name
	pause := imageutils.GetPauseImageName()

	// Firstly create 2 executor pods.
	var pods []*v1.Pod
	for i := 1; i <= 2; i++ {
		name := fmt.Sprintf("executor-%v", i)
		executor := st.MakePod().Name(name).Namespace(ns).Label("executor", "").Container(pause).Obj()
		pods = append(pods, executor)
		if _, err := cs.CoreV1().Pods(executor.Namespace).Create(context.TODO(), executor, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %v: %v", executor.Name, err)
		}
	}

	// Wait for the 2 executor pods to be unschedulable.
	for _, pod := range pods {
		if err := waitForPodUnschedulable(cs, pod); err != nil {
			t.Errorf("Failed to wait for Pod %v to be unschedulable: %v", pod.Name, err)
		}
	}

	// Create a driver pod.
	driver := st.MakePod().Name("driver").Namespace(ns).Label("driver", "").Container(pause).Obj()
	pods = append(pods, driver)
	if _, err := cs.CoreV1().Pods(driver.Namespace).Create(context.TODO(), driver, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create pod %v: %v", driver.Name, err)
	}

	// Verify all pods to be scheduled.
	for _, pod := range pods {
		if err := waitForPodToScheduleWithTimeout(cs, pod, wait.ForeverTestTimeout); err != nil {
			t.Fatalf("Failed to wait for Pod %v to be schedulable: %v", pod.Name, err)
		}
	}

	// Lastly verify the pods activation logic is really called.
	if jobPlugin.podsActivated == false {
		t.Errorf("JobPlugin's pods activation logic is not called")
	}
}

func initTestSchedulerForFrameworkTest(t *testing.T, testCtx *testutils.TestContext, nodeCount int, opts ...scheduler.Option) *testutils.TestContext {
	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, opts...)
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	if nodeCount > 0 {
		if _, err := createAndWaitForNodesInCache(testCtx, "test-node", st.MakeNode(), nodeCount); err != nil {
			t.Fatal(err)
		}
	}
	return testCtx
}

// initRegistryAndConfig returns registry and plugins config based on give plugins.
func initRegistryAndConfig(t *testing.T, plugins ...framework.Plugin) (frameworkruntime.Registry, schedulerconfig.KubeSchedulerProfile) {
	if len(plugins) == 0 {
		return frameworkruntime.Registry{}, schedulerconfig.KubeSchedulerProfile{}
	}

	registry := frameworkruntime.Registry{}
	pls := &v1beta3.Plugins{}

	for _, p := range plugins {
		registry.Register(p.Name(), newPlugin(p))
		plugin := v1beta3.Plugin{Name: p.Name()}

		switch p.(type) {
		case *PreFilterPlugin:
			pls.PreFilter.Enabled = append(pls.PreFilter.Enabled, plugin)
		case *FilterPlugin:
			pls.Filter.Enabled = append(pls.Filter.Enabled, plugin)
		case *PreScorePlugin:
			pls.PreScore.Enabled = append(pls.PreScore.Enabled, plugin)
		case *ScorePlugin, *ScoreWithNormalizePlugin:
			pls.Score.Enabled = append(pls.Score.Enabled, plugin)
		case *ReservePlugin:
			pls.Reserve.Enabled = append(pls.Reserve.Enabled, plugin)
		case *PreBindPlugin:
			pls.PreBind.Enabled = append(pls.PreBind.Enabled, plugin)
		case *BindPlugin:
			pls.Bind.Enabled = append(pls.Bind.Enabled, plugin)
			// It's intentional to disable the DefaultBind plugin. Otherwise, DefaultBinder's failure would fail
			// a pod's scheduling, as well as the test BindPlugin's execution.
			pls.Bind.Disabled = []v1beta3.Plugin{{Name: defaultbinder.Name}}
		case *PostBindPlugin:
			pls.PostBind.Enabled = append(pls.PostBind.Enabled, plugin)
		case *PermitPlugin:
			pls.Permit.Enabled = append(pls.Permit.Enabled, plugin)
		}
	}

	versionedCfg := v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins:       pls,
		}},
	}
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, versionedCfg)
	return registry, cfg.Profiles[0]
}
