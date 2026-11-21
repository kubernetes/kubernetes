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

package podcheckpoint

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	probetest "k8s.io/kubernetes/pkg/kubelet/prober/testing"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubeletutil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/mount-utils"
	"k8s.io/utils/ptr"
)

const testNodeName = "127.0.0.1"

type testManager struct {
	manager     *manager
	fakeRuntime *containertest.FakeRuntime
}

func (*testManager) Cleanup() {}

// newTestManager returns a manager whose dependencies are fakes or in-memory
// kubelet managers. Pod syncs run through checkpointPodProvider.
func newTestManager(t *testing.T) *testManager {
	t.Helper()
	ctx := ktesting.Init(t)
	client := fake.NewClientset()
	podManager := kubepod.NewBasicPodManager()
	statusManager := status.NewManager(client, podManager, &statustest.FakePodDeletionSafetyProvider{}, kubeletutil.NewPodStartupLatencyTracker())
	sourcesReady := config.NewSourcesReady(func(sets.Set[string]) bool { return true })
	recorder := record.NewFakeRecorder(100)
	fakeRuntime := &containertest.FakeRuntime{T: t}
	checkpointsDir := filepath.Join(t.TempDir(), "pod-checkpoints")
	require.NoError(t, os.MkdirAll(checkpointsDir, 0o700))
	pods := &checkpointPodProvider{Manager: podManager}
	m := newManager(ctx, Config{
		KubeClient:           client,
		NodeName:             testNodeName,
		Recorder:             recorder,
		PodProvider:          pods,
		ContainerRuntime:     fakeRuntime,
		ProbeManager:         probetest.FakeManager{},
		StatusManager:        statusManager,
		SourcesReady:         sourcesReady,
		KubeletConfiguration: &kubeletconfiginternal.KubeletConfiguration{},
		CheckpointsDir:       checkpointsDir,
		Mounter:              mount.NewFakeMounter(nil),
	})
	pods.manager = m
	return &testManager{manager: m, fakeRuntime: fakeRuntime}
}

type checkpointRuntime struct {
	kubecontainer.Runtime
	checkpointPod    func(context.Context, *runtimeapi.CheckpointPodRequest) error
	resizeInProgress func(*v1.Pod, *kubecontainer.PodStatus) bool
}

func (r *checkpointRuntime) CheckpointPod(ctx context.Context, request *runtimeapi.CheckpointPodRequest) error {
	return r.checkpointPod(ctx, request)
}

func (r *checkpointRuntime) IsPodResizeInProgress(pod *v1.Pod, status *kubecontainer.PodStatus) bool {
	if r.resizeInProgress != nil {
		return r.resizeInProgress(pod, status)
	}
	return r.Runtime.IsPodResizeInProgress(pod, status)
}

type checkpointProbeManager struct {
	probetest.FakeManager
	suspend func(context.Context, types.UID) error
	resumed chan types.UID
}

func (m *checkpointProbeManager) SuspendProbes(ctx context.Context, uid types.UID) error {
	return m.suspend(ctx, uid)
}

func (m *checkpointProbeManager) ResumeProbes(uid types.UID) {
	m.resumed <- uid
}

// checkpointPodProvider stands in for the Kubelet in checkpoint tests. A
// sync runs only the step of SyncPod that captures a requested checkpoint,
// with the status that the runtime reports. It runs before SyncPodNow returns
// unless async is set. onSync, if set, replaces the capture.
type checkpointPodProvider struct {
	kubepod.Manager
	manager              *manager
	async                bool
	terminationRequested map[types.UID]bool
	onSync               func(pod *v1.Pod)
}

func (w *checkpointPodProvider) IsPodTerminatingOrTerminal(pod *v1.Pod) bool {
	return pod.DeletionTimestamp != nil || w.terminationRequested[pod.UID]
}

func (w *checkpointPodProvider) SyncPodNow(ctx context.Context, pod *v1.Pod) {
	if w.onSync != nil {
		w.onSync(pod)
		return
	}
	syncPod := func() {
		podStatus, err := w.manager.containerRuntime.GetPodStatus(ctx, &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace})
		if err != nil {
			return
		}
		w.manager.syncPodCheckpointRequest(ctx, pod, podStatus)
	}
	if w.async {
		go syncPod()
		return
	}
	syncPod()
}

func useCheckpointStorage(kl *manager) {
	kl.kubeletConfiguration.PodCheckpointStorage = &kubeletconfiginternal.PodCheckpointStorageConfiguration{
		MaxBytes: 1 << 30, MaxInodes: 10000, MinAvailableBytes: 1 << 20, MinAvailableInodes: 100,
	}
	kl.podCheckpointStorageInfo = func(string) (podCheckpointStorageInfo, error) {
		return podCheckpointStorageInfo{CapacityBytes: 1 << 30, AvailableBytes: 1 << 29, CapacityInodes: 10000, AvailableInodes: 5000, Isolated: true}, nil
	}
}

func useCheckpointPodWorkers(kl *manager) {
	useCheckpointStorage(kl)
	kl.podProvider = &checkpointPodProvider{Manager: testPods(kl), manager: kl}
}

// checkpointPod hands a checkpoint of pod to its pod worker, as the watch does.
func checkpointPod(ctx context.Context, kl *manager, pod *v1.Pod, name string, uid types.UID, timeout time.Duration) error {
	return kl.requestPodCheckpoint(ctx, pod, &podCheckpointRequest{ctx: ctx, namespace: pod.Namespace, name: name, uid: uid, timeout: timeout})
}

func TestCheckpointPod(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	setup := func(t *testing.T) (*manager, *containertest.FakeRuntime, *v1.Pod) {
		t.Helper()
		testKubelet := newTestManager(t)
		t.Cleanup(testKubelet.Cleanup)
		kubelet := testKubelet.manager
		kubelet.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
		fakeRuntime := testKubelet.fakeRuntime
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "podFoo",
				Namespace: "nsFoo",
			},
			Spec: v1.PodSpec{
				NodeName:   "testNode",
				Containers: []v1.Container{{Name: "containerFoo"}},
			},
		}
		testPods(kubelet).SetPods([]*v1.Pod{pod})
		fakeRuntime.PodList = []*containertest.FakePod{
			{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}},
		}
		fakeRuntime.PodStatus = kubecontainer.PodStatus{
			SandboxStatuses: []*runtimeapi.PodSandboxStatus{{Id: "sandbox1234"}},
			ContainerStatuses: []*kubecontainer.Status{
				{ID: kubecontainer.ContainerID{Type: "containerd", ID: "container1234"}, Name: "containerFoo", State: kubecontainer.ContainerStateRunning, ImageRef: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
			},
		}
		kubelet.kubeClient = fake.NewClientset()
		useCheckpointPodWorkers(kubelet)
		seedCheckpointCache(t, kubelet)
		startCheckpointOutcomeWorker(t, kubelet)
		return kubelet, fakeRuntime, pod
	}
	// addCheckpoint stores a pending PodCheckpoint for pod and returns its name.
	addCheckpoint := func(t *testing.T, kubelet *manager, pod *v1.Pod, checkpointUID types.UID) string {
		t.Helper()
		pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "ckpt-" + string(checkpointUID), Namespace: pod.Namespace, UID: checkpointUID}}
		require.NoError(t, kubelet.kubeClient.(*fake.Clientset).Tracker().Add(pc))
		if state := kubelet.podCheckpointWatch; state != nil {
			require.NoError(t, state.indexer.Add(pc))
		}
		return pc.Name
	}
	setupCheckpoint := func(kubelet *manager, pod *v1.Pod, checkpointUID string, failTerminalStatus bool) *atomic.Bool {
		pc := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "ckpt", Namespace: pod.Namespace, UID: types.UID(checkpointUID)},
		}
		client := fake.NewClientset(pc)
		failStatus := &atomic.Bool{}
		failStatus.Store(failTerminalStatus)
		if failTerminalStatus {
			client.PrependReactor("update", "podcheckpoints", func(action core.Action) (bool, apiruntime.Object, error) {
				obj := action.(core.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
				condition := apimeta.FindStatusCondition(obj.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				if failStatus.Load() && condition != nil && condition.Reason != nodev1alpha1.PodCheckpointReasonInProgress {
					return true, nil, errors.New("injected terminal status failure")
				}
				return false, nil, nil
			})
		}
		kubelet.kubeClient = client
		if state := kubelet.podCheckpointWatch; state != nil {
			require.NoError(t, state.indexer.Add(pc))
		}
		return failStatus
	}
	syncPodStatus := func(t *testing.T, kubelet *manager, podUID types.UID) {
		t.Helper()
		pod, found := testPods(kubelet).GetPodByUID(podUID)
		require.True(t, found)
		cached, _ := kubelet.statusManager.GetPodStatus(podUID)
		conditions := cached.Conditions[:0]
		for _, condition := range cached.Conditions {
			if condition.Type != v1.PodCheckpointing {
				conditions = append(conditions, condition)
			}
		}
		cached.Conditions = conditions
		kubelet.statusManager.SetPodStatus(klog.FromContext(ktesting.Init(t)), pod, cached)
	}
	waitForCompletion := func(t *testing.T, kubelet *manager, podUID types.UID) {
		t.Helper()
		require.Eventually(t, func() bool {
			inFlight := kubelet.podCheckpointInFlight(podUID)
			return !inFlight
		}, 5*time.Second, 10*time.Millisecond)
		syncPodStatus(t, kubelet, podUID)
		cached, _ := kubelet.statusManager.GetPodStatus(podUID)
		_, condition := podutil.GetPodCondition(&cached, v1.PodCheckpointing)
		require.Nil(t, condition, "the next Pod sync must clear the checkpointing condition")
	}
	requireCheckpointFailure := func(t *testing.T, kubelet *manager, pod *v1.Pod, name, message string) {
		t.Helper()
		waitForCompletion(t, kubelet, pod.UID)
		stored, err := kubelet.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(context.Background(), name, metav1.GetOptions{})
		require.NoError(t, err)
		ready := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
		require.NotNil(t, ready)
		require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, ready.Reason)
		require.Contains(t, ready.Message, message)
	}
	receiveRequest := func(t *testing.T, requests <-chan *runtimeapi.CheckpointPodRequest) *runtimeapi.CheckpointPodRequest {
		t.Helper()
		select {
		case request := <-requests:
			return request
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for CheckpointPod runtime call")
			return nil
		}
	}

	for _, outcome := range []string{"drained", "canceled", "deadline exceeded"} {
		t.Run("probe draining/"+outcome, func(t *testing.T) {
			kl, fakeRuntime, pod := setup(t)
			ctx, cancel := context.WithCancel(ktesting.Init(t))
			defer cancel()
			draining := make(chan struct{})
			checkpointing := make(chan *v1.PodCondition, 1)
			finishDrain := make(chan struct{})
			requests := make(chan *runtimeapi.CheckpointPodRequest, 2)
			probes := &checkpointProbeManager{resumed: make(chan types.UID, 2)}
			probes.suspend = func(ctx context.Context, uid types.UID) error {
				if uid != pod.UID {
					return nil
				}
				cached, _ := kl.statusManager.GetPodStatus(uid)
				_, condition := podutil.GetPodCondition(&cached, v1.PodCheckpointing)
				checkpointing <- condition.DeepCopy()
				close(draining)
				select {
				case <-finishDrain:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			kl.probeManager = probes
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				requests <- request
				return nil
			}}
			timeout := time.Minute
			if outcome == "deadline exceeded" {
				timeout = 50 * time.Millisecond
			}
			name := addCheckpoint(t, kl, pod, "draining-uid")
			returned := make(chan error, 1)
			go func() {
				returned <- checkpointPod(ctx, kl, pod, name, "draining-uid", timeout)
			}()
			select {
			case <-draining:
			case <-time.After(5 * time.Second):
				t.Fatal("probe draining did not start")
			}
			require.Empty(t, requests, "CRI must not run before probes drain")
			condition := <-checkpointing
			require.NotNil(t, condition)
			require.Equal(t, v1.ConditionTrue, condition.Status)

			switch outcome {
			case "drained":
				other := pod.DeepCopy()
				other.UID, other.Name = "other-pod", "other"
				testPods(kl).AddPod(other)
				fakeRuntime.PodList = append(fakeRuntime.PodList, &containertest.FakePod{Pod: &kubecontainer.Pod{ID: other.UID, Name: other.Name, Namespace: other.Namespace}})
				require.NoError(t, checkpointPod(ctx, kl, other, addCheckpoint(t, kl, other, "other-uid"), "other-uid", 0))
				receiveRequest(t, requests)
				waitForCompletion(t, kl, other.UID)
				close(finishDrain)
				receiveRequest(t, requests)
			case "canceled":
				cancel()
			}
			waitForCompletion(t, kl, pod.UID)
			require.Empty(t, requests)
			if outcome == "deadline exceeded" {
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, name, metav1.GetOptions{})
				require.NoError(t, err)
				ready := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				require.NotNil(t, ready)
				require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, ready.Reason)
				require.Contains(t, ready.Message, "did not finish within the 50ms checkpoint timeout")
			}
			resumed := make(map[types.UID]bool)
			for len(probes.resumed) > 0 {
				resumed[<-probes.resumed] = true
			}
			require.True(t, resumed[pod.UID], "probes must resume after draining")
			require.NoError(t, <-returned)
			_, outputPath, err := kl.podCheckpointOutputPath("draining-uid")
			require.NoError(t, err)
			_, err = os.Stat(outputPath)
			if outcome == "drained" {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, os.ErrNotExist)
			}
		})
	}

	for _, outcome := range []string{"runtime error", "deadline exceeded"} {
		t.Run("probes resume after CRI call/"+outcome, func(t *testing.T) {
			kl, fakeRuntime, pod := setup(t)
			probes := &checkpointProbeManager{resumed: make(chan types.UID, 1), suspend: func(context.Context, types.UID) error { return nil }}
			kl.probeManager = probes
			var checkpointing *v1.PodCondition
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(ctx context.Context, _ *runtimeapi.CheckpointPodRequest) error {
				cached, _ := kl.statusManager.GetPodStatus(pod.UID)
				_, checkpointing = podutil.GetPodCondition(&cached, v1.PodCheckpointing)
				if outcome == "deadline exceeded" {
					<-ctx.Done()
					return ctx.Err()
				}
				return errors.New("runtime failed")
			}}
			name := addCheckpoint(t, kl, pod, "resume-uid")
			require.NoError(t, checkpointPod(ktesting.Init(t), kl, pod, name, "resume-uid", 50*time.Millisecond))
			cached, _ := kl.statusManager.GetPodStatus(pod.UID)
			_, condition := podutil.GetPodCondition(&cached, v1.PodCheckpointing)
			require.NotNil(t, condition, "the condition must remain until the next Pod sync")
			require.Equal(t, v1.ConditionTrue, condition.Status)
			waitForCompletion(t, kl, pod.UID)
			require.NotNil(t, checkpointing)
			require.Equal(t, v1.ConditionTrue, checkpointing.Status)
			select {
			case uid := <-probes.resumed:
				require.Equal(t, pod.UID, uid)
			default:
				t.Fatal("probes must resume once the CRI call returns")
			}
			requireCheckpointFailure(t, kl, pod, name, "checkpoint failed")
		})
	}

	for _, dimension := range []string{"container", "sidecar", "pod", "memory volume"} {
		for _, state := range []string{"pending", "in progress", "settled", "missing allocation"} {
			t.Run("resize/"+dimension+"/"+state, func(t *testing.T) {
				if dimension == "pod" {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
				}
				if dimension == "memory volume" {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingMemoryBackedVolumes, true)
				}
				kl, fakeRuntime, pod := setup(t)
				if dimension == "sidecar" {
					pod.Spec.InitContainers = []v1.Container{{Name: "sidecar", RestartPolicy: new(v1.ContainerRestartPolicyAlways)}}
					fakeRuntime.PodStatus.ContainerStatuses = append(fakeRuntime.PodStatus.ContainerStatuses, &kubecontainer.Status{
						ID: kubecontainer.ContainerID{Type: "containerd", ID: "sidecar-id"}, Name: "sidecar", State: kubecontainer.ContainerStateRunning,
						ImageRef: fakeRuntime.PodStatus.ContainerStatuses[0].ImageRef,
					})
				}
				setResources := func(p *v1.Pod, size string) {
					resources := v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(size), v1.ResourceMemory: resource.MustParse(size + "Gi")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse(size), v1.ResourceMemory: resource.MustParse(size + "Gi")},
					}
					switch dimension {
					case "container":
						p.Spec.Containers[0].Resources = resources
					case "sidecar":
						p.Spec.InitContainers[0].Resources = resources
					case "pod":
						p.Spec.Resources = &resources
					case "memory volume":
						p.Spec.Volumes = []v1.Volume{{Name: "memory", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory, SizeLimit: new(resource.MustParse(size + "Gi"))}}}}
					}
				}
				allocated := pod.DeepCopy()
				setResources(allocated, "1")
				setResources(pod, "2")
				if state == "settled" || state == "missing allocation" {
					allocated = pod.DeepCopy()
				}
				original := pod.DeepCopy()
				setupCheckpoint(kl, pod, "resize-checkpoint", false)
				called := make(chan struct{}, 1)
				checkedResize := false
				resizing := state == "in progress"
				kl.containerRuntime = &checkpointRuntime{
					Runtime: fakeRuntime,
					resizeInProgress: func(got *v1.Pod, _ *kubecontainer.PodStatus) bool {
						checkedResize = true
						require.Equal(t, allocated.Spec, got.Spec, "resize detection must use allocated state")
						return resizing
					},
					checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error { called <- struct{}{}; return nil },
				}
				ctx := ktesting.Init(t)
				workers := kl.podProvider.(*checkpointPodProvider)
				workers.onSync = func(_ *v1.Pod) {
					podStatus, err := kl.containerRuntime.GetPodStatus(ctx, &kubecontainer.Pod{ID: allocated.UID, Name: allocated.Name, Namespace: allocated.Namespace})
					require.NoError(t, err)
					kl.syncPodCheckpointRequest(ctx, allocated, podStatus)
				}
				require.NoError(t, checkpointPod(ctx, kl, pod, "ckpt", "resize-checkpoint", time.Second))
				if resizing {
					require.Empty(t, called)
					pending := kl.hasPodCheckpointRequest(pod.UID)
					require.True(t, pending, "a resize in progress must leave the checkpoint for a later sync")
					require.True(t, kl.podCheckpointInFlight(pod.UID))
					resizing = false
					kl.podProvider.SyncPodNow(ctx, pod)
				}
				waitForCompletion(t, kl, pod.UID)
				require.Len(t, called, 1)
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, "ckpt", metav1.GetOptions{})
				require.NoError(t, err)
				want := checkpointutil.SanitizePodTemplate(allocated)
				want.Spec.Containers[0].Image = fakeRuntime.PodStatus.ContainerStatuses[0].ImageRef
				if dimension == "sidecar" {
					want.Spec.InitContainers[0].Image = fakeRuntime.PodStatus.ContainerStatuses[1].ImageRef
				}
				require.NotNil(t, stored.Status.CheckpointedPodTemplate)
				require.Equal(t, want.Spec, stored.Status.CheckpointedPodTemplate.Spec, "template must describe allocated state")
				require.True(t, checkedResize)
				require.Equal(t, original, pod, "capture must not mutate the desired Pod")
			})
		}
	}

	t.Run("condition clears before status publication", func(t *testing.T) {
		kl, fakeRuntime, pod := setup(t)
		failStatus := setupCheckpoint(kl, pod, "observability-uid", true)
		ctx, cancel := context.WithCancel(ktesting.Init(t))
		defer cancel()
		recorder := record.NewFakeRecorder(20)
		kl.recorder = recorder
		started := make(chan struct{})
		finish := make(chan struct{})
		kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(ctx context.Context, _ *runtimeapi.CheckpointPodRequest) error {
			close(started)
			select {
			case <-finish:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}}
		returned := make(chan error, 1)
		go func() { returned <- checkpointPod(ctx, kl, pod, "ckpt", "observability-uid", 0) }()
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("checkpoint runtime call did not start")
		}
		cached, _ := kl.statusManager.GetPodStatus(pod.UID)
		_, condition := podutil.GetPodCondition(&cached, v1.PodCheckpointing)
		require.NotNil(t, condition)
		require.Equal(t, v1.ConditionTrue, condition.Status)
		require.Contains(t, condition.Message, "ckpt")
		close(finish)
		select {
		case err := <-returned:
			require.NoError(t, err)
		case <-time.After(5 * time.Second):
			t.Fatal("status publication held the pod worker")
		}
		syncPodStatus(t, kl, pod.UID)
		cached, _ = kl.statusManager.GetPodStatus(pod.UID)
		_, condition = podutil.GetPodCondition(&cached, v1.PodCheckpointing)
		require.Nil(t, condition, "the next Pod sync must clear the checkpointing condition before outcome publication")
		require.True(t, kl.podCheckpointInFlight(pod.UID), "status publication must still be pending")
		for _, reason := range []string{events.PodCheckpointStarted, events.SourcePodCheckpointing} {
			select {
			case event := <-recorder.Events:
				require.Contains(t, event, "Normal "+reason+" ")
			case <-time.After(5 * time.Second):
				t.Fatalf("missing %s event", reason)
			}
		}
		cancel()
		failStatus.Store(false)
		waitForCompletion(t, kl, pod.UID)
	})

	for _, change := range []struct {
		name   string
		mutate func(*v1.Pod)
	}{
		{name: "resize", mutate: func(pod *v1.Pod) {
			pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")}
		}},
		{name: "ephemeral container", mutate: func(pod *v1.Pod) {
			pod.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Image: "debug"}}}
		}},
	} {
		t.Run("source changes before outcome publication/"+change.name, func(t *testing.T) {
			kl, fakeRuntime, pod := setup(t)
			setupCheckpoint(kl, pod, "source-change-uid", false)
			ctx, cancel := context.WithCancel(ktesting.Init(t))
			defer cancel()
			publicationAttempted := make(chan struct{}, 1)
			allowPublication := make(chan struct{})
			kl.kubeClient.(*fake.Clientset).PrependReactor("update", "podcheckpoints", func(action core.Action) (bool, apiruntime.Object, error) {
				pc := action.(core.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
				if !apimeta.IsStatusConditionTrue(pc.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady) {
					return false, nil, nil
				}
				select {
				case <-allowPublication:
					return false, nil, nil
				default:
				}
				select {
				case publicationAttempted <- struct{}{}:
				default:
				}
				return true, nil, errors.New("injected terminal status failure")
			})
			captureCalls := 0
			var outputPath string
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				captureCalls++
				outputPath = request.OutputPath
				return os.WriteFile(filepath.Join(outputPath, "capture"), []byte("original capture"), 0600)
			}}
			require.NoError(t, checkpointPod(ctx, kl, pod, "ckpt", "source-change-uid", 0))
			select {
			case <-publicationAttempted:
			case <-time.After(5 * time.Second):
				t.Fatal("checkpoint outcome publication was not attempted")
			}
			pending, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, "ckpt", metav1.GetOptions{})
			require.NoError(t, err)
			require.NotNil(t, pending.Status.CheckpointedPodTemplate)
			captured := pending.Status.CheckpointedPodTemplate.DeepCopy()
			syncPodStatus(t, kl, pod.UID)
			cached, _ := kl.statusManager.GetPodStatus(pod.UID)
			_, condition := podutil.GetPodCondition(&cached, v1.PodCheckpointing)
			require.Nil(t, condition)
			require.True(t, kl.podCheckpointInFlight(pod.UID))
			require.False(t, apimeta.IsStatusConditionTrue(pending.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady))

			changed := pod.DeepCopy()
			change.mutate(changed)
			testPods(kl).SetPods([]*v1.Pod{changed})
			kl.podProvider.SyncPodNow(ctx, changed)
			require.Equal(t, 1, captureCalls, "a source update must not repeat a completed capture")
			close(allowPublication)
			waitForCompletion(t, kl, pod.UID)

			completed, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, "ckpt", metav1.GetOptions{})
			require.NoError(t, err)
			require.True(t, apimeta.IsStatusConditionTrue(completed.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady))
			require.Equal(t, captured, completed.Status.CheckpointedPodTemplate)
			require.Equal(t, filepath.Base(outputPath), completed.Status.CheckpointLocation.NodeLocal.Path)
			data, err := os.ReadFile(filepath.Join(outputPath, "capture"))
			require.NoError(t, err)
			require.Equal(t, "original capture", string(data))
			require.Equal(t, 1, captureCalls)
		})
	}

	t.Run("rejects empty PodCheckpoint name", func(t *testing.T) {
		kubelet, _, pod := setup(t)
		err := checkpointPod(ktesting.Init(t), kubelet, pod, "", "checkpoint-uid", 0)
		require.ErrorContains(t, err, "name must not be empty")
		require.False(t, kubelet.podCheckpointInFlight(pod.UID))
	})

	t.Run("rejects pod without sandbox", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		fakeRuntime.PodStatus.SandboxStatuses = nil
		name := addCheckpoint(t, kubelet, pod, "checkpoint-uid")
		require.NoError(t, checkpointPod(ktesting.Init(t), kubelet, pod, name, "checkpoint-uid", 0))
		requireCheckpointFailure(t, kubelet, pod, name, "has no sandbox")
	})

	t.Run("rejects pod whose active sandbox is not ready", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		fakeRuntime.PodStatus.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
		name := addCheckpoint(t, kubelet, pod, "checkpoint-uid")
		require.NoError(t, checkpointPod(ktesting.Init(t), kubelet, pod, name, "checkpoint-uid", 0))
		requireCheckpointFailure(t, kubelet, pod, name, "has no ready sandbox")
	})

	t.Run("drops a request whose PodCheckpoint was deleted", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		called := false
		kubelet.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error { called = true; return nil }}
		require.NoError(t, checkpointPod(ktesting.Init(t), kubelet, pod, "deleted", "deleted-uid", 0))
		require.False(t, called)
		require.False(t, kubelet.podCheckpointInFlight(pod.UID))
		for _, action := range kubelet.kubeClient.(*fake.Clientset).Actions() {
			require.NotEqual(t, "update", action.GetVerb(), "a deleted PodCheckpoint must not receive an outcome")
		}
	})

	t.Run("drops a deferred request whose PodCheckpoint was deleted", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		name := addCheckpoint(t, kubelet, pod, "deferred-uid")
		seedCheckpointCache(t, kubelet)
		called := false
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime:          fakeRuntime,
			resizeInProgress: func(*v1.Pod, *kubecontainer.PodStatus) bool { return true },
			checkpointPod:    func(context.Context, *runtimeapi.CheckpointPodRequest) error { called = true; return nil },
		}
		ctx := ktesting.Init(t)
		require.NoError(t, checkpointPod(ctx, kubelet, pod, name, "deferred-uid", 0))
		pending := kubelet.hasPodCheckpointRequest(pod.UID)
		require.True(t, pending, "a resize in progress defers the request")

		state := kubelet.podCheckpointWatch
		cached, err := state.lister.PodCheckpoints(pod.Namespace).Get(name)
		require.NoError(t, err)
		require.NoError(t, state.indexer.Delete(cached))
		kubelet.podProvider.SyncPodNow(ctx, pod)
		pending = kubelet.hasPodCheckpointRequest(pod.UID)
		require.False(t, pending, "a deferred request must not outlive its PodCheckpoint")
		require.False(t, kubelet.podCheckpointInFlight(pod.UID))
		require.False(t, called)
	})

	t.Run("pod termination stops capture", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		ctx := ktesting.Init(t)
		syncCtx, terminate := context.WithCancel(ctx)
		defer terminate()
		kubelet.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(criCtx context.Context, _ *runtimeapi.CheckpointPodRequest) error {
			// The pod worker cancels SyncPod when the Pod starts terminating.
			terminate()
			<-criCtx.Done()
			return criCtx.Err()
		}}
		name := addCheckpoint(t, kubelet, pod, "terminated-uid")
		require.NoError(t, kubelet.requestPodCheckpoint(syncCtx, pod, &podCheckpointRequest{ctx: ctx, namespace: pod.Namespace, name: name, uid: "terminated-uid"}))
		requireCheckpointFailure(t, kubelet, pod, name, "stopped because the source pod started terminating")
	})

	t.Run("creates empty private output and retains successful checkpoint", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		pod.Spec.RuntimeClassName = new("checkpoint-runtime")
		fakeRuntime.PodStatus.SandboxStatuses[0].RuntimeHandler = "checkpoint-handler"
		requests := make(chan *runtimeapi.CheckpointPodRequest, 1)
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				entries, err := os.ReadDir(request.OutputPath)
				if err != nil {
					return fmt.Errorf("read checkpoint output directory: %w", err)
				}
				if len(entries) != 0 {
					return fmt.Errorf("checkpoint output directory is not empty")
				}
				requests <- request
				return os.WriteFile(filepath.Join(request.OutputPath, "artifact"), []byte("checkpoint"), 0o600)
			},
		}

		err := checkpointPod(ktesting.Init(t), kubelet, pod, addCheckpoint(t, kubelet, pod, "checkpoint-uid"), "checkpoint-uid", 0)
		require.NoError(t, err)
		request := receiveRequest(t, requests)
		require.Equal(t, "sandbox1234", request.PodSandboxId)
		require.Equal(t, []string{"container1234"}, request.ContainerIds)
		require.Empty(t, request.Options)
		require.True(t, filepath.IsAbs(request.OutputPath))
		require.Equal(t, "checkpoint-checkpoint-uid", filepath.Base(request.OutputPath))
		rel, err := filepath.Rel(kubelet.checkpointsDir, request.OutputPath)
		require.NoError(t, err)
		require.True(t, filepath.IsLocal(rel))
		if goruntime.GOOS != "windows" {
			info, err := os.Stat(request.OutputPath)
			require.NoError(t, err)
			require.Equal(t, os.FileMode(0o700), info.Mode().Perm())
		}

		waitForCompletion(t, kubelet, pod.UID)
		data, err := os.ReadFile(filepath.Join(request.OutputPath, "artifact"))
		require.NoError(t, err)
		require.Equal(t, []byte("checkpoint"), data)
	})

	t.Run("removes partial output after runtime failure", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		requests := make(chan *runtimeapi.CheckpointPodRequest, 1)
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				requests <- request
				if err := os.WriteFile(filepath.Join(request.OutputPath, "partial"), []byte("partial"), 0o600); err != nil {
					return fmt.Errorf("write partial checkpoint: %w", err)
				}
				return errors.New("runtime checkpoint failed")
			},
		}

		err := checkpointPod(ktesting.Init(t), kubelet, pod, addCheckpoint(t, kubelet, pod, "checkpoint-uid"), "checkpoint-uid", 0)
		require.NoError(t, err)
		request := receiveRequest(t, requests)
		waitForCompletion(t, kubelet, pod.UID)
		_, err = os.Stat(request.OutputPath)
		require.ErrorIs(t, err, os.ErrNotExist)
	})

	t.Run("fails when runtime failure output cannot be removed", func(t *testing.T) {
		if goruntime.GOOS == "windows" {
			t.Skip("the test uses Unix directory permissions to inject a cleanup failure")
		}
		kubelet, fakeRuntime, pod := setup(t)
		requests := make(chan *runtimeapi.CheckpointPodRequest, 1)
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(ctx context.Context, request *runtimeapi.CheckpointPodRequest) error {
				requests <- request
				locked := filepath.Join(request.OutputPath, "locked")
				if err := os.Mkdir(locked, 0o700); err != nil {
					return err
				}
				if err := os.WriteFile(filepath.Join(locked, "partial"), []byte("partial"), 0o600); err != nil {
					return err
				}
				if err := os.Chmod(locked, 0o500); err != nil {
					return err
				}
				return errors.New("runtime checkpoint failed")
			},
		}

		name := addCheckpoint(t, kubelet, pod, "cleanup-failure-uid")
		require.NoError(t, checkpointPod(ktesting.Init(t), kubelet, pod, name, "cleanup-failure-uid", 0))
		request := receiveRequest(t, requests)
		t.Cleanup(func() {
			_ = os.Chmod(filepath.Join(request.OutputPath, "locked"), 0o700)
			_ = os.RemoveAll(request.OutputPath)
		})
		waitForCompletion(t, kubelet, pod.UID)
		requireCheckpointFailure(t, kubelet, pod, name, "runtime checkpoint failed")
		// The data sweeper removes the output after the PodCheckpoint is deleted.
		_, err := os.Stat(request.OutputPath)
		require.NoError(t, err)
	})

	t.Run("does not call runtime when output cannot be created", func(t *testing.T) {
		if goruntime.GOOS == "windows" {
			t.Skip("Windows MkdirAll treats an existing file at the target path as success")
		}
		kubelet, fakeRuntime, pod := setup(t)
		checkpointsDir := kubelet.checkpointsDir
		require.NoError(t, os.RemoveAll(checkpointsDir))
		require.NoError(t, os.WriteFile(checkpointsDir, nil, 0o600))
		called := false
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error {
				called = true
				return nil
			},
		}

		name := addCheckpoint(t, kubelet, pod, "checkpoint-uid")
		require.NoError(t, checkpointPod(ktesting.Init(t), kubelet, pod, name, "checkpoint-uid", 0))
		requireCheckpointFailure(t, kubelet, pod, name, "failed to create pod checkpoint directory")
		require.False(t, called)
	})

	t.Run("rejects duplicate runtime container IDs", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		pod.Spec.Containers = []v1.Container{{Name: "app-one"}, {Name: "app-two"}}
		fakeRuntime.PodStatus.ContainerStatuses = []*kubecontainer.Status{
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "duplicate-id"}, Name: "app-one", State: kubecontainer.ContainerStateRunning},
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "duplicate-id"}, Name: "app-two", State: kubecontainer.ContainerStateRunning},
		}
		called := false
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error {
				called = true
				return nil
			},
		}

		name := addCheckpoint(t, kubelet, pod, "duplicate-id-uid")
		require.NoError(t, checkpointPod(ktesting.Init(t), kubelet, pod, name, "duplicate-id-uid", 0))
		requireCheckpointFailure(t, kubelet, pod, name, "duplicate runtime ID")
		require.False(t, called)
	})

	t.Run("selects regular and restartable init containers", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		always := v1.ContainerRestartPolicyAlways
		pod.Spec.InitContainers = []v1.Container{
			{Name: "completed-init"},
			{Name: "sidecar", RestartPolicy: &always},
		}
		pod.Spec.Containers = []v1.Container{{Name: "app-one"}, {Name: "app-two"}}
		pod.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debugger"}}}
		digest := "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		fakeRuntime.PodStatus.ContainerStatuses = []*kubecontainer.Status{
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "completed-id"}, Name: "completed-init", State: kubecontainer.ContainerStateExited, ExitCode: 0},
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "sidecar-id"}, Name: "sidecar", State: kubecontainer.ContainerStateRunning, ImageRef: digest},
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "app-one-id"}, Name: "app-one", State: kubecontainer.ContainerStateRunning, ImageRef: digest},
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "app-two-id"}, Name: "app-two", State: kubecontainer.ContainerStateRunning, ImageRef: digest},
			{ID: kubecontainer.ContainerID{Type: "containerd", ID: "debugger-id"}, Name: "debugger", State: kubecontainer.ContainerStateRunning},
		}
		requests := make(chan *runtimeapi.CheckpointPodRequest, 1)
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				requests <- request
				return nil
			},
		}

		err := checkpointPod(ktesting.Init(t), kubelet, pod, addCheckpoint(t, kubelet, pod, "container-selection-uid"), "container-selection-uid", 0)
		require.NoError(t, err)
		request := receiveRequest(t, requests)
		require.Equal(t, []string{"sidecar-id", "app-one-id", "app-two-id"}, request.ContainerIds)
		waitForCompletion(t, kubelet, pod.UID)
	})

	t.Run("clamps the CRI deadline to the configured ceiling", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		const ceiling = 2 * time.Second
		kubelet.kubeletConfiguration.PodCheckpointTimeout.Duration = ceiling
		deadlines := make(chan time.Time, 1)
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(ctx context.Context, _ *runtimeapi.CheckpointPodRequest) error {
				deadline, ok := ctx.Deadline()
				if !ok {
					return errors.New("CheckpointPod context has no deadline")
				}
				deadlines <- deadline
				return nil
			},
		}

		before := time.Now()
		err := checkpointPod(ktesting.Init(t), kubelet, pod, addCheckpoint(t, kubelet, pod, "deadline-uid"), "deadline-uid", time.Hour)
		after := time.Now()
		require.NoError(t, err)
		select {
		case deadline := <-deadlines:
			require.False(t, deadline.Before(before.Add(ceiling)))
			require.False(t, deadline.After(after.Add(ceiling)))
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for CheckpointPod deadline")
		}
		waitForCompletion(t, kubelet, pod.UID)
	})

	for _, outcome := range []string{"retry", "deleted", "replaced", "terminal", "canceled", "response lost"} {
		t.Run("publishes runtime failure/"+outcome, func(t *testing.T) {
			kl, fakeRuntime, pod := setup(t)
			setupCheckpoint(kl, pod, "failure-uid", false)
			client := kl.kubeClient.(*fake.Clientset)
			ctx, cancel := context.WithCancel(ktesting.Init(t))
			attempted := make(chan struct{})
			retry := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(retry) })
			t.Cleanup(func() {
				cancel()
				unblock()
				waitForCompletion(t, kl, pod.UID)
			})
			messages := make(chan string, 4)
			first := true
			client.PrependReactor("update", "podcheckpoints", func(action core.Action) (bool, apiruntime.Object, error) {
				obj := action.(core.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
				condition := apimeta.FindStatusCondition(obj.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				if condition == nil || condition.Reason != nodev1alpha1.PodCheckpointReasonFailed {
					return false, nil, nil
				}
				messages <- condition.Message
				if !first {
					return false, nil, nil
				}
				first = false
				if outcome == "response lost" {
					if err := client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), obj, pod.Namespace); err != nil {
						return true, nil, err
					}
				}
				close(attempted)
				<-retry
				return true, nil, apierrors.NewServiceUnavailable("status response unavailable")
			})
			requests := make(chan *runtimeapi.CheckpointPodRequest, 2)
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				requests <- request
				return errors.New("runtime failed")
			}}
			require.NoError(t, checkpointPod(ctx, kl, pod, "ckpt", "failure-uid", 0))
			request := receiveRequest(t, requests)
			select {
			case <-attempted:
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for failure publication")
			}
			require.True(t, kl.podCheckpointInFlight(pod.UID))
			require.ErrorIs(t, checkpointPod(ctx, kl, pod, "ckpt", "failure-uid", 0), errPodCheckpointInFlight)
			resource := nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints")
			switch outcome {
			case "deleted":
				require.NoError(t, client.Tracker().Delete(resource, pod.Namespace, "ckpt"))
			case "replaced", "terminal":
				obj, err := client.Tracker().Get(resource, pod.Namespace, "ckpt")
				require.NoError(t, err)
				pc := obj.(*nodev1alpha1.PodCheckpoint).DeepCopy()
				if outcome == "replaced" {
					pc.UID = "replacement"
				} else {
					pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonSourcePodReplaced}}
				}
				require.NoError(t, client.Tracker().Update(resource, pc, pod.Namespace))
			case "canceled":
				cancel()
			}
			unblock()
			waitForCompletion(t, kl, pod.UID)
			require.Empty(t, requests, "status retries must not repeat the CRI checkpoint")
			_, err := os.Stat(request.OutputPath)
			require.ErrorIs(t, err, os.ErrNotExist)
			close(messages)
			attempts := 0
			for message := range messages {
				require.Equal(t, "checkpoint failed: runtime failed", message)
				attempts++
			}
			if outcome == "retry" {
				require.Equal(t, 2, attempts)
			}
			if outcome == "retry" || outcome == "response lost" {
				stored, err := client.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, "ckpt", metav1.GetOptions{})
				require.NoError(t, err)
				condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				require.NotNil(t, condition)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
			}
		})
	}

	t.Run("retains successful output when status publication is canceled", func(t *testing.T) {
		kubelet, fakeRuntime, pod := setup(t)
		requests := make(chan *runtimeapi.CheckpointPodRequest, 1)
		setupCheckpoint(kubelet, pod, "status-failure-uid", false)
		kubelet.containerRuntime = &checkpointRuntime{
			Runtime: fakeRuntime,
			checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
				requests <- request
				return os.WriteFile(filepath.Join(request.OutputPath, "artifact"), []byte("checkpoint"), 0o600)
			},
		}

		ctx, cancel := context.WithCancel(ktesting.Init(t))
		t.Cleanup(cancel)
		err := checkpointPod(ctx, kubelet, pod, "ckpt", "status-failure-uid", 0)
		require.NoError(t, err)
		request := receiveRequest(t, requests)
		cancel()
		waitForCompletion(t, kubelet, pod.UID)
		_, err = os.Stat(request.OutputPath)
		require.NoError(t, err)
	})
}

func TestValidateCheckpointPodPreconditions(t *testing.T) {
	boundPod := func(spec v1.PodSpec) *v1.Pod {
		spec.NodeName = "testNode"
		return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "ns"}, Spec: spec}
	}
	running := func(name string) *kubecontainer.Status {
		return &kubecontainer.Status{Name: name, State: kubecontainer.ContainerStateRunning}
	}
	exited := func(name string, code int) *kubecontainer.Status {
		return &kubecontainer.Status{Name: name, State: kubecontainer.ContainerStateExited, ExitCode: code}
	}
	always := v1.ContainerRestartPolicyAlways

	tests := []struct {
		name      string
		pod       *v1.Pod
		statuses  []*kubecontainer.Status
		expectErr bool
	}{
		{
			name:      "all running, init completed",
			pod:       boundPod(v1.PodSpec{InitContainers: []v1.Container{{Name: "init"}}, Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{exited("init", 0), running("app")},
			expectErr: false,
		},
		{
			name:      "completed init status removed by runtime GC",
			pod:       boundPod(v1.PodSpec{InitContainers: []v1.Container{{Name: "init"}}, Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{running("app")},
			expectErr: false,
		},
		{
			name:      "not bound to a node",
			pod:       &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "ns"}, Spec: v1.PodSpec{Containers: []v1.Container{{Name: "app"}}}},
			statuses:  []*kubecontainer.Status{running("app")},
			expectErr: true,
		},
		{
			name:      "regular container not running",
			pod:       boundPod(v1.PodSpec{Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{exited("app", 0)},
			expectErr: true,
		},
		{
			name:      "init container not completed",
			pod:       boundPod(v1.PodSpec{InitContainers: []v1.Container{{Name: "init"}}, Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{running("init"), running("app")},
			expectErr: true,
		},
		{
			name:      "init container failed",
			pod:       boundPod(v1.PodSpec{InitContainers: []v1.Container{{Name: "init"}}, Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{exited("init", 1), running("app")},
			expectErr: true,
		},
		{
			name:      "restartable init (sidecar) must be running",
			pod:       boundPod(v1.PodSpec{InitContainers: []v1.Container{{Name: "sidecar", RestartPolicy: &always}}, Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{running("sidecar"), running("app")},
			expectErr: false,
		},
		{
			name:      "restartable init (sidecar) not running",
			pod:       boundPod(v1.PodSpec{InitContainers: []v1.Container{{Name: "sidecar", RestartPolicy: &always}}, Containers: []v1.Container{{Name: "app"}}}),
			statuses:  []*kubecontainer.Status{exited("sidecar", 0), running("app")},
			expectErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podStatus := &kubecontainer.PodStatus{ContainerStatuses: test.statuses}
			err := validateCheckpointPodPreconditions(test.pod, podStatus)
			if test.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

// TestSyncPodCheckpoint exercises the kubelet-side PodCheckpoint watch handler
// routing (KEP-5823): it acts only on checkpoints whose source pod this kubelet
// runs, pins the instance by UID, defers checkpoints already in flight, and on the
// precondition failure records the source UID without capturing a template from
// unavailable runtime state. TestSyncPodCheckpointPinsImages covers capture and
// the CRI call with running containers.
func TestSyncPodCheckpoint(t *testing.T) {
	const thisNode = "testNode"

	localPod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "src", Namespace: "ns", UID: "pod-uid"},
			Spec: v1.PodSpec{
				NodeName: thisNode,
				InitContainers: []v1.Container{
					// A completed non-restartable init container: not captured, so
					// it must not appear in status.checkpointedContainers.
					{Name: "init", Image: "init-img"},
					// A restartable init (sidecar) container: captured alongside the
					// regular containers in the single merged list.
					{Name: "sidecar", Image: "sidecar-img", RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways)},
				},
				Containers: []v1.Container{{Name: "app", Image: "img"}},
			},
		}
	}

	pcObj := func(readyStatus, readyReason, specUID, statusUID string) *nodev1alpha1.PodCheckpoint {
		pc := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "ckpt", Namespace: "ns", UID: "checkpoint-uid"},
			Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "src"}},
			Status:     &nodev1alpha1.PodCheckpointStatus{NodeName: ptr.To(thisNode)},
		}
		if specUID != "" {
			pc.Spec.SourcePod.UID = new(types.UID(specUID))
		}
		if readyReason != "" {
			pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionStatus(readyStatus), Reason: readyReason}}
		}
		if statusUID != "" {
			pc.Status.SourcePodUID = new(types.UID(statusUID))
		}
		return pc
	}

	tests := []struct {
		name          string
		obj           *nodev1alpha1.PodCheckpoint
		register      bool // register localPod in the pod manager
		apiPod        *v1.Pod
		inFlight      bool // pre-mark the source pod as having a checkpoint in flight
		wantReason    string
		wantStatusUID string
		wantInFlight  bool
	}{
		{name: "terminal object is skipped", obj: pcObj("True", "CheckpointCompleted", "", ""), register: true, wantReason: "CheckpointCompleted"},
		{name: "source pod not managed by this kubelet is skipped", obj: pcObj("", "", "", ""), register: false},
		{name: "assigned source pod not yet observed is skipped", obj: pcObj("", "", "", ""), apiPod: localPod()},
		{name: "source pod assigned to another node is skipped", obj: pcObj("", "", "", ""), apiPod: func() *v1.Pod { pod := localPod(); pod.Spec.NodeName = "other-node"; return pod }()},
		{name: "spec sourcePod.uid mismatch fails as replaced", obj: pcObj("", "", "other-uid", ""), register: true, wantReason: "SourcePodReplaced"},
		{name: "status sourcePodUID mismatch fails as replaced", obj: pcObj("", "", "", "other-uid"), register: true, wantReason: "SourcePodReplaced"},
		{name: "checkpoint already in flight is deferred", obj: pcObj("", "", "", ""), register: true, inFlight: true, wantInFlight: true},
		{name: "precondition failure records source UID without a template", obj: pcObj("", "", "", ""), register: true, wantReason: "CheckpointFailed", wantStatusUID: "pod-uid"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
			kl.nodeName = thisNode

			if tc.register {
				testPods(kl).SetPods([]*v1.Pod{localPod()})
			}
			if tc.inFlight {
				kl.startPodCheckpoint("pod-uid", "")
			}
			objects := []apiruntime.Object{tc.obj}
			if tc.apiPod != nil {
				objects = append(objects, tc.apiPod)
			}
			kl.kubeClient = fake.NewClientset(objects...)
			useCheckpointPodWorkers(kl)

			seedCheckpointCache(t, kl)
			err := kl.syncPodCheckpoint(tCtx, "ns/ckpt")
			if _, pending := kl.pendingPodCheckpointOutcome("ns/ckpt"); pending {
				require.True(t, kl.podCheckpointWatch.processNext(tCtx))
			}
			if tc.apiPod != nil {
				for _, action := range kl.kubeClient.(*fake.Clientset).Actions() {
					require.NotEqual(t, "pods", action.GetResource().Resource, "routing must not issue source Pod GETs")
				}
			}
			if tc.wantInFlight {
				require.ErrorIs(t, err, errPodCheckpointInFlight)
			} else {
				require.NoError(t, err)
				require.Eventually(t, func() bool { return !kl.podCheckpointInFlight("pod-uid") }, 5*time.Second, time.Millisecond)
			}

			obj, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, "ckpt", metav1.GetOptions{})
			require.NoError(t, err)

			var reason string
			if condition := apimeta.FindStatusCondition(obj.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady); condition != nil {
				reason = condition.Reason
			}

			if tc.wantReason != "" {
				require.Equal(t, tc.wantReason, reason)
			}
			require.Nil(t, obj.Status.CheckpointedPodTemplate)
			if tc.wantStatusUID != "" {
				require.Equal(t, types.UID(tc.wantStatusUID), ptr.Deref(obj.Status.SourcePodUID, ""))
			}
		})
	}
}

func TestPodCheckpointRequestSyncsCurrentPod(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
	observed := podWithUIDNameNsSpec("pod-uid", "source", "ns", v1.PodSpec{NodeName: "testNode", Containers: []v1.Container{{Name: "app"}}})
	current := observed.DeepCopy()
	current.Generation = 2
	testPods(kl).SetPods([]*v1.Pod{current})
	var synced []*v1.Pod
	kl.podProvider = &checkpointPodProvider{Manager: testPods(kl), manager: kl, onSync: func(pod *v1.Pod) { synced = append(synced, pod) }}

	require.NoError(t, kl.requestPodCheckpoint(tCtx, observed, &podCheckpointRequest{ctx: tCtx, namespace: "ns", name: "checkpoint", uid: "checkpoint-uid"}))
	require.Len(t, synced, 1)
	require.Equal(t, current.Generation, synced[0].Generation, "the pod worker must sync the Pod that the Pod manager has now")
}

func TestPodCheckpointRequestFailsForTerminatingPod(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, tc := range []struct {
		name        string
		terminating bool // termination was requested before the checkpoint
		deleted     bool // the Pod manager no longer has the Pod
	}{
		{name: "terminates after request"},
		{name: "terminating before request", terminating: true},
		{name: "deleted before request", deleted: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
			pod := podWithUIDNameNsSpec("pod-uid", "source", "ns", v1.PodSpec{NodeName: "testNode", Containers: []v1.Container{{Name: "app"}}})
			if !tc.deleted {
				testPods(kl).SetPods([]*v1.Pod{pod})
			}
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: pod.Namespace, UID: "checkpoint-uid"},
				Status:     &nodev1alpha1.PodCheckpointStatus{NodeName: new(string(kl.nodeName))},
			}
			kl.kubeClient = fake.NewClientset(pc)
			seedCheckpointCache(t, kl)
			startCheckpointOutcomeWorker(t, kl)
			syncs := 0
			kl.podProvider = &checkpointPodProvider{
				Manager:              testPods(kl),
				manager:              kl,
				terminationRequested: map[types.UID]bool{pod.UID: tc.terminating},
				onSync:               func(*v1.Pod) { syncs++ },
			}
			kl.containerRuntime = &checkpointRuntime{Runtime: testKubelet.fakeRuntime, checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error {
				t.Error("a terminating Pod must not be checkpointed")
				return nil
			}}

			require.NoError(t, kl.requestPodCheckpoint(tCtx, pod, &podCheckpointRequest{ctx: tCtx, namespace: pod.Namespace, name: pc.Name, uid: pc.UID}))
			if tc.terminating || tc.deleted {
				require.Zero(t, syncs, "a terminating or deleted Pod does not sync again")
			} else {
				require.Equal(t, 1, syncs)
				kl.AbandonPod(pod)
			}
			require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
			pending := kl.hasPodCheckpointRequest(pod.UID)
			require.False(t, pending)
			stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(tCtx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			ready := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			require.NotNil(t, ready)
			require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, ready.Reason)
			require.Equal(t, "checkpoint failed: source pod is terminating", ready.Message)
		})
	}
}

func TestFinalizeInterruptedCheckpointsRemovesOutput(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
	kl.nodeName = "testNode"

	pc := &nodev1alpha1.PodCheckpoint{
		TypeMeta: metav1.TypeMeta{APIVersion: "node.k8s.io/v1alpha1", Kind: "PodCheckpoint"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ckpt",
			Namespace: "ns",
			UID:       "interrupted-uid",
		},
		Status: &nodev1alpha1.PodCheckpointStatus{
			NodeName: new("testNode"),
			Conditions: []metav1.Condition{{
				Type:   nodev1alpha1.PodCheckpointConditionReady,
				Status: metav1.ConditionFalse,
				Reason: nodev1alpha1.PodCheckpointReasonInProgress,
			}},
		},
	}
	checkpointClient := fake.NewClientset(pc)
	listCalls := 0
	checkpointClient.PrependReactor("list", "podcheckpoints", func(core.Action) (bool, apiruntime.Object, error) {
		listCalls++
		if listCalls == 1 {
			return true, nil, errors.New("temporary list failure")
		}
		return false, nil, nil
	})
	kl.kubeClient = checkpointClient

	root, outputPath, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(root, 0o700))
	require.NoError(t, os.Mkdir(outputPath, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(outputPath, "partial"), []byte("partial"), 0o600))

	require.True(t, kl.finalizeInterruptedCheckpointsWithRetry(tCtx, time.Millisecond))
	require.Equal(t, 2, listCalls)

	_, err = os.Stat(outputPath)
	require.ErrorIs(t, err, os.ErrNotExist)
	updated, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(tCtx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	condition := apimeta.FindStatusCondition(updated.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.NotNil(t, condition)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
	require.Contains(t, condition.Message, "interrupted by a kubelet restart")
}

func TestFinalizeInterruptedCheckpointsRetriesStatus(t *testing.T) {
	for _, test := range []struct {
		name         string
		cancel       bool
		lostResponse bool
	}{
		{name: "temporary outage"},
		{name: "lost committed response", lostResponse: true},
		{name: "canceled outage", cancel: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			tCtx, cancel := context.WithCancel(ktesting.Init(t))
			defer cancel()
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
			kl.nodeName = "testNode"

			pc := &nodev1alpha1.PodCheckpoint{
				TypeMeta: metav1.TypeMeta{APIVersion: "node.k8s.io/v1alpha1", Kind: "PodCheckpoint"},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "ckpt",
					Namespace: "ns",
					UID:       "interrupted-uid",
				},
				Status: &nodev1alpha1.PodCheckpointStatus{
					NodeName: new("testNode"),
					Conditions: []metav1.Condition{{
						Type:   nodev1alpha1.PodCheckpointConditionReady,
						Status: metav1.ConditionFalse,
						Reason: nodev1alpha1.PodCheckpointReasonInProgress,
					}},
				},
			}
			checkpointClient := fake.NewClientset(pc)
			listCalls := 0
			checkpointClient.PrependReactor("list", "podcheckpoints", func(action core.Action) (bool, apiruntime.Object, error) {
				require.Equal(t, "status.nodeName=testNode", action.(core.ListAction).GetListRestrictions().Fields.String())
				listCalls++
				if listCalls == 1 {
					return true, nil, errors.New("temporary list failure")
				}
				return false, nil, nil
			})
			kl.kubeClient = checkpointClient

			updateCalls := 0
			checkpointClient.PrependReactor("update", "podcheckpoints", func(action core.Action) (bool, apiruntime.Object, error) {
				updateCalls++
				if test.cancel {
					cancel()
					return true, nil, errors.New("persistent status outage")
				}
				if updateCalls == 1 {
					if test.lostResponse {
						update := action.(core.UpdateAction)
						err := checkpointClient.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), update.GetObject(), pc.Namespace)
						require.NoError(t, err)
					}
					return true, nil, errors.New("temporary status outage")
				}
				return false, nil, nil
			})

			root, outputPath, err := kl.podCheckpointOutputPath(pc.UID)
			require.NoError(t, err)
			require.NoError(t, os.MkdirAll(root, 0o700))
			require.NoError(t, os.Mkdir(outputPath, 0o700))
			require.NoError(t, os.WriteFile(filepath.Join(outputPath, "partial"), []byte("partial"), 0o600))

			require.Equal(t, !test.cancel, kl.finalizeInterruptedCheckpointsWithRetry(tCtx, time.Millisecond))
			require.Equal(t, 2, listCalls)
			if !test.cancel {
				pending := kl.checkpointInterrupted(pc.UID)
				require.True(t, pending, "an unconfirmed failure must block capture")
				seedCheckpointCache(t, kl)
				require.NoError(t, kl.syncPodCheckpoint(tCtx, "ns/ckpt"))
				pending = kl.checkpointInterrupted(pc.UID)
				require.False(t, pending)
			}
			if test.cancel || test.lostResponse {
				require.Equal(t, 1, updateCalls)
			} else {
				require.Equal(t, 2, updateCalls)
			}

			_, err = os.Stat(outputPath)
			require.ErrorIs(t, err, os.ErrNotExist)
			updated, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(tCtx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			condition := apimeta.FindStatusCondition(updated.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			require.NotNil(t, condition)
			if test.cancel {
				require.Equal(t, nodev1alpha1.PodCheckpointReasonInProgress, condition.Reason)
			} else {
				require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
			}
		})
	}
}

func TestFinalizeInterruptedCheckpointsFailsWhenCleanupPathIsInvalid(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
	kl.nodeName = "testNode"

	pc := &nodev1alpha1.PodCheckpoint{
		TypeMeta: metav1.TypeMeta{APIVersion: "node.k8s.io/v1alpha1", Kind: "PodCheckpoint"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ckpt",
			Namespace: "ns",
			UID:       "../invalid-uid",
		},
		Status: &nodev1alpha1.PodCheckpointStatus{
			NodeName: new("testNode"),
			Conditions: []metav1.Condition{{
				Type:   nodev1alpha1.PodCheckpointConditionReady,
				Status: metav1.ConditionFalse,
				Reason: nodev1alpha1.PodCheckpointReasonInProgress,
			}},
		},
	}
	kl.kubeClient = fake.NewClientset(pc)

	require.True(t, kl.finalizeInterruptedCheckpoints(tCtx))

	updated, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(tCtx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	condition := apimeta.FindStatusCondition(updated.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.NotNil(t, condition)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
}

func TestFinalizeInterruptedCheckpointsStopsWhenListContextIsCanceled(t *testing.T) {
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
	checkpointClient := fake.NewClientset()
	listAttempted := make(chan struct{}, 1)
	checkpointClient.PrependReactor("list", "podcheckpoints", func(core.Action) (bool, apiruntime.Object, error) {
		select {
		case listAttempted <- struct{}{}:
		default:
		}
		return true, nil, errors.New("API server unavailable")
	})
	kl.kubeClient = checkpointClient

	ctx, cancel := context.WithCancel(ktesting.Init(t))
	result := make(chan bool, 1)
	go func() {
		result <- kl.finalizeInterruptedCheckpointsWithRetry(ctx, time.Millisecond)
	}()

	select {
	case <-listAttempted:
		cancel()
	case <-time.After(time.Second):
		t.Fatal("startup recovery did not attempt to list PodCheckpoints")
	}
	select {
	case completed := <-result:
		require.False(t, completed)
	case <-time.After(time.Second):
		t.Fatal("startup recovery did not stop after context cancellation")
	}
}

func TestCheckpointPodPreservesDataAfterLostStatusResponse(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	tk := newTestManager(t)
	defer tk.Cleanup()
	kl := tk.manager
	recorder := record.NewFakeRecorder(20)
	kl.recorder = recorder
	fr := tk.fakeRuntime
	kl.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
	pod := podWithUIDNameNsSpec("pod-uid", "source", "ns", v1.PodSpec{})
	pod.Spec.NodeName = "testNode"
	pod.Spec.Containers = []v1.Container{{Name: "app"}}
	testPods(kl).SetPods([]*v1.Pod{pod})
	fr.PodList = []*containertest.FakePod{{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}}}
	fr.PodStatus = kubecontainer.PodStatus{
		SandboxStatuses:   []*runtimeapi.PodSandboxStatus{{Id: "sandbox"}},
		ContainerStatuses: []*kubecontainer.Status{{ID: kubecontainer.ContainerID{Type: "test", ID: "app"}, Name: "app", State: kubecontainer.ContainerStateRunning, ImageRef: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}
	pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "cp", Namespace: "ns", UID: "cp-uid"}}
	client := fake.NewSimpleClientset(pc)
	kl.kubeClient = client
	seedCheckpointCache(t, kl)
	startCheckpointOutcomeWorker(t, kl)
	client.PrependReactor("update", "podcheckpoints", func(action core.Action) (bool, apiruntime.Object, error) {
		obj := action.(core.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
		for _, c := range obj.Status.Conditions {
			if c.Status == metav1.ConditionTrue {
				err := client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), obj, "ns")
				if err != nil {
					return true, nil, err
				}
				return true, nil, errors.New("response lost after status committed")
			}
		}
		return false, nil, nil
	})
	kl.containerRuntime = &checkpointRuntime{Runtime: fr, checkpointPod: func(_ context.Context, req *runtimeapi.CheckpointPodRequest) error {
		return os.WriteFile(filepath.Join(req.OutputPath, "data"), []byte("saved"), 0600)
	}}
	useCheckpointPodWorkers(kl)
	require.NoError(t, checkpointPod(ctx, kl, pod, "cp", "cp-uid", 0))
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	got, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "cp", metav1.GetOptions{})
	require.NoError(t, err)
	require.NotNil(t, got.Status.CheckpointLocation)
	path := filepath.Join(kl.checkpointsDir, got.Status.CheckpointLocation.NodeLocal.Path)
	if _, err := os.Stat(path); err != nil {
		t.Errorf("committed Ready=True checkpoint has no data: %v", err)
	}
	counts := map[string]int{}
	for len(recorder.Events) > 0 {
		parts := strings.SplitN(<-recorder.Events, " ", 3)
		require.Len(t, parts, 3)
		counts[parts[1]]++
	}
	require.Equal(t, 1, counts[events.PodCheckpointStarted])
	require.Equal(t, 1, counts[events.SourcePodCheckpointing])
	require.Equal(t, 1, counts[events.CheckpointSucceeded], "lost status response must not lose the completion event")
}

func TestCheckpointPublicationUsesWatchQueueRetries(t *testing.T) {
	for _, denied := range []error{
		apierrors.NewForbidden(schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, "checkpoint", errors.New("status permission denied")),
		apierrors.NewInvalid(schema.GroupKind{Group: "node.k8s.io", Kind: "PodCheckpoint"}, "checkpoint", field.ErrorList{field.Invalid(field.NewPath("status"), "invalid", "injected validation error")}),
	} {
		t.Run(string(apierrors.ReasonForError(denied)), func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, source, pc := checkpointFixture(t)
			client := kl.kubeClient.(*fake.Clientset)
			attempts := 0
			var allowed atomic.Bool
			client.PrependReactor("update", "podcheckpoints", func(core.Action) (bool, apiruntime.Object, error) {
				attempts++
				if !allowed.Load() {
					return true, nil, denied
				}
				return false, nil, nil
			})
			_, output, err := kl.createPodCheckpointOutput(pc.UID)
			require.NoError(t, err)
			require.NoError(t, os.WriteFile(filepath.Join(output, "saved"), []byte("captured state"), 0600))
			kl.startPodCheckpoint(source.UID, pc.UID)
			seedCheckpointCache(t, kl)
			state := kl.podCheckpointWatch
			state.queue = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
			defer state.queue.ShutDown()
			run := &podCheckpointRun{pod: source, namespace: pc.Namespace, name: pc.Name, uid: pc.UID, outputPath: output}
			kl.publishPodCheckpointOutcome(ctx, run, true, filepath.Base(output), "")
			key := pc.Namespace + "/" + pc.Name
			require.True(t, state.processNext(ctx))
			require.Equal(t, 1, attempts)
			require.Equal(t, 1, state.queue.NumRequeues(key))
			_, pending := kl.pendingPodCheckpointOutcome(key)
			require.True(t, pending)
			require.FileExists(t, filepath.Join(output, "saved"))
			require.True(t, kl.podCheckpointInFlight(source.UID))
			allowed.Store(true)
			state.queue.Add(key)
			require.True(t, state.processNext(ctx))
			require.Equal(t, 2, attempts)
			_, pending = kl.pendingPodCheckpointOutcome(key)
			require.False(t, pending)
			require.FileExists(t, filepath.Join(output, "saved"))
			require.False(t, kl.podCheckpointInFlight(source.UID))
		})
	}
}

func podWithUIDNameNsSpec(uid types.UID, name, namespace string, spec v1.PodSpec) *v1.Pod {
	return &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: uid, Name: name, Namespace: namespace}, Spec: spec}
}

func TestPodCheckpointStatePhases(t *testing.T) {
	kl := &manager{}
	request := &podCheckpointRequest{uid: "checkpoint-uid"}
	require.True(t, kl.startPodCheckpoint("pod-uid", "checkpoint-uid"))
	require.False(t, kl.startPodCheckpoint("pod-uid", "other-uid"), "one checkpoint per Pod")
	_, ok := kl.takePodCheckpointRequest("pod-uid")
	require.False(t, ok, "no request was handed to SyncPod yet")

	kl.setPodCheckpointRequest("pod-uid", request)
	taken, ok := kl.takePodCheckpointRequest("pod-uid")
	require.True(t, ok)
	require.Same(t, request, taken)
	_, ok = kl.takePodCheckpointRequest("pod-uid")
	require.False(t, ok, "a capturing checkpoint is not taken again")

	// A deferred capture is handed back to the next SyncPod.
	kl.setPodCheckpointRequest("pod-uid", request)
	_, ok = kl.takePodCheckpointRequest("pod-uid")
	require.True(t, ok)
	kl.setPodCheckpointRequest("pod-uid", &podCheckpointRequest{uid: "other-uid"})
	require.False(t, kl.hasPodCheckpointRequest("pod-uid"), "a request for another checkpoint is ignored")

	require.True(t, kl.podCheckpointDataInUse().Has("checkpoint-uid"))
	kl.finishPodCheckpoint("pod-uid", "other-uid")
	require.True(t, kl.podCheckpointInFlight("pod-uid"), "only the checkpoint in flight can finish it")
	kl.finishPodCheckpoint("pod-uid", "checkpoint-uid")
	require.False(t, kl.podCheckpointInFlight("pod-uid"))
	require.Empty(t, kl.podCheckpointDataInUse())
}

// testPods returns the Pod manager behind the test Pod provider.
func testPods(kl *manager) kubepod.Manager {
	return kl.podProvider.(*checkpointPodProvider).Manager
}

// useTestClient replaces the API client and rebuilds the watch around it.
func useTestClient(t *testing.T, kl *manager, client clientset.Interface) {
	t.Helper()
	kl.kubeClient = client
	kl.podCheckpointWatch = newPodCheckpointWatch(klog.FromContext(ktesting.Init(t)), kl)
	t.Cleanup(kl.podCheckpointWatch.queue.ShutDown)
}

func TestNewManagerFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, enabled)
			client := fake.NewClientset()
			m := NewManager(ktesting.Init(t), Config{KubeClient: client, NodeName: testNodeName})
			if enabled {
				real, ok := m.(*manager)
				require.True(t, ok)
				t.Cleanup(real.podCheckpointWatch.queue.ShutDown)
				return
			}
			require.IsType(t, noopManager{}, m)
			// The kubelet calls these unconditionally; they must not touch the API.
			ctx, cancel := context.WithCancel(ktesting.Init(t))
			cancel()
			m.Run(ctx)
			require.False(t, m.SyncPod(ctx, &v1.Pod{}, &kubecontainer.PodStatus{}))
			m.AbandonPod(&v1.Pod{})
			require.Empty(t, client.Actions())
		})
	}
}
