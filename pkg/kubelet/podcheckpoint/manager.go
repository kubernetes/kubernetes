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
	"sync"
	"time"

	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/kubernetes/pkg/kubelet/status"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/kubernetes/pkg/util/removeall"
	"k8s.io/mount-utils"
	"k8s.io/utils/ptr"
)

// podProvider is the subset of Kubelet methods the PodCheckpoint manager
// depends on. SyncPodNow triggers the Pod syncs that capture requested
// checkpoints.
type podProvider interface {
	GetPodByUID(types.UID) (*v1.Pod, bool)
	GetPodByName(namespace, name string) (*v1.Pod, bool)
	IsPodTerminatingOrTerminal(*v1.Pod) bool
	SyncPodNow(ctx context.Context, pod *v1.Pod)
}

// Config holds the dependencies of a manager.
type Config struct {
	KubeClient       clientset.Interface
	NodeName         types.NodeName
	Recorder         record.EventRecorderLogger
	PodProvider      podProvider
	ContainerRuntime kubecontainer.Runtime
	ProbeManager     prober.Manager
	StatusManager    status.Manager
	SourcesReady     config.SourcesReady
	// KubeletConfiguration provides podCheckpointTimeout and podCheckpointStorage.
	KubeletConfiguration *kubeletconfig.KubeletConfiguration
	// CheckpointsDir is the directory that holds checkpoint data.
	CheckpointsDir string
	// Mounter detects mount points, so that removing runtime-written
	// checkpoint data never crosses into another filesystem.
	Mounter mount.Interface
}

// manager executes the PodCheckpoints assigned to this node. The kubelet
// calls SyncPod from the Pod worker of a source Pod, so a capture does not
// overlap container operations of that Pod.
type manager struct {
	kubeClient           clientset.Interface
	nodeName             types.NodeName
	recorder             record.EventRecorderLogger
	podProvider          podProvider
	containerRuntime     kubecontainer.Runtime
	probeManager         prober.Manager
	statusManager        status.Manager
	sourcesReady         config.SourcesReady
	kubeletConfiguration *kubeletconfig.KubeletConfiguration
	checkpointsDir       string

	// stateLock guards podCheckpoints and checkpointsInterrupted.
	stateLock sync.Mutex
	// podCheckpoints holds the checkpoint in flight for each source Pod, keyed
	// by Pod UID, from the time the watch hands it to the pod worker until its
	// outcome is published. It prevents replay and keeps the checkpoint data.
	// The source PodCheckpointing condition ends earlier, when probes resume.
	// This state does not survive restart; startup fails interrupted
	// checkpoints (see finalizeInterruptedCheckpoints).
	podCheckpoints map[types.UID]*podCheckpointState

	// podCheckpointWatch is built with the manager and started by Run. It is
	// nil when the kubelet has no API client.
	podCheckpointWatch *podCheckpointWatchState
	// podCheckpointStorageInfo replaces inspectPodCheckpointStorage in tests.
	// It is nil in production.
	podCheckpointStorageInfo func(string) (podCheckpointStorageInfo, error)
	// checkpointsInterrupted holds PodCheckpoints that startup recovery could
	// not mark failed. The watch fails them instead of capturing them again.
	checkpointsInterrupted sets.Set[types.UID]
	mounter                mount.Interface
}

// podCheckpointPhase is the stage of a checkpoint in flight for a Pod.
type podCheckpointPhase int

const (
	// podCheckpointPending waits for the pod worker to call SyncPod.
	podCheckpointPending podCheckpointPhase = iota
	// podCheckpointCapturing runs in SyncPod.
	podCheckpointCapturing
	// podCheckpointPublishing waits for the watch to write the outcome.
	podCheckpointPublishing
)

// podCheckpointState is the checkpoint in flight for a Pod.
type podCheckpointState struct {
	checkpointUID types.UID
	phase         podCheckpointPhase
	// request is what SyncPod captures next. It is set only while pending.
	request *podCheckpointRequest
}

// Manager captures the PodCheckpoints assigned to this node during Pod syncs.
type Manager interface {
	// Run fails checkpoints interrupted by a kubelet restart and then executes
	// the PodCheckpoints assigned to this node until ctx is canceled.
	Run(ctx context.Context)
	// SyncPod captures the checkpoint requested for the Pod, if any. The
	// kubelet passes the allocated Pod. It reports whether the runtime was
	// asked to capture the Pod, after which podStatus may no longer describe
	// the containers.
	SyncPod(ctx context.Context, allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool
	// AbandonPod fails the checkpoint requested for a terminating Pod if
	// SyncPod has not started it.
	AbandonPod(pod *v1.Pod)
}

// NewManager returns the Manager for the Pods of this node. It returns a no-op
// Manager when PodLevelCheckpointRestore is disabled.
func NewManager(ctx context.Context, c Config) Manager {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) {
		return noopManager{}
	}
	return newManager(ctx, c)
}

func newManager(ctx context.Context, c Config) *manager {
	m := &manager{
		kubeClient:           c.KubeClient,
		nodeName:             c.NodeName,
		recorder:             c.Recorder,
		podProvider:          c.PodProvider,
		containerRuntime:     c.ContainerRuntime,
		probeManager:         c.ProbeManager,
		statusManager:        c.StatusManager,
		sourcesReady:         c.SourcesReady,
		kubeletConfiguration: c.KubeletConfiguration,
		checkpointsDir:       c.CheckpointsDir,
		mounter:              c.Mounter,
	}
	if m.kubeClient != nil {
		m.podCheckpointWatch = newPodCheckpointWatch(klog.FromContext(ctx), m)
	}
	return m
}

func (m *manager) Run(ctx context.Context) {
	m.runPodCheckpointWatch(ctx)
}

func (m *manager) SyncPod(ctx context.Context, allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	return m.syncPodCheckpointRequest(ctx, allocatedPod, podStatus)
}

func (m *manager) AbandonPod(pod *v1.Pod) {
	m.abandonPodCheckpointRequest(pod)
}

// noopManager is the Manager when PodLevelCheckpointRestore is disabled.
type noopManager struct{}

func (noopManager) Run(context.Context) {}

func (noopManager) SyncPod(context.Context, *v1.Pod, *kubecontainer.PodStatus) bool { return false }

func (noopManager) AbandonPod(*v1.Pod) {}

// podCheckpointRequest is a PodCheckpoint that the watch handed to the pod
// worker of its source Pod (KEP-5823).
type podCheckpointRequest struct {
	// ctx is the watch context. Outcome publication uses it because it
	// continues after SyncPod returns.
	ctx       context.Context
	namespace string
	name      string
	uid       types.UID
	timeout   time.Duration
}

// requestPodCheckpoint hands a PodCheckpoint to the pod worker of its source
// Pod and triggers a sync. SyncPod captures the checkpoint, so the capture does
// not overlap a container operation of the Pod, and terminating the Pod
// cancels it. A second request for the Pod returns errPodCheckpointInFlight
// until the outcome of the first one is published, so the watch can retry it.
func (m *manager) requestPodCheckpoint(ctx context.Context, pod *v1.Pod, request *podCheckpointRequest) error {
	if request.name == "" {
		return errors.New("PodCheckpoint name must not be empty")
	}
	if !m.startPodCheckpoint(pod.UID, request.uid) {
		klog.FromContext(ctx).V(2).Info("Checkpoint already in flight for pod; deferring duplicate trigger", "pod", klog.KObj(pod), "podUID", pod.UID)
		return errPodCheckpointInFlight
	}
	m.setPodCheckpointRequest(pod.UID, request)
	// The Pod may have been updated or deleted while the watch wrote the
	// in-progress status, so sync the Pod that the Pod manager has now. The
	// pod worker does not call SyncPod for a terminating Pod, and
	// SyncTerminatingPod may have looked for a request before this one.
	current, ok := m.podProvider.GetPodByUID(pod.UID)
	if !ok || m.podProvider.IsPodTerminatingOrTerminal(current) {
		m.abandonPodCheckpointRequest(pod)
		return nil
	}
	m.podProvider.SyncPodNow(ctx, current)
	return nil
}

// abandonPodCheckpointRequest fails the checkpoint requested for a terminating
// Pod if SyncPod has not started it.
func (m *manager) abandonPodCheckpointRequest(pod *v1.Pod) {
	request, ok := m.takePodCheckpointRequest(pod.UID)
	if !ok {
		return
	}
	run := &podCheckpointRun{pod: pod, namespace: request.namespace, name: request.name, uid: request.uid}
	m.publishPodCheckpointOutcome(request.ctx, run, false, "", "checkpoint failed: source pod is terminating")
}

// syncPodCheckpointRequest captures the checkpoint requested for the Pod, if
// any, from podStatus, the status that the runtime sync uses. SyncPod calls it
// before the runtime sync, so the pod worker does not start, stop, or resize
// containers of the Pod during capture. Terminating the Pod cancels ctx, which
// stops the capture. It reports whether the runtime was asked to capture the
// Pod, after which podStatus may no longer describe the containers.
func (m *manager) syncPodCheckpointRequest(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	request, ok := m.takePodCheckpointRequest(pod.UID)
	if !ok {
		return false
	}
	logger := klog.FromContext(ctx)
	run, err := m.preparePodCheckpoint(ctx, pod, podStatus, request)
	switch {
	case errors.Is(err, errPodCheckpointInFlight) || errors.Is(err, errPodCheckpointAPIUnavailable):
		if !m.podCheckpointPending(request) {
			m.finishPodCheckpoint(pod.UID, request.uid)
			return false
		}
		// The pod worker syncs a running Pod at least once per sync period, so
		// a later sync retries the request.
		logger.V(2).Info("Deferring Pod checkpoint", "pod", klog.KObj(pod), "podCheckpoint", klog.KRef(request.namespace, request.name), "err", err)
		m.setPodCheckpointRequest(pod.UID, request)
		return false
	case errors.Is(err, errPodCheckpointNotPending):
		// The PodCheckpoint was deleted, replaced, or finished elsewhere.
		m.finishPodCheckpoint(pod.UID, request.uid)
		return false
	case err != nil:
		if run == nil {
			run = &podCheckpointRun{pod: pod, namespace: request.namespace, name: request.name, uid: request.uid}
		}
		m.publishPodCheckpointOutcome(request.ctx, run, false, "", fmt.Sprintf("checkpoint failed: %v", err))
		return false
	}

	// Bound probe draining and the CRI call together so draining cannot
	// extend the requested checkpoint duration.
	criCtx, cancel := context.WithTimeout(ctx, run.timeout)
	defer cancel()
	err = m.capturePodCheckpoint(ctx, criCtx, run)
	succeeded, relLocation, message := m.finishPodCheckpointCapture(ctx, run, err)
	// Publication retries until the API server confirms the outcome. It must
	// not hold the pod worker.
	m.publishPodCheckpointOutcome(request.ctx, run, succeeded, relLocation, message)
	return true
}

// podCheckpointPending reports whether the watch cache still has the
// PodCheckpoint of a request, with the same UID and without an outcome. A
// deferred request must not outlive its PodCheckpoint.
func (m *manager) podCheckpointPending(request *podCheckpointRequest) bool {
	state := m.podCheckpointWatch
	if state == nil || state.lister == nil {
		return true
	}
	pc, err := state.lister.PodCheckpoints(request.namespace).Get(request.name)
	if err != nil {
		return !apierrors.IsNotFound(err)
	}
	return pc.UID == request.uid && !podCheckpointTerminal(pc)
}

// finishPodCheckpoint allows the next checkpoint of the Pod.
func (m *manager) finishPodCheckpoint(podUID, checkpointUID types.UID) {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	if state := m.podCheckpoints[podUID]; state != nil && state.checkpointUID == checkpointUID {
		delete(m.podCheckpoints, podUID)
	}
}

// podCheckpointRun holds one accepted Pod checkpoint: the request sent to the
// runtime and the PodCheckpoint that receives the outcome.
type podCheckpointRun struct {
	pod            *v1.Pod
	namespace      string
	name           string
	uid            types.UID
	timeout        time.Duration
	checkpointsDir string
	outputPath     string
	request        *runtimeapi.CheckpointPodRequest
}

// preparePodCheckpoint checks that the Pod can be checkpointed now, captures its
// template, and creates the output directory.
func (m *manager) preparePodCheckpoint(ctx context.Context, allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus, request *podCheckpointRequest) (*podCheckpointRun, error) {
	// SyncPod passes the allocated Pod. Keep one snapshot for validation, resize
	// detection, and the captured template while the operation runs.
	pod := allocatedPod.DeepCopy()

	podFullName := kubecontainer.GetPodFullName(pod)
	if len(podStatus.SandboxStatuses) == 0 {
		return nil, fmt.Errorf("pod %v has no sandbox", podFullName)
	}
	activeSandbox := podStatus.SandboxStatuses[0]
	if activeSandbox == nil || activeSandbox.State != runtimeapi.PodSandboxState_SANDBOX_READY || activeSandbox.Id == "" {
		return nil, fmt.Errorf("pod %v has no ready sandbox", podFullName)
	}

	// Enforce the execution-time preconditions (KEP-5823) before reaching the
	// container runtime: the pod must be bound to a node, non-restartable init
	// containers completed, and regular and restartable init containers running.
	if err := validateCheckpointPodPreconditions(pod, podStatus); err != nil {
		return nil, err
	}
	if m.containerRuntime.IsPodResizeInProgress(pod, podStatus) {
		// The runtime sync that follows finishes the resize. Until then, the
		// allocated resources in the template do not describe the containers.
		return nil, errPodCheckpointInFlight
	}
	containerIDs, err := checkpointPodContainerIDs(pod, podStatus)
	if err != nil {
		return nil, err
	}
	ceiling := m.kubeletConfiguration.PodCheckpointTimeout.Duration
	if ceiling <= 0 {
		return nil, fmt.Errorf("pod checkpoint timeout must be positive, got %v", ceiling)
	}
	timeout := request.timeout
	if timeout <= 0 || timeout > ceiling {
		timeout = ceiling
	}
	if err := m.admitPodCheckpointStorage(); err != nil {
		return nil, err
	}
	if err := m.publishPodCheckpointTemplate(ctx, request.namespace, request.name, request.uid, pod, podStatus); err != nil {
		return nil, err
	}

	checkpointsDir, outputPath, err := m.createPodCheckpointOutput(request.uid)
	if err != nil {
		return nil, err
	}
	return &podCheckpointRun{
		pod:            pod,
		namespace:      request.namespace,
		name:           request.name,
		uid:            request.uid,
		timeout:        timeout,
		checkpointsDir: checkpointsDir,
		outputPath:     outputPath,
		request: &runtimeapi.CheckpointPodRequest{
			PodSandboxId: activeSandbox.Id,
			OutputPath:   outputPath,
			ContainerIds: containerIDs,
		},
	}, nil
}

// createPodCheckpointOutput creates the private output directory for one
// PodCheckpoint and returns the checkpoint root and that directory.
func (m *manager) createPodCheckpointOutput(podCheckpointUID types.UID) (string, string, error) {
	// The kubelet owns checkpoint storage. A deterministic leaf keyed by the
	// PodCheckpoint UID lets startup recovery find and remove an interrupted
	// caller-owned directory after the kubelet restarts.
	checkpointsDir, outputPath, err := m.podCheckpointOutputPath(podCheckpointUID)
	if err != nil {
		return "", "", err
	}
	if err := utilfs.MkdirAll(checkpointsDir, 0o700); err != nil {
		return "", "", fmt.Errorf("failed to create pod checkpoint directory %q: %w", checkpointsDir, err)
	}
	if err := os.Mkdir(outputPath, 0o700); err != nil {
		return "", "", fmt.Errorf("failed to create checkpoint output directory %q: %w", outputPath, err)
	}
	// os.FileMode alone does not install a restrictive Windows ACL. Apply the
	// platform-specific permission helper after exclusive creation so the leaf
	// that receives checkpoint data is private on every supported OS.
	if err := utilfs.Chmod(outputPath, 0o700); err != nil {
		if cleanupErr := removeall.RemoveAllOneFilesystem(m.mounter, outputPath); cleanupErr != nil {
			return "", "", fmt.Errorf("failed to restrict access to checkpoint output directory %q: %w (cleanup also failed: %w)", outputPath, err, cleanupErr)
		}
		return "", "", fmt.Errorf("failed to restrict access to checkpoint output directory %q: %w", outputPath, err)
	}
	return checkpointsDir, outputPath, nil
}

// capturePodCheckpoint suspends probes and calls CheckpointPod. Probes resume
// when the call returns, including on cancellation or deadline expiry. The
// checkpointing window is removed without publishing; the next Pod sync
// publishes the cleared condition. The runtime must stop the capture and resume
// containers before returning or soon after the caller is gone.
func (m *manager) capturePodCheckpoint(ctx, criCtx context.Context, run *podCheckpointRun) error {
	podUID := run.pod.UID
	m.statusManager.SetPodCheckpointingCondition(klog.FromContext(ctx), podUID, &v1.PodCondition{
		Type: v1.PodCheckpointing, Status: v1.ConditionTrue,
		Reason: events.SourcePodCheckpointing, Message: fmt.Sprintf("Checkpointing for PodCheckpoint %q", run.name),
		ObservedGeneration: run.pod.Generation,
	})
	defer m.statusManager.SetPodCheckpointingCondition(klog.FromContext(ctx), podUID, nil)
	defer m.probeManager.ResumeProbes(podUID)
	m.recordPodCheckpointStarted(ctx, run.pod, run.namespace, run.name, run.uid)
	err := m.probeManager.SuspendProbes(criCtx, podUID)
	if err == nil {
		err = criCtx.Err()
	}
	if err == nil {
		err = m.containerRuntime.CheckpointPod(criCtx, run.request)
		if grpcstatus.Code(err) == grpccodes.Unimplemented {
			err = fmt.Errorf("runtime does not support Pod checkpoints: %w", err)
		}
	}
	switch {
	case err != nil && errors.Is(criCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil:
		err = fmt.Errorf("did not finish within the %v checkpoint timeout (spec.timeoutSeconds, limited to the node's podCheckpointTimeout): %w", run.timeout, err)
	case err != nil && ctx.Err() != nil:
		err = fmt.Errorf("stopped because the source pod started terminating: %w", err)
	}
	return err
}

// finishPodCheckpointCapture removes the output of a failed capture and returns
// the outcome to publish.
func (m *manager) finishPodCheckpointCapture(ctx context.Context, run *podCheckpointRun, err error) (succeeded bool, relLocation, message string) {
	logger := klog.FromContext(ctx)
	if err == nil {
		// Publish a path relative to the checkpoint root, never an absolute host path.
		relLocation, err = filepath.Rel(run.checkpointsDir, run.outputPath)
		if err == nil && !filepath.IsLocal(relLocation) {
			err = fmt.Errorf("computed location %q is not local to the checkpoint root", relLocation)
		}
		if err != nil {
			err = fmt.Errorf("cannot compute checkpoint location relative to %q: %w", run.checkpointsDir, err)
		}
	}
	if err != nil {
		logger.Error(err, "Pod checkpoint failed", "pod", klog.KObj(run.pod), "checkpointPath", run.outputPath)
		m.removePodCheckpointOutput(ctx, run, "checkpoint failure")
		return false, "", fmt.Sprintf("checkpoint failed: %v", err)
	}
	logger.V(2).Info("Pod checkpoint completed", "pod", klog.KObj(run.pod), "checkpointPath", run.outputPath)
	return true, relLocation, ""
}

// publishPodCheckpointOutcome writes the outcome to the PodCheckpoint status.
// An update error does not establish that the API server rejected the write,
// so it retains successful data and retries the same result until confirmed.
// The in-flight marker stays set until it returns, so watch events cannot
// repeat the capture.
func (m *manager) publishPodCheckpointOutcome(ctx context.Context, run *podCheckpointRun, succeeded bool, relLocation, message string) {
	m.stateLock.Lock()
	if state := m.podCheckpoints[run.pod.UID]; state != nil && state.checkpointUID == run.uid {
		state.phase = podCheckpointPublishing
	}
	m.stateLock.Unlock()
	go m.doPublishPodCheckpointOutcome(ctx, run, succeeded, relLocation, message)
}

func (m *manager) doPublishPodCheckpointOutcome(ctx context.Context, run *podCheckpointRun, succeeded bool, relLocation, message string) {
	defer m.finishPodCheckpoint(run.pod.UID, run.uid)
	logger := klog.FromContext(ctx)
	delay := time.Second
	lastError := ""
	for ctx.Err() == nil {
		err := m.finalizePodCheckpointOutcome(ctx, run.namespace, run.name, run.uid, succeeded, relLocation, message)
		if err == nil {
			return
		}
		if apierrors.IsNotFound(err) || errors.Is(err, errPodCheckpointNotPending) {
			// The PodCheckpoint was deleted or replaced, so nothing can use this data.
			if succeeded {
				m.removePodCheckpointOutput(ctx, run, "checkpoint object removed or replaced")
			}
			return
		}
		if err.Error() != lastError {
			logMessage := "Failed to publish PodCheckpoint result; retrying"
			switch {
			case apierrors.IsForbidden(err):
				logMessage = "Cannot publish PodCheckpoint result; check status update permissions and admission errors"
			case apierrors.IsInvalid(err):
				logMessage = "PodCheckpoint status was rejected; check the reported validation errors"
			}
			logger.Error(err, logMessage, "podCheckpoint", klog.KRef(run.namespace, run.name), "podCheckpointUID", run.uid)
			lastError = err.Error()
		}
		// The capture outcome is already known. Keep retrying the same outcome
		// without flooding an API server that continues denying the status write.
		timer := time.NewTimer(min(wait.Jitter(delay, 0.1), time.Minute))
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
		delay = min(2*delay, time.Minute)
	}
}

// removePodCheckpointOutput removes the output directory after a failed or
// abandoned checkpoint. If removal fails, the data sweeper removes the data
// once the PodCheckpoint is deleted.
func (m *manager) removePodCheckpointOutput(ctx context.Context, run *podCheckpointRun, after string) {
	if run.outputPath == "" {
		return
	}
	if err := removeall.RemoveAllOneFilesystem(m.mounter, run.outputPath); err != nil {
		klog.FromContext(ctx).Error(err, "Failed to remove checkpoint output; it is removed when the PodCheckpoint is deleted", "pod", klog.KObj(run.pod), "checkpointPath", run.outputPath, "after", after)
	}
}

// podCheckpointDataPrefix starts the name of each PodCheckpoint data directory.
const podCheckpointDataPrefix = "checkpoint-"

// podCheckpointOutputPath returns the caller-owned directory for one
// PodCheckpoint object. Kubernetes UIDs are path-safe, but validate the value
// before joining it to the checkpoint root because this path contains pod data.
func (m *manager) podCheckpointOutputPath(podCheckpointUID types.UID) (string, string, error) {
	if podCheckpointUID == "" {
		return "", "", errors.New("PodCheckpoint UID must not be empty")
	}
	directoryName := podCheckpointDataPrefix + string(podCheckpointUID)
	if !filepath.IsLocal(directoryName) || filepath.Base(directoryName) != directoryName {
		return "", "", fmt.Errorf("PodCheckpoint UID %q cannot be used as a checkpoint directory name", podCheckpointUID)
	}
	checkpointsDir, err := filepath.Abs(m.checkpointsDir)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve pod checkpoint directory %q: %w", m.checkpointsDir, err)
	}
	return checkpointsDir, filepath.Join(checkpointsDir, directoryName), nil
}

// checkpointPodContainerIDs selects the exact live container set represented by
// a Pod checkpoint. Completed init containers are no longer running, and
// ephemeral containers are not part of a Pod checkpoint.
func checkpointPodContainerIDs(pod *v1.Pod, podStatus *kubecontainer.PodStatus) ([]string, error) {
	containerIDs := make([]string, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
	containerNamesByID := make(map[string]string, cap(containerIDs))
	appendRunningID := func(name string) error {
		status := podStatus.FindContainerStatusByName(name)
		if status == nil || status.State != kubecontainer.ContainerStateRunning {
			return fmt.Errorf("cannot checkpoint pod %v: container %q is not running", klog.KObj(pod), name)
		}
		if status.ID.ID == "" {
			return fmt.Errorf("cannot checkpoint pod %v: running container %q has no runtime ID", klog.KObj(pod), name)
		}
		if existingName, duplicate := containerNamesByID[status.ID.ID]; duplicate {
			return fmt.Errorf("cannot checkpoint pod %v: containers %q and %q have duplicate runtime ID %q", klog.KObj(pod), existingName, name, status.ID.ID)
		}
		containerNamesByID[status.ID.ID] = name
		containerIDs = append(containerIDs, status.ID.ID)
		return nil
	}

	for container := range kubecontainer.CheckpointContainers(&pod.Spec) {
		if err := appendRunningID(container.Name); err != nil {
			return nil, err
		}
	}
	if len(containerIDs) == 0 {
		return nil, fmt.Errorf("cannot checkpoint pod %v: no running containers were selected", klog.KObj(pod))
	}
	return containerIDs, nil
}

// validateCheckpointPodPreconditions enforces the execution-time gate from
// KEP-5823: the pod must be bound to a node, all non-restartable init containers
// must have completed successfully, and all regular containers (plus restartable
// init/sidecar containers) must be running. Checkpoint requests that do not meet
// these preconditions are rejected before reaching the container runtime.
func validateCheckpointPodPreconditions(pod *v1.Pod, podStatus *kubecontainer.PodStatus) error {
	if pod.Spec.NodeName == "" {
		return fmt.Errorf("cannot checkpoint pod %v: pod is not bound to a node", klog.KObj(pod))
	}

	requireRunning := func(name string) error {
		cs := podStatus.FindContainerStatusByName(name)
		if cs == nil || cs.State != kubecontainer.ContainerStateRunning {
			return fmt.Errorf("cannot checkpoint pod %v: container %q is not running", klog.KObj(pod), name)
		}
		return nil
	}

	// Regular containers can only be running after every non-restartable init
	// container has completed successfully. Validate them first so a missing
	// completed-init runtime status (which runtime GC may remove) is not treated
	// as a failed precondition.
	for i := range pod.Spec.Containers {
		if err := requireRunning(pod.Spec.Containers[i].Name); err != nil {
			return err
		}
	}

	for i := range pod.Spec.InitContainers {
		c := &pod.Spec.InitContainers[i]
		if podutil.IsRestartableInitContainer(c) {
			// Restartable init (sidecar) containers run alongside the regular
			// containers, so they must be running like regular containers.
			if err := requireRunning(c.Name); err != nil {
				return err
			}
			continue
		}
		cs := podStatus.FindContainerStatusByName(c.Name)
		if cs == nil {
			continue
		}
		if cs.State != kubecontainer.ContainerStateExited || cs.ExitCode != 0 {
			return fmt.Errorf("cannot checkpoint pod %v: init container %q has not completed successfully", klog.KObj(pod), c.Name)
		}
	}

	return nil
}

// podCheckpointInFlight reports whether a checkpoint of the Pod is waiting for
// the pod worker, capturing, or publishing its outcome.
func (m *manager) podCheckpointInFlight(podUID types.UID) bool {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	_, ok := m.podCheckpoints[podUID]
	return ok
}

// startPodCheckpoint marks a checkpoint of the Pod in flight. It reports false
// if another checkpoint of the Pod is already in flight.
func (m *manager) startPodCheckpoint(podUID, checkpointUID types.UID) bool {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	if _, ok := m.podCheckpoints[podUID]; ok {
		return false
	}
	if m.podCheckpoints == nil {
		m.podCheckpoints = make(map[types.UID]*podCheckpointState)
	}
	m.podCheckpoints[podUID] = &podCheckpointState{checkpointUID: checkpointUID, phase: podCheckpointPending}
	return true
}

// setPodCheckpointRequest hands the in-flight checkpoint of the Pod to SyncPod,
// including after a capture was deferred.
func (m *manager) setPodCheckpointRequest(podUID types.UID, request *podCheckpointRequest) {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	state := m.podCheckpoints[podUID]
	if state == nil || state.checkpointUID != request.uid {
		return
	}
	state.phase = podCheckpointPending
	state.request = request
}

// takePodCheckpointRequest returns the checkpoint that SyncPod should capture
// for the Pod, if one is pending, and marks it capturing.
func (m *manager) takePodCheckpointRequest(podUID types.UID) (*podCheckpointRequest, bool) {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	state := m.podCheckpoints[podUID]
	if state == nil || state.phase != podCheckpointPending || state.request == nil {
		return nil, false
	}
	request := state.request
	state.phase = podCheckpointCapturing
	state.request = nil
	return request, true
}

func (m *manager) setCheckpointInterrupted(checkpointUID types.UID, interrupted bool) {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	if !interrupted {
		m.checkpointsInterrupted.Delete(checkpointUID)
		return
	}
	if m.checkpointsInterrupted == nil {
		m.checkpointsInterrupted = sets.New[types.UID]()
	}
	m.checkpointsInterrupted.Insert(checkpointUID)
}

func (m *manager) checkpointInterrupted(checkpointUID types.UID) bool {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	return m.checkpointsInterrupted.Has(checkpointUID)
}

// finalizePodCheckpointOutcome writes the terminal outcome while preserving the
// routing assignment and captured template. It retries conflicts and checks the
// live UID and terminal status before updating the object. It also reports an
// outcome observed after a lost update response.
func (m *manager) finalizePodCheckpointOutcome(ctx context.Context, namespace, name string, checkpointUID types.UID, success bool, checkpointLocation, failureMessage string) error {
	if m.kubeClient == nil {
		return errors.New("cannot finalize PodCheckpoint: kubelet has no API client")
	}

	status := metav1.ConditionTrue
	reason := nodev1alpha1.PodCheckpointReasonCompleted
	message := "checkpoint completed successfully"
	if !success {
		status = metav1.ConditionFalse
		reason = nodev1alpha1.PodCheckpointReasonFailed
		message = failureMessage
	}

	ctx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
	defer cancel()
	recordOutcome := false
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		pc, err := m.kubeClient.NodeV1alpha1().PodCheckpoints(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		checkpointStatus := pc.Status
		if checkpointStatus == nil {
			checkpointStatus = &nodev1alpha1.PodCheckpointStatus{}
		}
		if pc.UID != checkpointUID || (checkpointStatus.NodeName != nil && *checkpointStatus.NodeName != string(m.nodeName)) {
			return errPodCheckpointNotPending
		}
		if podCheckpointTerminal(pc) {
			condition := apimeta.FindStatusCondition(checkpointStatus.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			matches := condition != nil && condition.Status == status && condition.Reason == reason
			if success && condition != nil && condition.Status == metav1.ConditionTrue &&
				checkpointStatus.CheckpointLocation != nil && checkpointStatus.CheckpointLocation.NodeLocal != nil &&
				checkpointStatus.CheckpointLocation.NodeLocal.Path == checkpointLocation {
				recordOutcome = matches && pc.DeletionTimestamp == nil
				return nil
			}
			if !success && matches && condition.Message == message {
				recordOutcome = pc.DeletionTimestamp == nil
				return nil
			}
			return errPodCheckpointNotPending
		}

		pc.Status = checkpointStatus
		apimeta.SetStatusCondition(&pc.Status.Conditions, metav1.Condition{
			Type:               nodev1alpha1.PodCheckpointConditionReady,
			Status:             status,
			Reason:             reason,
			Message:            message,
			ObservedGeneration: pc.Generation,
		})
		pc.Status.CompletionTime = new(metav1.Now())
		if success {
			if checkpointLocation != "" {
				pc.Status.CheckpointLocation = &nodev1alpha1.CheckpointSource{
					Type:      nodev1alpha1.CheckpointSourceTypeNodeLocal,
					NodeLocal: &nodev1alpha1.NodeLocalCheckpointSource{Path: checkpointLocation},
				}
			}
		}

		_, err = m.kubeClient.NodeV1alpha1().PodCheckpoints(namespace).UpdateStatus(ctx, pc, metav1.UpdateOptions{})
		if err == nil && pc.DeletionTimestamp == nil {
			recordOutcome = true
		}
		return err
	})
	if err == nil && recordOutcome {
		eventType, eventReason := v1.EventTypeNormal, events.CheckpointSucceeded
		eventMessage := "Checkpoint completed successfully"
		if !success {
			eventType, eventReason = v1.EventTypeWarning, events.CheckpointFailed
			eventMessage = "Checkpoint failed. See PodCheckpoint status for details"
		}
		m.recordPodCheckpointEvent(ctx, namespace, name, checkpointUID, eventType, eventReason, eventMessage)
	}
	return err
}

// finalizeInterruptedCheckpoints runs on kubelet startup, before the
// PodCheckpoint watch. A checkpoint that is still in progress on this node was
// interrupted by the restart. The runtime stops a CheckpointPod call whose
// caller is gone and resumes the containers, so the kubelet fails the
// checkpoint and removes its partial output rather than capturing it again.
// The initial list is retried until it succeeds or ctx is canceled, so the
// watch cannot start a capture of an interrupted checkpoint. It reports
// whether the watch can start.
func (m *manager) finalizeInterruptedCheckpoints(ctx context.Context) bool {
	return m.finalizeInterruptedCheckpointsWithRetry(ctx, 5*time.Second)
}

func (m *manager) finalizeInterruptedCheckpointsWithRetry(ctx context.Context, retryPeriod time.Duration) bool {
	logger := klog.FromContext(ctx)
	if m.kubeClient == nil {
		return false
	}
	var list *nodev1alpha1.PodCheckpointList
	if err := wait.PollUntilContextCancel(ctx, retryPeriod, true, func(ctx context.Context) (bool, error) {
		var err error
		requestCtx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
		defer cancel()
		list, err = m.kubeClient.NodeV1alpha1().PodCheckpoints(metav1.NamespaceAll).List(requestCtx, metav1.ListOptions{
			FieldSelector: fields.OneTermEqualSelector("status.nodeName", string(m.nodeName)).String(),
		})
		switch {
		case err == nil:
			return true, nil
		case apierrors.IsNotFound(err) || apierrors.IsForbidden(err) || apierrors.IsUnauthorized(err):
			return false, err
		default:
			logger.Error(err, "Failed to list PodCheckpoints for startup checkpoint recovery; retrying before starting the watch")
			return false, nil
		}
	}); err != nil {
		if ctx.Err() != nil {
			logger.V(2).Info("Stopped startup checkpoint recovery before the PodCheckpoint list succeeded", "err", err)
			return false
		}
		// These errors last until the API server configuration or the node's
		// credentials change, and the watch informer retries the same list with
		// backoff. The data sweeper removes output of checkpoints deleted meanwhile.
		logger.Error(err, "Cannot list PodCheckpoints for startup checkpoint recovery; starting the watch without it. Check that the API server serves node.k8s.io/v1alpha1, enables PodLevelCheckpointRestore and authorizes this node")
		return true
	}
	for i := range list.Items {
		pc := &list.Items[i]
		// Only this node's checkpoints, and only ones still recorded in progress.
		if pc.Status == nil || ptr.Deref(pc.Status.NodeName, "") != string(m.nodeName) {
			continue
		}
		cond := apimeta.FindStatusCondition(pc.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
		if cond == nil || cond.Status != metav1.ConditionFalse || cond.Reason != nodev1alpha1.PodCheckpointReasonInProgress {
			continue
		}
		if _, outputPath, err := m.podCheckpointOutputPath(pc.UID); err != nil {
			logger.Error(err, "Failed to resolve interrupted checkpoint output", "podCheckpoint", klog.KObj(pc))
		} else if err := removeall.RemoveAllOneFilesystem(m.mounter, outputPath); err != nil {
			// The data sweeper removes it once the PodCheckpoint is deleted.
			logger.Error(err, "Failed to remove output of interrupted checkpoint", "podCheckpoint", klog.KObj(pc))
		}
		err := m.writePodCheckpointStatus(ctx, pc.Namespace, pc.Name, pc.UID, metav1.ConditionFalse,
			nodev1alpha1.PodCheckpointReasonFailed, "checkpoint failed: interrupted by a kubelet restart", nil, pc.Status.SourcePodUID)
		if err != nil && !apierrors.IsNotFound(err) && !errors.Is(err, errPodCheckpointNotPending) {
			// The watch observes the object again and retries the failure.
			m.setCheckpointInterrupted(pc.UID, true)
			logger.Error(err, "Failed to fail interrupted checkpoint; retrying from the watch", "podCheckpoint", klog.KObj(pc))
		}
		if ctx.Err() != nil {
			return false
		}
	}
	return ctx.Err() == nil
}
