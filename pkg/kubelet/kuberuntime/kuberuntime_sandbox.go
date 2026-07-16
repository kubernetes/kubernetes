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

package kuberuntime

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	netutils "k8s.io/utils/net"
)

// preparePodSandboxConfig generates the sandbox config and creates the Pod log
// directory. Both new and restored sandboxes use it. On failure, it returns a
// message for the Pod's sync result.
func (m *kubeGenericRuntimeManager) preparePodSandboxConfig(ctx context.Context, pod *v1.Pod, attempt uint32, runtimeHandler string) (*runtimeapi.PodSandboxConfig, string, error) {
	logger := klog.FromContext(ctx)
	podSandboxConfig, err := m.generatePodSandboxConfigWithRuntimeHandler(ctx, pod, attempt, runtimeHandler)
	if err != nil {
		message := fmt.Sprintf("Failed to generate sandbox config for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to generate sandbox config for pod", "pod", klog.KObj(pod))
		return nil, message, err
	}

	// Create pod logs directory
	err = m.osInterface.MkdirAll(podSandboxConfig.LogDirectory, 0755)
	if err != nil {
		message := fmt.Sprintf("Failed to create log directory for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to create log directory for pod", "pod", klog.KObj(pod))
		return nil, message, err
	}
	return podSandboxConfig, "", nil
}

// createPodSandbox creates a pod sandbox and returns (podSandBoxID, message, error).
func (m *kubeGenericRuntimeManager) createPodSandbox(ctx context.Context, pod *v1.Pod, attempt uint32) (string, string, error) {
	logger := klog.FromContext(ctx)
	runtimeHandler, err := m.lookupRuntimeHandler(pod)
	if err != nil {
		message := fmt.Sprintf("Failed to create sandbox for pod %q: %v", format.Pod(pod), err)
		return "", message, err
	}
	if runtimeHandler != "" {
		logger.V(2).Info("Running pod with runtime handler", "pod", klog.KObj(pod), "runtimeHandler", runtimeHandler)
	}

	podSandboxConfig, message, err := m.preparePodSandboxConfig(ctx, pod, attempt, runtimeHandler)
	if err != nil {
		return "", message, err
	}

	podSandBoxID, err := m.runtimeService.RunPodSandbox(ctx, podSandboxConfig, runtimeHandler)
	if err != nil {
		message := fmt.Sprintf("Failed to create sandbox for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to create sandbox for pod", "pod", klog.KObj(pod))
		return "", message, err
	}

	return podSandBoxID, "", nil
}

// acquireRestore records that a sandbox restore is starting for the given pod
// (keyed by namespace/name). It returns false if a restore for that pod is
// already in flight, so the caller can reject the second attempt instead of
// racing the first. It is keyed by namespace/name rather than UID because each
// restore attempt is admitted under a fresh pod UID, so a UID key would never
// collide.
func (m *kubeGenericRuntimeManager) acquireRestore(key string) bool {
	m.restoresInFlightLock.Lock()
	defer m.restoresInFlightLock.Unlock()
	if _, inFlight := m.restoresInFlight[key]; inFlight {
		return false
	}
	m.restoresInFlight[key] = struct{}{}
	return true
}

// releaseRestore clears the in-flight restore marker for the given pod key.
func (m *kubeGenericRuntimeManager) releaseRestore(key string) {
	m.restoresInFlightLock.Lock()
	defer m.restoresInFlightLock.Unlock()
	delete(m.restoresInFlight, key)
}

type restoredPodSandbox struct {
	id     string
	status *runtimeapi.PodSandboxStatus
}

// restorePodSandbox uses the restoring Pod's namespace and security settings.
// Networking and hosts setup finish before restored processes resume.
func (m *kubeGenericRuntimeManager) restorePodSandbox(ctx context.Context, pod *v1.Pod, attempt uint32, pullSecrets []v1.Secret) (_ *restoredPodSandbox, _ string, retErr error) {
	logger := klog.FromContext(ctx)

	// Preparation can retry safely until the first CRI restore call. After that
	// call, an error may hide runtime side effects and must remain terminal.
	restoreStarted := false
	defer func() {
		if retErr != nil && !restoreStarted && kubecontainer.RestoreErrorReason(retErr) == "" {
			retErr = &kubecontainer.RestoreError{Reason: events.RestorePreparationFailed, Err: retErr}
		}
	}()

	if pod.Spec.RestoreFrom == nil || pod.Spec.RestoreFrom.Name == "" {
		message := "spec.restoreFrom.name is not specified"
		logger.Error(nil, message, "pod", klog.KObj(pod))
		return nil, message, errors.New(message)
	}

	// Gate concurrent restores of the same pod. The CRI RestorePod call below
	// can be long-running, so reject a second restore for this pod while one is
	// already in flight rather than racing two restores into the same pod
	// sandbox. Keyed by namespace/name because each restore attempt is admitted
	// under a fresh pod UID.
	restoreKey := pod.Namespace + "/" + pod.Name
	if !m.acquireRestore(restoreKey) {
		// Another restore for the same (namespace, name) holds the lock. Record
		// the blocked state so PodRestored=Unknown explains that this Pod is
		// waiting for another restore while it retries (KEP-5823).
		m.runtimeHelper.SetPodRestoreBlocked(pod.UID, true)
		message := fmt.Sprintf("A restore is already in progress for pod %q", format.Pod(pod))
		logger.Info(message, "pod", klog.KObj(pod), "podUID", pod.UID)
		return nil, message, &kubecontainer.RestoreError{Reason: events.RestoreInProgress, Err: fmt.Errorf("restore already in progress for pod %s", format.Pod(pod))}
	}
	defer m.releaseRestore(restoreKey)
	// This Pod now holds the restore lock, so its progress message no longer
	// reports waiting for another restore. PodRestored remains Unknown.
	m.runtimeHelper.SetPodRestoreBlocked(pod.UID, false)

	// Resolve spec.restoreFrom.name (the name of a PodCheckpoint in the pod's
	// namespace) to the on-node checkpoint directory.
	checkpointPath, err := m.runtimeHelper.ValidateRestoreCheckpoint(ctx, pod)
	if err != nil {
		message := fmt.Sprintf("Failed to resolve checkpoint for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to resolve checkpoint directory for restore", "pod", klog.KObj(pod), "checkpointName", pod.Spec.RestoreFrom.Name)
		return nil, message, err
	}

	runtimeHandler := ""
	if m.runtimeClassManager != nil {
		runtimeHandler, err = m.runtimeClassManager.ValidateRestoreOptions(pod.Spec.RuntimeClassName, pod.Spec.RestoreFrom.Options)
		if err != nil {
			message := fmt.Sprintf("Failed to restore sandbox for pod %q: %v", format.Pod(pod), err)
			// As on the normal sandbox path, a RuntimeClass missing from the
			// cache is retried, because the cache may not have received it yet.
			if len(pod.Spec.RestoreFrom.Options) != 0 && !apierrors.IsNotFound(err) {
				reason := events.PodSpecMismatch
				if errors.Is(err, runtimeclass.ErrNotSynced) {
					reason = events.RestorePolicyUnavailable
				}
				err = &kubecontainer.RestoreError{Reason: reason, Err: err}
			}
			return nil, message, err
		}
		if runtimeHandler != "" {
			logger.V(2).Info("Restoring pod with runtime handler", "pod", klog.KObj(pod), "runtimeHandler", runtimeHandler)
		}
	} else if len(pod.Spec.RestoreFrom.Options) != 0 {
		err := fmt.Errorf("cannot validate spec.restoreFrom.options without a RuntimeClass manager")
		return nil, err.Error(), &kubecontainer.RestoreError{Reason: events.PodSpecMismatch, Err: err}
	}

	podSandboxConfig, message, err := m.preparePodSandboxConfig(ctx, pod, attempt, runtimeHandler)
	if err != nil {
		return nil, message, err
	}

	logger.V(2).Info("Restoring pod sandbox from checkpoint", "pod", klog.KObj(pod), "checkpointName", pod.Spec.RestoreFrom.Name, "checkpointPath", checkpointPath, "runtimeHandler", runtimeHandler)

	// The sandbox config (including namespace options and security context) is
	// derived from the pod spec by generatePodSandboxConfig, exactly as for
	// createPodSandbox. Because the restoring pod's spec is validated to match
	// the checkpoint, its ShareProcessNamespace/HostPID settings reproduce the
	// process-namespace topology that was captured. We deliberately do not
	// override the PID namespace here.

	// Generate container configs with mount information for all containers
	// This tells the CRI runtime where to mount host paths into the restored containers
	if err := m.pullRestoreContainerImages(ctx, pod, podSandboxConfig, runtimeHandler, pullSecrets); err != nil {
		return nil, err.Error(), err
	}
	imageVolumePullResults, err := m.getImageVolumesWithRuntimeHandler(ctx, pod, podSandboxConfig, pullSecrets, runtimeHandler)
	if err != nil {
		message := fmt.Sprintf("Failed to prepare image volumes for restore for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to prepare image volumes for pod restore", "pod", klog.KObj(pod))
		return nil, message, err
	}
	containerConfigs, cleanupAction, err := m.generateContainerConfigsForRestore(ctx, pod, podSandboxConfig, imageVolumePullResults, runtimeHandler)
	if cleanupAction != nil {
		defer cleanupAction()
	}
	if err != nil {
		message := fmt.Sprintf("Failed to generate container configs for restore for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to generate container configs for restore", "pod", klog.KObj(pod))
		return nil, message, err
	}

	logger.V(1).Info("Generated container configs for restore", "pod", klog.KObj(pod), "containerCount", len(containerConfigs))

	// Create the RestorePodRequest.
	restoreRequest := &runtimeapi.RestorePodRequest{
		CheckpointPath:   checkpointPath,
		Config:           podSandboxConfig,
		RuntimeHandler:   runtimeHandler,
		Options:          maps.Clone(pod.Spec.RestoreFrom.Options),
		ContainerConfigs: containerConfigs,
	}

	// After a kubelet restart, this record separates a restore that may have
	// reached the runtime from one that can start again.
	if err := m.runtimeHelper.MarkPodRestoreStarted(pod.UID); err != nil {
		message := fmt.Sprintf("Failed to record restore start for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to record restore start", "pod", klog.KObj(pod))
		return nil, message, err
	}

	// Call the CRI RestorePod RPC. Only this call gets podRestoreTimeout;
	// without a deadline, the CRI client would apply its shorter default.
	restoreStarted = true
	restoreCtx, cancelRestore := context.WithTimeout(ctx, m.podRestoreTimeout)
	restoreResponse, err := m.runtimeService.RestorePod(restoreCtx, restoreRequest)
	restoreTimedOut := errors.Is(restoreCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil
	cancelRestore()
	if err != nil {
		if restoreTimedOut {
			err = fmt.Errorf("RestorePod did not finish within %v (the node's podRestoreTimeout): %w", m.podRestoreTimeout, err)
		}
		message := fmt.Sprintf("Failed to restore pod sandbox from checkpoint %q for pod %q: %v", checkpointPath, format.Pod(pod), err)
		logger.Error(err, "Failed to restore pod sandbox from checkpoint", "pod", klog.KObj(pod), "checkpointPath", checkpointPath)
		return nil, message, err
	}

	containerIDsByName, err := validateRestorePodResponse(restoreRequest, restoreResponse)
	if err != nil {
		message := fmt.Sprintf("RestorePod returned an invalid response for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "RestorePod returned an invalid response", "pod", klog.KObj(pod), "checkpointPath", checkpointPath)
		if cleanupErr := m.cleanupRestoredPod(ctx, restoreResponse.GetPodSandboxId()); cleanupErr != nil {
			logger.Error(cleanupErr, "Failed to clean up restored pod after invalid RestorePod response", "pod", klog.KObj(pod), "podSandboxID", restoreResponse.GetPodSandboxId())
			err = errors.Join(err, fmt.Errorf("failed to clean up invalid restore: %w", cleanupErr))
		}
		return nil, message, err
	}
	podSandBoxID := restoreResponse.GetPodSandboxId()
	sandboxStatus, err := m.prepareRestoredPodHosts(ctx, pod, podSandBoxID)
	if err != nil {
		message := fmt.Sprintf("Failed to prepare hosts for restored pod %q: %v", format.Pod(pod), err)
		if cleanupErr := m.cleanupRestoredPod(ctx, podSandBoxID); cleanupErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to clean up unstarted restore: %w", cleanupErr))
		}
		return nil, message, err
	}

	// RestorePod returns prepared containers without executing their restored
	// processes. Run PreStart and StartContainer in deterministic Pod-spec order,
	// matching the normal container lifecycle. PostStart is intentionally not
	// rerun because it already executed before the checkpoint was captured.
	configIndex := 0
	for container, containerType := range kubecontainer.CheckpointContainers(&pod.Spec) {
		metricLabel := metrics.Container
		if containerType == podutil.InitContainers {
			metricLabel = metrics.InitContainer
		}
		containerID := containerIDsByName[container.Name]
		message, err := m.startRestoredContainer(ctx, pod, container, containerID, containerConfigs[configIndex], podSandboxConfig, metricLabel)
		configIndex++
		if err != nil {
			if cleanupErr := m.cleanupRestoredPod(ctx, podSandBoxID); cleanupErr != nil {
				logger.Error(cleanupErr, "Failed to clean up restored pod after container start failure", "pod", klog.KObj(pod), "podSandboxID", podSandBoxID)
				err = errors.Join(err, fmt.Errorf("failed to clean up rejected restore: %w", cleanupErr))
			}
			return nil, message, err
		}
	}

	logger.V(2).Info("Successfully restored pod sandbox from checkpoint", "pod", klog.KObj(pod), "podSandboxID", podSandBoxID, "checkpointName", pod.Spec.RestoreFrom.Name, "checkpointPath", checkpointPath)

	return &restoredPodSandbox{id: podSandBoxID, status: sandboxStatus}, "", nil
}

func (m *kubeGenericRuntimeManager) startRestoredContainer(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID string, containerConfig *runtimeapi.ContainerConfig, sandboxConfig *runtimeapi.PodSandboxConfig, metricLabel string) (string, error) {
	countFailure := countContainerStart(pod, container, metricLabel)
	msg, syncErr, err := m.startCreatedContainer(ctx, pod, container, containerID, containerConfig, sandboxConfig)
	if syncErr != nil {
		countFailure(syncErr)
		message := fmt.Sprintf("Failed to start restored container %q in pod %q: %s", container.Name, format.Pod(pod), msg)
		return message, fmt.Errorf("%w: %w", syncErr, err)
	}
	return "", nil
}

// prepareRestoredPodHosts finishes node-local network preparation before any
// restored process can read its hosts file. Failures after RestorePod cannot use
// normal container reconciliation to resume the prepared containers safely.
func (m *kubeGenericRuntimeManager) prepareRestoredPodHosts(ctx context.Context, pod *v1.Pod, sandboxID string) (*runtimeapi.PodSandboxStatus, error) {
	response, err := m.runtimeService.PodSandboxStatus(ctx, sandboxID, false)
	if err != nil {
		return nil, fmt.Errorf("failed to get restored sandbox status: %w", err)
	}
	sandboxStatus := response.GetStatus()
	if sandboxStatus == nil {
		return nil, errors.New("restored sandbox status is nil")
	}
	if sandboxStatus.State != runtimeapi.PodSandboxState_SANDBOX_READY {
		return nil, errors.New("restored sandbox is not ready")
	}
	var podIPs []string
	if !kubecontainer.IsHostNetworkPod(pod) {
		podIPs = m.determinePodSandboxIPs(ctx, pod.Namespace, pod.Name, sandboxStatus)
		if len(podIPs) == 0 {
			return nil, errors.New("restored sandbox has no Pod IP")
		}
	}
	if err := m.runtimeHelper.UpdatePodRestoreHosts(ctx, pod, podIPs); err != nil {
		return nil, fmt.Errorf("failed to update restored Pod hosts: %w", err)
	}
	return sandboxStatus, nil
}

// validateRestorePodResponse verifies that the runtime returned one non-empty,
// unique container ID for exactly every container configuration in the request.
func validateRestorePodResponse(request *runtimeapi.RestorePodRequest, response *runtimeapi.RestorePodResponse) (map[string]string, error) {
	if response == nil {
		return nil, errors.New("response is nil")
	}
	if response.GetPodSandboxId() == "" {
		return nil, errors.New("pod sandbox ID is empty")
	}

	expectedNames := make(map[string]struct{}, len(request.GetContainerConfigs()))
	for i, config := range request.GetContainerConfigs() {
		name := config.GetMetadata().GetName()
		if name == "" {
			return nil, fmt.Errorf("request container_configs[%d] has an empty metadata name", i)
		}
		if _, duplicate := expectedNames[name]; duplicate {
			return nil, fmt.Errorf("request container_configs contains duplicate metadata name %q", name)
		}
		expectedNames[name] = struct{}{}
	}
	if len(expectedNames) == 0 {
		return nil, errors.New("request container_configs is empty")
	}
	if len(response.GetRestoredContainers()) != len(expectedNames) {
		return nil, fmt.Errorf("restored container count %d does not match requested count %d", len(response.GetRestoredContainers()), len(expectedNames))
	}

	containerIDsByName := make(map[string]string, len(expectedNames))
	containerNamesByID := make(map[string]string, len(expectedNames))
	for i, restored := range response.GetRestoredContainers() {
		if restored == nil {
			return nil, fmt.Errorf("restored_containers[%d] is nil", i)
		}
		name := restored.GetName()
		if name == "" {
			return nil, fmt.Errorf("restored_containers[%d] has an empty name", i)
		}
		if _, expected := expectedNames[name]; !expected {
			return nil, fmt.Errorf("restored_containers[%d] has unexpected name %q", i, name)
		}
		if _, duplicate := containerIDsByName[name]; duplicate {
			return nil, fmt.Errorf("restored_containers contains duplicate name %q", name)
		}
		containerID := restored.GetContainerId()
		if containerID == "" {
			return nil, fmt.Errorf("restored container %q has an empty container ID", name)
		}
		if existingName, duplicate := containerNamesByID[containerID]; duplicate {
			return nil, fmt.Errorf("restored containers %q and %q have duplicate container ID %q", existingName, name, containerID)
		}
		containerIDsByName[name] = containerID
		containerNamesByID[containerID] = name
	}
	return containerIDsByName, nil
}

func (m *kubeGenericRuntimeManager) cleanupRestoredPod(ctx context.Context, podSandboxID string) error {
	if podSandboxID == "" {
		return nil
	}
	cleanupCtx := context.WithoutCancel(ctx)
	if err := m.runtimeService.StopPodSandbox(cleanupCtx, podSandboxID); err != nil {
		return err
	}
	containers, err := m.runtimeService.ListContainers(cleanupCtx, &runtimeapi.ContainerFilter{PodSandboxId: podSandboxID})
	if err != nil {
		return err
	}
	var removalErrors []error
	for _, container := range containers {
		// Keep failed-restore logs for diagnosis, and run the normal post-stop
		// hook before the runtime forgets the container ID needed for cleanup.
		if err := m.removeContainer(cleanupCtx, container.Id, true); err != nil {
			removalErrors = append(removalErrors, err)
		}
	}
	if len(removalErrors) > 0 {
		// Leave failed removals visible to ordinary container garbage collection.
		return errors.Join(removalErrors...)
	}
	return m.runtimeService.RemovePodSandbox(cleanupCtx, podSandboxID)
}

// pullRestoreContainerImages applies the Pod's pull policy and credentials
// before generating the same image-derived security context used at creation.
func (m *kubeGenericRuntimeManager) pullRestoreContainerImages(ctx context.Context, pod *v1.Pod, sandboxConfig *runtimeapi.PodSandboxConfig, runtimeHandler string, pullSecrets []v1.Secret) error {
	for container := range kubecontainer.CheckpointContainers(&pod.Spec) {
		ref, err := kubecontainer.GenerateContainerRef(pod, container)
		if err != nil {
			return fmt.Errorf("failed to reference restore container %q: %w", container.Name, err)
		}
		if _, message, err := m.imagePuller.EnsureImageExists(ctx, ref, pod, container.Image, pullSecrets, sandboxConfig, runtimeHandler, container.ImagePullPolicy); err != nil {
			return fmt.Errorf("failed to prepare image for restore container %q: %s: %w", container.Name, message, err)
		}
	}
	return nil
}

// generateContainerConfigsForRestore generates the same complete container
// configurations used by the normal start path. The runtime uses the base image
// and applies restore-time security, resource, logging, device, and mount settings.
// The checkpoint supplies the process state, including its argv and environment.
func (m *kubeGenericRuntimeManager) generateContainerConfigsForRestore(ctx context.Context, pod *v1.Pod, podSandboxConfig *runtimeapi.PodSandboxConfig, imageVolumePullResults imageVolumePulls, runtimeHandler string) ([]*runtimeapi.ContainerConfig, func(), error) {
	logger := klog.FromContext(ctx)

	var containerConfigs []*runtimeapi.ContainerConfig
	var cleanupActions []func()
	cleanup := func() {
		for _, action := range slices.Backward(cleanupActions) {
			action()
		}
	}

	for container, containerType := range kubecontainer.CheckpointContainers(&pod.Spec) {
		syncResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, container.Name)
		imageVolumes, err := m.toKubeContainerImageVolumes(ctx, imageVolumePullResults, container, pod, syncResult)
		if err != nil {
			cleanup()
			return nil, nil, fmt.Errorf("failed to prepare image volumes for container %s: %w", container.Name, err)
		}

		config, cleanupAction, msg, err := m.createContainerConfig(ctx, pod, container, containerType, 0, "", container.Image, nil, nil, imageVolumes, runtimeHandler)
		if cleanupAction != nil {
			cleanupActions = append(cleanupActions, cleanupAction)
		}
		if err != nil {
			cleanup()
			return nil, nil, fmt.Errorf("container %s: %w: %s", container.Name, err, msg)
		}

		// A restored sandbox has no Pod IP yet, so the normal options helper may
		// omit /etc/hosts. Preserve kubelet ownership of that node-local mount.
		hasHostsMount := false
		for _, mount := range config.Mounts {
			if mount.ContainerPath == "/etc/hosts" {
				hasHostsMount = true
				break
			}
		}
		if !hasHostsMount {
			hostsPath := filepath.Join(m.runtimeHelper.GetPodDir(pod.UID), "etc-hosts")
			config.Mounts = append(config.Mounts, &runtimeapi.Mount{
				HostPath:       hostsPath,
				ContainerPath:  "/etc/hosts",
				SelinuxRelabel: true,
			})
		}

		containerConfigs = append(containerConfigs, config)
		logger.V(2).Info("Generated container config for restore", "pod", klog.KObj(pod), "containerName", container.Name, "mountCount", len(config.Mounts))
	}

	return containerConfigs, cleanup, nil
}

// generatePodSandboxConfig generates pod sandbox config from v1.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxConfig(ctx context.Context, pod *v1.Pod, attempt uint32) (*runtimeapi.PodSandboxConfig, error) {
	runtimeHandler, err := m.lookupRuntimeHandler(pod)
	if err != nil {
		return nil, err
	}
	return m.generatePodSandboxConfigWithRuntimeHandler(ctx, pod, attempt, runtimeHandler)
}

func (m *kubeGenericRuntimeManager) generatePodSandboxConfigWithRuntimeHandler(ctx context.Context, pod *v1.Pod, attempt uint32, runtimeHandler string) (*runtimeapi.PodSandboxConfig, error) {
	// TODO: deprecating podsandbox resource requirements in favor of the pod level cgroup
	// Refer https://github.com/kubernetes/kubernetes/issues/29871
	logger := klog.FromContext(ctx)
	podUID := string(pod.UID)
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       podUID,
			Attempt:   attempt,
		},
		Labels:      newPodLabels(pod),
		Annotations: newPodAnnotations(pod),
	}

	dnsConfig, err := m.runtimeHelper.GetPodDNS(ctx, pod)
	if err != nil {
		return nil, err
	}
	podSandboxConfig.DnsConfig = dnsConfig

	if !kubecontainer.IsHostNetworkPod(pod) {
		// TODO: Add domain support in new runtime interface
		podHostname, podDomain, err := m.runtimeHelper.GeneratePodHostNameAndDomain(logger, pod)
		if err != nil {
			return nil, err
		}
		podHostname, err = util.GetNodenameForKernel(podHostname, podDomain, pod.Spec.SetHostnameAsFQDN)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.Hostname = podHostname
	}

	logDir := BuildPodLogsDirectory(m.podLogsDirectory, pod.Namespace, pod.Name, pod.UID)
	podSandboxConfig.LogDirectory = logDir

	portMappings := []*runtimeapi.PortMapping{}
	for _, c := range pod.Spec.Containers {
		containerPortMappings := kubecontainer.MakePortMappings(logger, &c)

		for idx := range containerPortMappings {
			port := containerPortMappings[idx]
			hostPort := int32(port.HostPort)
			containerPort := int32(port.ContainerPort)
			protocol := toRuntimeProtocol(logger, port.Protocol)
			portMappings = append(portMappings, &runtimeapi.PortMapping{
				HostIp:        port.HostIP,
				HostPort:      hostPort,
				ContainerPort: containerPort,
				Protocol:      protocol,
			})
		}

	}
	if len(portMappings) > 0 {
		podSandboxConfig.PortMappings = portMappings
	}

	lc, err := m.generatePodSandboxLinuxConfig(ctx, pod, runtimeHandler)
	if err != nil {
		return nil, err
	}
	podSandboxConfig.Linux = lc

	if runtime.GOOS == "windows" {
		wc, err := m.generatePodSandboxWindowsConfig(pod)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.Windows = wc
	}

	// Update config to include overhead, sandbox level resources
	if err := m.applySandboxResources(ctx, pod, podSandboxConfig); err != nil {
		return nil, err
	}
	return podSandboxConfig, nil
}

// generatePodSandboxLinuxConfig generates LinuxPodSandboxConfig from v1.Pod.
// We've to call PodSandboxLinuxConfig always irrespective of the underlying OS as securityContext is not part of
// podSandboxConfig. It is currently part of LinuxPodSandboxConfig. In future, if we have securityContext pulled out
// in podSandboxConfig we should be able to use it.
func (m *kubeGenericRuntimeManager) generatePodSandboxLinuxConfig(ctx context.Context, pod *v1.Pod, runtimeHandler string) (*runtimeapi.LinuxPodSandboxConfig, error) {
	cgroupParent := m.runtimeHelper.GetPodCgroupParent(pod)
	lc := &runtimeapi.LinuxPodSandboxConfig{
		CgroupParent: cgroupParent,
		SecurityContext: &runtimeapi.LinuxSandboxSecurityContext{
			Privileged: kubecontainer.HasPrivilegedContainer(pod),

			// Forcing sandbox to run as `runtime/default` allow users to
			// use least privileged seccomp profiles at pod level. Issue #84623
			Seccomp: &runtimeapi.SecurityProfile{
				ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
			},
		},
	}

	sysctls := make(map[string]string)
	if utilfeature.DefaultFeatureGate.Enabled(features.DefaultPodSysctls) {
		applyPodSysctls(sysctls, m.defaultPodSysctls, pod)
	}
	if pod.Spec.SecurityContext != nil {
		for _, c := range pod.Spec.SecurityContext.Sysctls {
			sysctls[c.Name] = c.Value
		}
	}

	lc.Sysctls = sysctls

	if pod.Spec.SecurityContext != nil {
		sc := pod.Spec.SecurityContext
		if sc.RunAsUser != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.RunAsUser = &runtimeapi.Int64Value{Value: int64(*sc.RunAsUser)}
		}
		if sc.RunAsGroup != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.RunAsGroup = &runtimeapi.Int64Value{Value: int64(*sc.RunAsGroup)}
		}
		namespaceOptions, err := runtimeutil.NamespacesForPodWithRuntimeHandler(ctx, pod, m.runtimeHelper, runtimeHandler)
		if err != nil {
			return nil, err
		}
		lc.SecurityContext.NamespaceOptions = namespaceOptions

		if sc.FSGroup != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, int64(*sc.FSGroup))
		}
		if groups := m.runtimeHelper.GetExtraSupplementalGroupsForPod(pod); len(groups) > 0 {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, groups...)
		}
		if sc.SupplementalGroups != nil {
			for _, sg := range sc.SupplementalGroups {
				lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, int64(sg))
			}
		}
		if sc.SupplementalGroupsPolicy != nil {
			policyValue, ok := runtimeapi.SupplementalGroupsPolicy_value[string(*sc.SupplementalGroupsPolicy)]
			if !ok {
				return nil, fmt.Errorf("unsupported supplementalGroupsPolicy: %s", string(*sc.SupplementalGroupsPolicy))
			}
			lc.SecurityContext.SupplementalGroupsPolicy = runtimeapi.SupplementalGroupsPolicy(policyValue)
		}

		if sc.SELinuxOptions != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.SelinuxOptions = &runtimeapi.SELinuxOption{
				User:  sc.SELinuxOptions.User,
				Role:  sc.SELinuxOptions.Role,
				Type:  sc.SELinuxOptions.Type,
				Level: sc.SELinuxOptions.Level,
			}
		}
	}

	return lc, nil
}

// generatePodSandboxWindowsConfig generates WindowsPodSandboxConfig from v1.Pod.
// On Windows this will get called in addition to LinuxPodSandboxConfig because not all relevant fields have been added to
// WindowsPodSandboxConfig at this time.
func (m *kubeGenericRuntimeManager) generatePodSandboxWindowsConfig(pod *v1.Pod) (*runtimeapi.WindowsPodSandboxConfig, error) {
	wc := &runtimeapi.WindowsPodSandboxConfig{
		SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
	}

	// If all of the containers in a pod are HostProcess containers, set the pod's HostProcess field
	// explicitly because the container runtime requires this information at sandbox creation time.
	if kubecontainer.HasWindowsHostProcessContainer(pod) {
		// At present Windows all containers in a Windows pod must be HostProcess containers
		// and HostNetwork is required to be set.
		if !kubecontainer.AllContainersAreWindowsHostProcess(pod) {
			return nil, fmt.Errorf("pod must not contain both HostProcess and non-HostProcess containers")
		}

		if !kubecontainer.IsHostNetworkPod(pod) {
			return nil, fmt.Errorf("hostNetwork is required if Pod contains HostProcess containers")
		}

		wc.SecurityContext.HostProcess = true
	}

	sc := pod.Spec.SecurityContext
	if sc == nil || sc.WindowsOptions == nil {
		return wc, nil
	}

	wo := sc.WindowsOptions
	if wo.GMSACredentialSpec != nil {
		wc.SecurityContext.CredentialSpec = *wo.GMSACredentialSpec
	}

	if wo.RunAsUserName != nil {
		wc.SecurityContext.RunAsUsername = *wo.RunAsUserName
	}

	if kubecontainer.HasWindowsHostProcessContainer(pod) {

		if wo.HostProcess != nil && !*wo.HostProcess {
			return nil, fmt.Errorf("pod must not contain any HostProcess containers if Pod's WindowsOptions.HostProcess is set to false")
		}
	}

	return wc, nil
}

// determinePodSandboxIP determines the IP addresses of the given pod sandbox.
func (m *kubeGenericRuntimeManager) determinePodSandboxIPs(ctx context.Context, podNamespace, podName string, podSandbox *runtimeapi.PodSandboxStatus) []string {
	logger := klog.FromContext(ctx)
	podIPs := make([]string, 0)
	if podSandbox.Network == nil {
		logger.Info("Pod Sandbox status doesn't have network information, cannot report IPs", "pod", klog.KRef(podNamespace, podName))
		return podIPs
	}

	// ip could be an empty string if runtime is not responsible for the
	// IP (e.g., host networking).

	// pick primary IP
	if len(podSandbox.Network.Ip) != 0 {
		if netutils.ParseIPSloppy(podSandbox.Network.Ip) == nil {
			logger.Info("Pod Sandbox reported an unparseable primary IP", "pod", klog.KRef(podNamespace, podName), "IP", podSandbox.Network.Ip)
			return nil
		}
		podIPs = append(podIPs, podSandbox.Network.Ip)
	}

	// pick additional ips, if cri reported them
	for _, podIP := range podSandbox.Network.AdditionalIps {
		if nil == netutils.ParseIPSloppy(podIP.Ip) {
			logger.Info("Pod Sandbox reported an unparseable additional IP", "pod", klog.KRef(podNamespace, podName), "IP", podIP.Ip)
			return nil
		}
		podIPs = append(podIPs, podIP.Ip)
	}

	return podIPs
}

// getPodSandboxIDByPodUID gets the sandbox id by podUID and returns ([]sandboxID, error).
// Param state could be nil in order to get all sandboxes belonging to same pod.
func (m *kubeGenericRuntimeManager) getSandboxIDByPodUID(ctx context.Context, podUID kubetypes.UID) ([]string, error) {
	logger := klog.FromContext(ctx)
	sandboxes, err := m.getSandboxes(ctx, listOptions{podUID: podUID})
	if err != nil {
		logger.Error(err, "Failed to list sandboxes for pod", "podUID", podUID)
		return nil, err
	}

	if len(sandboxes) == 0 {
		return nil, nil
	}

	// Sort with newest first.
	sandboxIDs := make([]string, len(sandboxes))
	sort.Sort(podSandboxByCreatedThenID(sandboxes))
	for i, s := range sandboxes {
		sandboxIDs[i] = s.Id
	}

	return sandboxIDs, nil
}

// GetPortForward gets the endpoint the runtime will serve the port-forward request from.
func (m *kubeGenericRuntimeManager) GetPortForward(ctx context.Context, podName, podNamespace string, podUID kubetypes.UID, ports []int32) (*url.URL, error) {
	sandboxIDs, err := m.getSandboxIDByPodUID(ctx, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to find sandboxID for pod %s: %v", format.PodDesc(podName, podNamespace, podUID), err)
	}
	if len(sandboxIDs) == 0 {
		return nil, fmt.Errorf("failed to find sandboxID for pod %s", format.PodDesc(podName, podNamespace, podUID))
	}
	req := &runtimeapi.PortForwardRequest{
		PodSandboxId: sandboxIDs[0],
		Port:         ports,
	}
	resp, err := m.runtimeService.PortForward(ctx, req)
	if err != nil {
		return nil, err
	}
	return url.Parse(resp.Url)
}

var validSysctlMap = map[string]bool{
	"kernel.msgmax":          true,
	"kernel.msgmnb":          true,
	"kernel.msgmni":          true,
	"kernel.sem":             true,
	"kernel.shmall":          true,
	"kernel.shmmax":          true,
	"kernel.shmmni":          true,
	"kernel.shm_rmid_forced": true,
}

// From https://github.com/opencontainers/runc/blob/master/libcontainer/configs/validate/validator.go
func applyPodSysctls(sysctls, podSysctls map[string]string, pod *v1.Pod) {
	for k, v := range podSysctls {
		normalizedKey := utilsysctl.NormalizeName(k)
		if validSysctlMap[normalizedKey] || strings.HasPrefix(normalizedKey, "fs.mqueue.") {
			if !pod.Spec.HostIPC {
				// Set the IPC parameters unless the pod runs in the host IPC
				// namespace.
				sysctls[normalizedKey] = v
			}
			continue
		}
		if strings.HasPrefix(normalizedKey, "net.") {
			if !pod.Spec.HostNetwork {
				// Set the networking parameters unless the pod runs in the
				// host network namespace.
				sysctls[normalizedKey] = v
			}
			continue
		}
		if strings.HasPrefix(normalizedKey, "user.") {
			// The default for HostUsers is true, so a nil HostUsers means the
			// pod runs in the host user namespace.
			if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
				// Set the user parameters unless the pod runs in the host user
				// namespace.
				sysctls[normalizedKey] = v
			}
			continue
		}
		if !pod.Spec.HostNetwork {
			if normalizedKey == "kernel.domainname" {
				// This is namespaced and there's no explicit OCI field for it.
				sysctls[normalizedKey] = v
			}
			continue
		}
	}
}
