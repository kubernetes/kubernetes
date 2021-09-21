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
	"io"
	"io/ioutil"
	"math/rand"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	goruntime "runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	grpcstatus "google.golang.org/grpc/status"

	"github.com/armon/circbuf"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util/selinux"
	"k8s.io/kubernetes/pkg/util/tail"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

var (
	// ErrCreateContainerConfig - failed to create container config
	ErrCreateContainerConfig = errors.New("CreateContainerConfigError")
	// ErrPreCreateHook - failed to execute PreCreateHook
	ErrPreCreateHook = errors.New("PreCreateHookError")
	// ErrCreateContainer - failed to create container
	ErrCreateContainer = errors.New("CreateContainerError")
	// ErrPreStartHook - failed to execute PreStartHook
	ErrPreStartHook = errors.New("PreStartHookError")
	// ErrPostStartHook - failed to execute PostStartHook
	ErrPostStartHook = errors.New("PostStartHookError")
)

// recordContainerEvent should be used by the runtime manager for all container related events.
// it has sanity checks to ensure that we do not write events that can abuse our masters.
// in particular, it ensures that a containerID never appears in an event message as that
// is prone to causing a lot of distinct events that do not count well.
// it replaces any reference to a containerID with the containerName which is stable, and is what users know.
func (m *kubeGenericRuntimeManager) recordContainerEvent(pod *v1.Pod, container *v1.Container, containerID, eventType, reason, message string, args ...interface{}) {
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		klog.ErrorS(err, "Can't make a container ref", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name)
		return
	}
	eventMessage := message
	if len(args) > 0 {
		eventMessage = fmt.Sprintf(message, args...)
	}
	// this is a hack, but often the error from the runtime includes the containerID
	// which kills our ability to deduplicate events.  this protection makes a huge
	// difference in the number of unique events
	if containerID != "" {
		eventMessage = strings.Replace(eventMessage, containerID, container.Name, -1)
	}
	m.recorder.Event(ref, eventType, reason, eventMessage)
}

// startSpec wraps the spec required to start a container, either a regular/init container
// or an ephemeral container. Ephemeral containers contain all the fields of regular/init
// containers, plus some additional fields. In both cases startSpec.container will be set.
type startSpec struct {
	container          *v1.Container
	ephemeralContainer *v1.EphemeralContainer
}

func containerStartSpec(c *v1.Container) *startSpec {
	return &startSpec{container: c}
}

func ephemeralContainerStartSpec(ec *v1.EphemeralContainer) *startSpec {
	return &startSpec{
		container:          (*v1.Container)(&ec.EphemeralContainerCommon),
		ephemeralContainer: ec,
	}
}

// getTargetID returns the kubecontainer.ContainerID for ephemeral container namespace
// targeting. The target is stored as EphemeralContainer.TargetContainerName, which must be
// resolved to a ContainerID using podStatus. The target container must already exist, which
// usually isn't a problem since ephemeral containers aren't allowed at pod creation time.
// This always returns nil when the EphemeralContainers feature is disabled.
func (s *startSpec) getTargetID(podStatus *kubecontainer.PodStatus) (*kubecontainer.ContainerID, error) {
	if s.ephemeralContainer == nil || s.ephemeralContainer.TargetContainerName == "" || !utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) {
		return nil, nil
	}

	targetStatus := podStatus.FindContainerStatusByName(s.ephemeralContainer.TargetContainerName)
	if targetStatus == nil {
		return nil, fmt.Errorf("unable to find target container %v", s.ephemeralContainer.TargetContainerName)
	}

	return &targetStatus.ID, nil
}

func calcRestartCountByLogDir(path string) (int, error) {
	// if the path doesn't exist then it's not an error
	if _, err := os.Stat(path); err != nil {
		return 0, nil
	}
	restartCount := int(0)
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return 0, err
	}
	if len(files) == 0 {
		return 0, err
	}
	restartCountLogFileRegex := regexp.MustCompile(`(\d+).log(\..*)?`)
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		matches := restartCountLogFileRegex.FindStringSubmatch(file.Name())
		if len(matches) == 0 {
			continue
		}
		count, err := strconv.Atoi(matches[1])
		if err != nil {
			return restartCount, err
		}
		count++
		if count > restartCount {
			restartCount = count
		}
	}
	return restartCount, nil
}

// startContainer starts a container and returns a message indicates why it is failed on error.
// It starts the container through the following steps:
// * pull the image
// * create the container
// * start the container
// * run the post start lifecycle hooks (if applicable)
func (m *kubeGenericRuntimeManager) startContainer(podSandboxID string, podSandboxConfig *runtimeapi.PodSandboxConfig, spec *startSpec, pod *v1.Pod, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret, podIP string, podIPs []string) (string, error) {
	container := spec.container

	// Step 1: pull the image.
	imageRef, msg, err := m.imagePuller.EnsureImageExists(pod, container, pullSecrets, podSandboxConfig)
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, "", v1.EventTypeWarning, events.FailedToCreateContainer, "Error: %v", s.Message())
		return msg, err
	}

	// Step 2: create the container.
	// For a new container, the RestartCount should be 0
	restartCount := 0
	containerStatus := podStatus.FindContainerStatusByName(container.Name)
	if containerStatus != nil {
		restartCount = containerStatus.RestartCount + 1
	} else {
		// The container runtime keeps state on container statuses and
		// what the container restart count is. When nodes are rebooted
		// some container runtimes clear their state which causes the
		// restartCount to be reset to 0. This causes the logfile to
		// start at 0.log, which either overwrites or appends to the
		// already existing log.
		//
		// We are checking to see if the log directory exists, and find
		// the latest restartCount by checking the log name -
		// {restartCount}.log - and adding 1 to it.
		logDir := BuildContainerLogsDirectory(pod.Namespace, pod.Name, pod.UID, container.Name)
		restartCount, err = calcRestartCountByLogDir(logDir)
		if err != nil {
			klog.InfoS("Log directory exists but could not calculate restartCount", "logDir", logDir, "err", err)
		}
	}

	target, err := spec.getTargetID(podStatus)
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, "", v1.EventTypeWarning, events.FailedToCreateContainer, "Error: %v", s.Message())
		return s.Message(), ErrCreateContainerConfig
	}

	containerConfig, cleanupAction, err := m.generateContainerConfig(container, pod, restartCount, podIP, imageRef, podIPs, target)
	if cleanupAction != nil {
		defer cleanupAction()
	}
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, "", v1.EventTypeWarning, events.FailedToCreateContainer, "Error: %v", s.Message())
		return s.Message(), ErrCreateContainerConfig
	}

	err = m.internalLifecycle.PreCreateContainer(pod, container, containerConfig)
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, "", v1.EventTypeWarning, events.FailedToCreateContainer, "Internal PreCreateContainer hook failed: %v", s.Message())
		return s.Message(), ErrPreCreateHook
	}

	containerID, err := m.runtimeService.CreateContainer(podSandboxID, containerConfig, podSandboxConfig)
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, containerID, v1.EventTypeWarning, events.FailedToCreateContainer, "Error: %v", s.Message())
		return s.Message(), ErrCreateContainer
	}
	err = m.internalLifecycle.PreStartContainer(pod, container, containerID)
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, containerID, v1.EventTypeWarning, events.FailedToStartContainer, "Internal PreStartContainer hook failed: %v", s.Message())
		return s.Message(), ErrPreStartHook
	}
	m.recordContainerEvent(pod, container, containerID, v1.EventTypeNormal, events.CreatedContainer, fmt.Sprintf("Created container %s", container.Name))

	// Step 3: start the container.
	err = m.runtimeService.StartContainer(containerID)
	if err != nil {
		s, _ := grpcstatus.FromError(err)
		m.recordContainerEvent(pod, container, containerID, v1.EventTypeWarning, events.FailedToStartContainer, "Error: %v", s.Message())
		return s.Message(), kubecontainer.ErrRunContainer
	}
	m.recordContainerEvent(pod, container, containerID, v1.EventTypeNormal, events.StartedContainer, fmt.Sprintf("Started container %s", container.Name))

	// Symlink container logs to the legacy container log location for cluster logging
	// support.
	// TODO(random-liu): Remove this after cluster logging supports CRI container log path.
	containerMeta := containerConfig.GetMetadata()
	sandboxMeta := podSandboxConfig.GetMetadata()
	legacySymlink := legacyLogSymlink(containerID, containerMeta.Name, sandboxMeta.Name,
		sandboxMeta.Namespace)
	containerLog := filepath.Join(podSandboxConfig.LogDirectory, containerConfig.LogPath)
	// only create legacy symlink if containerLog path exists (or the error is not IsNotExist).
	// Because if containerLog path does not exist, only dangling legacySymlink is created.
	// This dangling legacySymlink is later removed by container gc, so it does not make sense
	// to create it in the first place. it happens when journald logging driver is used with docker.
	if _, err := m.osInterface.Stat(containerLog); !os.IsNotExist(err) {
		if err := m.osInterface.Symlink(containerLog, legacySymlink); err != nil {
			klog.ErrorS(err, "Failed to create legacy symbolic link", "path", legacySymlink,
				"containerID", containerID, "containerLogPath", containerLog)
		}
	}

	// Step 4: execute the post start hook.
	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		kubeContainerID := kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   containerID,
		}
		msg, handlerErr := m.runner.Run(kubeContainerID, pod, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			klog.ErrorS(handlerErr, "Failed to execute PostStartHook", "pod", klog.KObj(pod),
				"podUID", pod.UID, "containerName", container.Name, "containerID", kubeContainerID.String())
			m.recordContainerEvent(pod, container, kubeContainerID.ID, v1.EventTypeWarning, events.FailedPostStartHook, msg)
			if err := m.killContainer(pod, kubeContainerID, container.Name, "FailedPostStartHook", reasonFailedPostStartHook, nil); err != nil {
				klog.ErrorS(err, "Failed to kill container", "pod", klog.KObj(pod),
					"podUID", pod.UID, "containerName", container.Name, "containerID", kubeContainerID.String())
			}
			return msg, ErrPostStartHook
		}
	}

	return "", nil
}

// generateContainerConfig generates container config for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) generateContainerConfig(container *v1.Container, pod *v1.Pod, restartCount int, podIP, imageRef string, podIPs []string, nsTarget *kubecontainer.ContainerID) (*runtimeapi.ContainerConfig, func(), error) {
	opts, cleanupAction, err := m.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP, podIPs)
	if err != nil {
		return nil, nil, err
	}

	uid, username, err := m.getImageUser(container.Image)
	if err != nil {
		return nil, cleanupAction, err
	}

	// Verify RunAsNonRoot. Non-root verification only supports numeric user.
	if err := verifyRunAsNonRoot(pod, container, uid, username); err != nil {
		return nil, cleanupAction, err
	}

	command, args := kubecontainer.ExpandContainerCommandAndArgs(container, opts.Envs)
	logDir := BuildContainerLogsDirectory(pod.Namespace, pod.Name, pod.UID, container.Name)
	err = m.osInterface.MkdirAll(logDir, 0755)
	if err != nil {
		return nil, cleanupAction, fmt.Errorf("create container log directory for container %s failed: %v", container.Name, err)
	}
	containerLogsPath := buildContainerLogsPath(container.Name, restartCount)
	restartCountUint32 := uint32(restartCount)
	config := &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    container.Name,
			Attempt: restartCountUint32,
		},
		Image:       &runtimeapi.ImageSpec{Image: imageRef},
		Command:     command,
		Args:        args,
		WorkingDir:  container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(container, pod, restartCount, opts),
		Devices:     makeDevices(opts),
		Mounts:      m.makeMounts(opts, container),
		LogPath:     containerLogsPath,
		Stdin:       container.Stdin,
		StdinOnce:   container.StdinOnce,
		Tty:         container.TTY,
	}

	// set platform specific configurations.
	if err := m.applyPlatformSpecificContainerConfig(config, container, pod, uid, username, nsTarget); err != nil {
		return nil, cleanupAction, err
	}

	// set environment variables
	envs := make([]*runtimeapi.KeyValue, len(opts.Envs))
	for idx := range opts.Envs {
		e := opts.Envs[idx]
		envs[idx] = &runtimeapi.KeyValue{
			Key:   e.Name,
			Value: e.Value,
		}
	}
	config.Envs = envs

	return config, cleanupAction, nil
}

// makeDevices generates container devices for kubelet runtime v1.
func makeDevices(opts *kubecontainer.RunContainerOptions) []*runtimeapi.Device {
	devices := make([]*runtimeapi.Device, len(opts.Devices))

	for idx := range opts.Devices {
		device := opts.Devices[idx]
		devices[idx] = &runtimeapi.Device{
			HostPath:      device.PathOnHost,
			ContainerPath: device.PathInContainer,
			Permissions:   device.Permissions,
		}
	}

	return devices
}

// makeMounts generates container volume mounts for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) makeMounts(opts *kubecontainer.RunContainerOptions, container *v1.Container) []*runtimeapi.Mount {
	volumeMounts := []*runtimeapi.Mount{}

	for idx := range opts.Mounts {
		v := opts.Mounts[idx]
		selinuxRelabel := v.SELinuxRelabel && selinux.SELinuxEnabled()
		mount := &runtimeapi.Mount{
			HostPath:       v.HostPath,
			ContainerPath:  v.ContainerPath,
			Readonly:       v.ReadOnly,
			SelinuxRelabel: selinuxRelabel,
			Propagation:    v.Propagation,
		}

		volumeMounts = append(volumeMounts, mount)
	}

	// The reason we create and mount the log file in here (not in kubelet) is because
	// the file's location depends on the ID of the container, and we need to create and
	// mount the file before actually starting the container.
	// we can only mount individual files (e.g.: /etc/hosts, termination-log files) on Windows only if we're using Containerd.
	supportsSingleFileMapping := m.SupportsSingleFileMapping()
	if opts.PodContainerDir != "" && len(container.TerminationMessagePath) != 0 && supportsSingleFileMapping {
		// Because the PodContainerDir contains pod uid and container name which is unique enough,
		// here we just add a random id to make the path unique for different instances
		// of the same container.
		cid := makeUID()
		containerLogPath := filepath.Join(opts.PodContainerDir, cid)
		fs, err := m.osInterface.Create(containerLogPath)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("error on creating termination-log file %q: %v", containerLogPath, err))
		} else {
			fs.Close()

			// Chmod is needed because ioutil.WriteFile() ends up calling
			// open(2) to create the file, so the final mode used is "mode &
			// ~umask". But we want to make sure the specified mode is used
			// in the file no matter what the umask is.
			if err := m.osInterface.Chmod(containerLogPath, 0666); err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to set termination-log file permissions %q: %v", containerLogPath, err))
			}

			// Volume Mounts fail on Windows if it is not of the form C:/
			containerLogPath = volumeutil.MakeAbsolutePath(goruntime.GOOS, containerLogPath)
			terminationMessagePath := volumeutil.MakeAbsolutePath(goruntime.GOOS, container.TerminationMessagePath)
			selinuxRelabel := selinux.SELinuxEnabled()
			volumeMounts = append(volumeMounts, &runtimeapi.Mount{
				HostPath:       containerLogPath,
				ContainerPath:  terminationMessagePath,
				SelinuxRelabel: selinuxRelabel,
			})
		}
	}

	return volumeMounts
}

// getKubeletContainers lists containers managed by kubelet.
// The boolean parameter specifies whether returns all containers including
// those already exited and dead containers (used for garbage collection).
func (m *kubeGenericRuntimeManager) getKubeletContainers(allContainers bool) ([]*runtimeapi.Container, error) {
	filter := &runtimeapi.ContainerFilter{}
	if !allContainers {
		filter.State = &runtimeapi.ContainerStateValue{
			State: runtimeapi.ContainerState_CONTAINER_RUNNING,
		}
	}

	containers, err := m.runtimeService.ListContainers(filter)
	if err != nil {
		klog.ErrorS(err, "ListContainers failed")
		return nil, err
	}

	return containers, nil
}

// makeUID returns a randomly generated string.
func makeUID() string {
	return fmt.Sprintf("%08x", rand.Uint32())
}

// getTerminationMessage looks on the filesystem for the provided termination message path, returning a limited
// amount of those bytes, or returns true if the logs should be checked.
func getTerminationMessage(status *runtimeapi.ContainerStatus, terminationMessagePath string, fallbackToLogs bool) (string, bool) {
	if len(terminationMessagePath) == 0 {
		return "", fallbackToLogs
	}
	// Volume Mounts fail on Windows if it is not of the form C:/
	terminationMessagePath = volumeutil.MakeAbsolutePath(goruntime.GOOS, terminationMessagePath)
	for _, mount := range status.Mounts {
		if mount.ContainerPath != terminationMessagePath {
			continue
		}
		path := mount.HostPath
		data, _, err := tail.ReadAtMost(path, kubecontainer.MaxContainerTerminationMessageLength)
		if err != nil {
			if os.IsNotExist(err) {
				return "", fallbackToLogs
			}
			return fmt.Sprintf("Error on reading termination log %s: %v", path, err), false
		}
		return string(data), (fallbackToLogs && len(data) == 0)
	}
	return "", fallbackToLogs
}

// readLastStringFromContainerLogs attempts to read up to the max log length from the end of the CRI log represented
// by path. It reads up to max log lines.
func (m *kubeGenericRuntimeManager) readLastStringFromContainerLogs(path string) string {
	value := int64(kubecontainer.MaxContainerTerminationMessageLogLines)
	buf, _ := circbuf.NewBuffer(kubecontainer.MaxContainerTerminationMessageLogLength)
	if err := m.ReadLogs(context.Background(), path, "", &v1.PodLogOptions{TailLines: &value}, buf, buf); err != nil {
		return fmt.Sprintf("Error on reading termination message from logs: %v", err)
	}
	return buf.String()
}

// getPodContainerStatuses gets all containers' statuses for the pod.
func (m *kubeGenericRuntimeManager) getPodContainerStatuses(uid kubetypes.UID, name, namespace string) ([]*kubecontainer.Status, error) {
	// Select all containers of the given pod.
	containers, err := m.runtimeService.ListContainers(&runtimeapi.ContainerFilter{
		LabelSelector: map[string]string{types.KubernetesPodUIDLabel: string(uid)},
	})
	if err != nil {
		klog.ErrorS(err, "ListContainers error")
		return nil, err
	}

	statuses := make([]*kubecontainer.Status, len(containers))
	// TODO: optimization: set maximum number of containers per container name to examine.
	for i, c := range containers {
		status, err := m.runtimeService.ContainerStatus(c.Id)
		if err != nil {
			// Merely log this here; GetPodStatus will actually report the error out.
			klog.V(4).InfoS("ContainerStatus return error", "containerID", c.Id, "err", err)
			return nil, err
		}
		cStatus := toKubeContainerStatus(status, m.runtimeName)
		if status.State == runtimeapi.ContainerState_CONTAINER_EXITED {
			// Populate the termination message if needed.
			annotatedInfo := getContainerInfoFromAnnotations(status.Annotations)
			// If a container cannot even be started, it certainly does not have logs, so no need to fallbackToLogs.
			fallbackToLogs := annotatedInfo.TerminationMessagePolicy == v1.TerminationMessageFallbackToLogsOnError &&
				cStatus.ExitCode != 0 && cStatus.Reason != "ContainerCannotRun"
			tMessage, checkLogs := getTerminationMessage(status, annotatedInfo.TerminationMessagePath, fallbackToLogs)
			if checkLogs {
				// if dockerLegacyService is populated, we're supposed to use it to fetch logs
				if m.legacyLogProvider != nil {
					tMessage, err = m.legacyLogProvider.GetContainerLogTail(uid, name, namespace, kubecontainer.ContainerID{Type: m.runtimeName, ID: c.Id})
					if err != nil {
						tMessage = fmt.Sprintf("Error reading termination message from logs: %v", err)
					}
				} else {
					tMessage = m.readLastStringFromContainerLogs(status.GetLogPath())
				}
			}
			// Enrich the termination message written by the application is not empty
			if len(tMessage) != 0 {
				if len(cStatus.Message) != 0 {
					cStatus.Message += ": "
				}
				cStatus.Message += tMessage
			}
		}
		statuses[i] = cStatus
	}

	sort.Sort(containerStatusByCreated(statuses))
	return statuses, nil
}

func toKubeContainerStatus(status *runtimeapi.ContainerStatus, runtimeName string) *kubecontainer.Status {
	annotatedInfo := getContainerInfoFromAnnotations(status.Annotations)
	labeledInfo := getContainerInfoFromLabels(status.Labels)
	cStatus := &kubecontainer.Status{
		ID: kubecontainer.ContainerID{
			Type: runtimeName,
			ID:   status.Id,
		},
		Name:         labeledInfo.ContainerName,
		Image:        status.Image.Image,
		ImageID:      status.ImageRef,
		Hash:         annotatedInfo.Hash,
		RestartCount: annotatedInfo.RestartCount,
		State:        toKubeContainerState(status.State),
		CreatedAt:    time.Unix(0, status.CreatedAt),
	}

	if status.State != runtimeapi.ContainerState_CONTAINER_CREATED {
		// If container is not in the created state, we have tried and
		// started the container. Set the StartedAt time.
		cStatus.StartedAt = time.Unix(0, status.StartedAt)
	}
	if status.State == runtimeapi.ContainerState_CONTAINER_EXITED {
		cStatus.Reason = status.Reason
		cStatus.Message = status.Message
		cStatus.ExitCode = int(status.ExitCode)
		cStatus.FinishedAt = time.Unix(0, status.FinishedAt)
	}
	return cStatus
}

// executePreStopHook runs the pre-stop lifecycle hooks if applicable and returns the duration it takes.
func (m *kubeGenericRuntimeManager) executePreStopHook(pod *v1.Pod, containerID kubecontainer.ContainerID, containerSpec *v1.Container, gracePeriod int64) int64 {
	klog.V(3).InfoS("Running preStop hook", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerSpec.Name, "containerID", containerID.String())

	start := metav1.Now()
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer utilruntime.HandleCrash()
		if msg, err := m.runner.Run(containerID, pod, containerSpec, containerSpec.Lifecycle.PreStop); err != nil {
			klog.ErrorS(err, "PreStop hook failed", "pod", klog.KObj(pod), "podUID", pod.UID,
				"containerName", containerSpec.Name, "containerID", containerID.String())
			m.recordContainerEvent(pod, containerSpec, containerID.ID, v1.EventTypeWarning, events.FailedPreStopHook, msg)
		}
	}()

	select {
	case <-time.After(time.Duration(gracePeriod) * time.Second):
		klog.V(2).InfoS("PreStop hook not completed in grace period", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerSpec.Name, "containerID", containerID.String(), "gracePeriod", gracePeriod)
	case <-done:
		klog.V(3).InfoS("PreStop hook completed", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerSpec.Name, "containerID", containerID.String())
	}

	return int64(metav1.Now().Sub(start.Time).Seconds())
}

// restoreSpecsFromContainerLabels restores all information needed for killing a container. In some
// case we may not have pod and container spec when killing a container, e.g. pod is deleted during
// kubelet restart.
// To solve this problem, we've already written necessary information into container labels. Here we
// just need to retrieve them from container labels and restore the specs.
// TODO(random-liu): Add a node e2e test to test this behaviour.
// TODO(random-liu): Change the lifecycle handler to just accept information needed, so that we can
// just pass the needed function not create the fake object.
func (m *kubeGenericRuntimeManager) restoreSpecsFromContainerLabels(containerID kubecontainer.ContainerID) (*v1.Pod, *v1.Container, error) {
	var pod *v1.Pod
	var container *v1.Container
	s, err := m.runtimeService.ContainerStatus(containerID.ID)
	if err != nil {
		return nil, nil, err
	}

	l := getContainerInfoFromLabels(s.Labels)
	a := getContainerInfoFromAnnotations(s.Annotations)
	// Notice that the followings are not full spec. The container killing code should not use
	// un-restored fields.
	pod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:                        l.PodUID,
			Name:                       l.PodName,
			Namespace:                  l.PodNamespace,
			DeletionGracePeriodSeconds: a.PodDeletionGracePeriod,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: a.PodTerminationGracePeriod,
		},
	}
	container = &v1.Container{
		Name:                   l.ContainerName,
		Ports:                  a.ContainerPorts,
		TerminationMessagePath: a.TerminationMessagePath,
	}
	if a.PreStopHandler != nil {
		container.Lifecycle = &v1.Lifecycle{
			PreStop: a.PreStopHandler,
		}
	}
	return pod, container, nil
}

// killContainer kills a container through the following steps:
// * Run the pre-stop lifecycle hooks (if applicable).
// * Stop the container.
func (m *kubeGenericRuntimeManager) killContainer(pod *v1.Pod, containerID kubecontainer.ContainerID, containerName string, message string, reason containerKillReason, gracePeriodOverride *int64) error {
	var containerSpec *v1.Container
	if pod != nil {
		if containerSpec = kubecontainer.GetContainerSpec(pod, containerName); containerSpec == nil {
			return fmt.Errorf("failed to get containerSpec %q (id=%q) in pod %q when killing container for reason %q",
				containerName, containerID.String(), format.Pod(pod), message)
		}
	} else {
		// Restore necessary information if one of the specs is nil.
		restoredPod, restoredContainer, err := m.restoreSpecsFromContainerLabels(containerID)
		if err != nil {
			return err
		}
		pod, containerSpec = restoredPod, restoredContainer
	}

	// From this point, pod and container must be non-nil.
	gracePeriod := int64(minimumGracePeriodInSeconds)
	switch {
	case pod.DeletionGracePeriodSeconds != nil:
		gracePeriod = *pod.DeletionGracePeriodSeconds
	case pod.Spec.TerminationGracePeriodSeconds != nil:
		gracePeriod = *pod.Spec.TerminationGracePeriodSeconds

		switch reason {
		case reasonStartupProbe:
			if containerSpec.StartupProbe != nil && containerSpec.StartupProbe.TerminationGracePeriodSeconds != nil {
				gracePeriod = *containerSpec.StartupProbe.TerminationGracePeriodSeconds
			}
		case reasonLivenessProbe:
			if containerSpec.LivenessProbe != nil && containerSpec.LivenessProbe.TerminationGracePeriodSeconds != nil {
				gracePeriod = *containerSpec.LivenessProbe.TerminationGracePeriodSeconds
			}
		}
	}

	if len(message) == 0 {
		message = fmt.Sprintf("Stopping container %s", containerSpec.Name)
	}
	m.recordContainerEvent(pod, containerSpec, containerID.ID, v1.EventTypeNormal, events.KillingContainer, message)

	// Run internal pre-stop lifecycle hook
	if err := m.internalLifecycle.PreStopContainer(containerID.ID); err != nil {
		return err
	}

	// Run the pre-stop lifecycle hooks if applicable and if there is enough time to run it
	if containerSpec.Lifecycle != nil && containerSpec.Lifecycle.PreStop != nil && gracePeriod > 0 {
		gracePeriod = gracePeriod - m.executePreStopHook(pod, containerID, containerSpec, gracePeriod)
	}
	// always give containers a minimal shutdown window to avoid unnecessary SIGKILLs
	if gracePeriod < minimumGracePeriodInSeconds {
		gracePeriod = minimumGracePeriodInSeconds
	}
	if gracePeriodOverride != nil {
		gracePeriod = *gracePeriodOverride
		klog.V(3).InfoS("Killing container with a grace period override", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String(), "gracePeriod", gracePeriod)
	}

	klog.V(2).InfoS("Killing container with a grace period", "pod", klog.KObj(pod), "podUID", pod.UID,
		"containerName", containerName, "containerID", containerID.String(), "gracePeriod", gracePeriod)

	err := m.runtimeService.StopContainer(containerID.ID, gracePeriod)
	if err != nil {
		klog.ErrorS(err, "Container termination failed with gracePeriod", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String(), "gracePeriod", gracePeriod)
	} else {
		klog.V(3).InfoS("Container exited normally", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String())
	}

	return err
}

// killContainersWithSyncResult kills all pod's containers with sync results.
func (m *kubeGenericRuntimeManager) killContainersWithSyncResult(pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) (syncResults []*kubecontainer.SyncResult) {
	containerResults := make(chan *kubecontainer.SyncResult, len(runningPod.Containers))
	wg := sync.WaitGroup{}

	wg.Add(len(runningPod.Containers))
	for _, container := range runningPod.Containers {
		go func(container *kubecontainer.Container) {
			defer utilruntime.HandleCrash()
			defer wg.Done()

			killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, container.Name)
			if err := m.killContainer(pod, container.ID, container.Name, "", reasonUnknown, gracePeriodOverride); err != nil {
				killContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
				// Use runningPod for logging as the pod passed in could be *nil*.
				klog.ErrorS(err, "Kill container failed", "pod", klog.KRef(runningPod.Namespace, runningPod.Name), "podUID", runningPod.ID,
					"containerName", container.Name, "containerID", container.ID)
			}
			containerResults <- killContainerResult
		}(container)
	}
	wg.Wait()
	close(containerResults)

	for containerResult := range containerResults {
		syncResults = append(syncResults, containerResult)
	}
	return
}

// pruneInitContainersBeforeStart ensures that before we begin creating init
// containers, we have reduced the number of outstanding init containers still
// present. This reduces load on the container garbage collector by only
// preserving the most recent terminated init container.
func (m *kubeGenericRuntimeManager) pruneInitContainersBeforeStart(pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	// only the last execution of each init container should be preserved, and only preserve it if it is in the
	// list of init containers to keep.
	initContainerNames := sets.NewString()
	for _, container := range pod.Spec.InitContainers {
		initContainerNames.Insert(container.Name)
	}
	for name := range initContainerNames {
		count := 0
		for _, status := range podStatus.ContainerStatuses {
			if status.Name != name ||
				(status.State != kubecontainer.ContainerStateExited &&
					status.State != kubecontainer.ContainerStateUnknown) {
				continue
			}
			// Remove init containers in unknown state. It should have
			// been stopped before pruneInitContainersBeforeStart is
			// called.
			count++
			// keep the first init container for this name
			if count == 1 {
				continue
			}
			// prune all other init containers that match this container name
			klog.V(4).InfoS("Removing init container", "containerName", status.Name, "containerID", status.ID.ID, "count", count)
			if err := m.removeContainer(status.ID.ID); err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to remove pod init container %q: %v; Skipping pod %q", status.Name, err, format.Pod(pod)))
				continue
			}
		}
	}
}

// Remove all init containers. Note that this function does not check the state
// of the container because it assumes all init containers have been stopped
// before the call happens.
func (m *kubeGenericRuntimeManager) purgeInitContainers(pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	initContainerNames := sets.NewString()
	for _, container := range pod.Spec.InitContainers {
		initContainerNames.Insert(container.Name)
	}
	for name := range initContainerNames {
		count := 0
		for _, status := range podStatus.ContainerStatuses {
			if status.Name != name {
				continue
			}
			count++
			// Purge all init containers that match this container name
			klog.V(4).InfoS("Removing init container", "containerName", status.Name, "containerID", status.ID.ID, "count", count)
			if err := m.removeContainer(status.ID.ID); err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to remove pod init container %q: %v; Skipping pod %q", status.Name, err, format.Pod(pod)))
				continue
			}
		}
	}
}

// findNextInitContainerToRun returns the status of the last failed container, the
// index of next init container to start, or done if there are no further init containers.
// Status is only returned if an init container is failed, in which case next will
// point to the current container.
func findNextInitContainerToRun(pod *v1.Pod, podStatus *kubecontainer.PodStatus) (status *kubecontainer.Status, next *v1.Container, done bool) {
	if len(pod.Spec.InitContainers) == 0 {
		return nil, nil, true
	}

	// If any of the main containers have status and are Running, then all init containers must
	// have been executed at some point in the past.  However, they could have been removed
	// from the container runtime now, and if we proceed, it would appear as if they
	// never ran and will re-execute improperly.
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		status := podStatus.FindContainerStatusByName(container.Name)
		if status != nil && status.State == kubecontainer.ContainerStateRunning {
			return nil, nil, true
		}
	}

	// If there are failed containers, return the status of the last failed one.
	for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
		container := &pod.Spec.InitContainers[i]
		status := podStatus.FindContainerStatusByName(container.Name)
		if status != nil && isInitContainerFailed(status) {
			return status, container, false
		}
	}

	// There are no failed containers now.
	for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
		container := &pod.Spec.InitContainers[i]
		status := podStatus.FindContainerStatusByName(container.Name)
		if status == nil {
			continue
		}

		// container is still running, return not done.
		if status.State == kubecontainer.ContainerStateRunning {
			return nil, nil, false
		}

		if status.State == kubecontainer.ContainerStateExited {
			// all init containers successful
			if i == (len(pod.Spec.InitContainers) - 1) {
				return nil, nil, true
			}

			// all containers up to i successful, go to i+1
			return nil, &pod.Spec.InitContainers[i+1], false
		}
	}

	return nil, &pod.Spec.InitContainers[0], false
}

// GetContainerLogs returns logs of a specific container.
func (m *kubeGenericRuntimeManager) GetContainerLogs(ctx context.Context, pod *v1.Pod, containerID kubecontainer.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) (err error) {
	status, err := m.runtimeService.ContainerStatus(containerID.ID)
	if err != nil {
		klog.V(4).InfoS("Failed to get container status", "containerID", containerID.String(), "err", err)
		return fmt.Errorf("unable to retrieve container logs for %v", containerID.String())
	}
	return m.ReadLogs(ctx, status.GetLogPath(), containerID.ID, logOptions, stdout, stderr)
}

// GetExec gets the endpoint the runtime will serve the exec request from.
func (m *kubeGenericRuntimeManager) GetExec(id kubecontainer.ContainerID, cmd []string, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	req := &runtimeapi.ExecRequest{
		ContainerId: id.ID,
		Cmd:         cmd,
		Tty:         tty,
		Stdin:       stdin,
		Stdout:      stdout,
		Stderr:      stderr,
	}
	resp, err := m.runtimeService.Exec(req)
	if err != nil {
		return nil, err
	}

	return url.Parse(resp.Url)
}

// GetAttach gets the endpoint the runtime will serve the attach request from.
func (m *kubeGenericRuntimeManager) GetAttach(id kubecontainer.ContainerID, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	req := &runtimeapi.AttachRequest{
		ContainerId: id.ID,
		Stdin:       stdin,
		Stdout:      stdout,
		Stderr:      stderr,
		Tty:         tty,
	}
	resp, err := m.runtimeService.Attach(req)
	if err != nil {
		return nil, err
	}
	return url.Parse(resp.Url)
}

// RunInContainer synchronously executes the command in the container, and returns the output.
func (m *kubeGenericRuntimeManager) RunInContainer(id kubecontainer.ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	stdout, stderr, err := m.runtimeService.ExecSync(id.ID, cmd, timeout)
	// NOTE(tallclair): This does not correctly interleave stdout & stderr, but should be sufficient
	// for logging purposes. A combined output option will need to be added to the ExecSyncRequest
	// if more precise output ordering is ever required.
	return append(stdout, stderr...), err
}

// removeContainer removes the container and the container logs.
// Notice that we remove the container logs first, so that container will not be removed if
// container logs are failed to be removed, and kubelet will retry this later. This guarantees
// that container logs to be removed with the container.
// Notice that we assume that the container should only be removed in non-running state, and
// it will not write container logs anymore in that state.
func (m *kubeGenericRuntimeManager) removeContainer(containerID string) error {
	klog.V(4).InfoS("Removing container", "containerID", containerID)
	// Call internal container post-stop lifecycle hook.
	if err := m.internalLifecycle.PostStopContainer(containerID); err != nil {
		return err
	}

	// Remove the container log.
	// TODO: Separate log and container lifecycle management.
	if err := m.removeContainerLog(containerID); err != nil {
		return err
	}
	// Remove the container.
	return m.runtimeService.RemoveContainer(containerID)
}

// removeContainerLog removes the container log.
func (m *kubeGenericRuntimeManager) removeContainerLog(containerID string) error {
	// Use log manager to remove rotated logs.
	err := m.logManager.Clean(containerID)
	if err != nil {
		return err
	}

	status, err := m.runtimeService.ContainerStatus(containerID)
	if err != nil {
		return fmt.Errorf("failed to get container status %q: %v", containerID, err)
	}
	// Remove the legacy container log symlink.
	// TODO(random-liu): Remove this after cluster logging supports CRI container log path.
	labeledInfo := getContainerInfoFromLabels(status.Labels)
	legacySymlink := legacyLogSymlink(containerID, labeledInfo.ContainerName, labeledInfo.PodName,
		labeledInfo.PodNamespace)
	if err := m.osInterface.Remove(legacySymlink); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove container %q log legacy symbolic link %q: %v",
			containerID, legacySymlink, err)
	}
	return nil
}

// DeleteContainer removes a container.
func (m *kubeGenericRuntimeManager) DeleteContainer(containerID kubecontainer.ContainerID) error {
	return m.removeContainer(containerID.ID)
}
