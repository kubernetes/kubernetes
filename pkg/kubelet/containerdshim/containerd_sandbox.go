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

package containerdshim

import (
	gocontext "context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/docker/containerd/api/services/execution"
	"github.com/docker/containerd/api/types/container"
	"github.com/docker/containerd/api/types/mount"
	protobuf "github.com/gogo/protobuf/types"
	"github.com/golang/glog"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
)

// containerStore is used to store container metadata.
// TODO: Consider to checkpoint ourselves or use containerd metadata store.
var sandboxStore map[string]*runtimeapi.PodSandboxStatus = map[string]*runtimeapi.PodSandboxStatus{}
var sandboxStoreLock sync.RWMutex

// TODO: use Kubernetes pause image when gcr.io is supported.
const sandboxImage = "docker.io/library/alpine:latest"

// RunPodSandbox creates and runs a pod-level sandbox.
// P0
func (cs *containerdService) RunPodSandbox(config *runtimeapi.PodSandboxConfig) (string, error) {
	// TODO Report error if already exist.
	glog.V(2).Infof("RunPodSandbox for pod %+v", config.GetMetadata())
	if config == nil {
		return "", fmt.Errorf("config is nil for pod %+v", config.GetMetadata())
	}
	// TODO Add support for label and annotation
	sandboxID := dockershim.MakeSandboxName(config)

	sandboxDir, err := ensureContainerDir(sandboxID)
	if err != nil {
		return "", err
	}

	// Note that there is lock inside `PullImage`
	if _, err := cs.PullImage(&runtimeapi.ImageSpec{Image: sandboxImage}, nil); err != nil {
		return "", fmt.Errorf("failed to pull image %q: %v", sandboxImage, err)
	}

	rootfsPath := filepath.Join(sandboxDir, "rootfs")
	if err := cs.createRootfs(sandboxImage, rootfsPath); err != nil {
		return "", err
	}

	// Sleep forever
	// TODO: Use pause container, get default entrypoint from image config.
	processArgs := []string{"sh", "-c", "while true; do sleep 1000000000; done"}

	// TODO: Set other configs, such as envs, working directory etc.
	s := defaultOCISpec(sandboxID, processArgs, rootfsPath, false)

	// Set cgroup parent
	if config.GetLinux().GetCgroupParent() != "" && s.Linux != nil {
		cgroupPath := filepath.Join(config.GetLinux().GetCgroupParent(), sandboxID)
		s.Linux.CgroupsPath = &cgroupPath
	}

	s.Hostname = config.GetHostname()

	data, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	create := &execution.CreateRequest{
		ID: sandboxID,
		Spec: &protobuf.Any{
			TypeUrl: specs.Version,
			Value:   data,
		},
		// mikebrow for now configure to bind mount the rootfs
		Rootfs: []*mount.Mount{
			{
				Type:   "bind",
				Source: rootfsPath,
				Options: []string{
					"rw",
					"rbind",
				},
			},
		},
		Runtime:  "linux",
		Terminal: false,
		Stdin:    filepath.Join(sandboxDir, "stdin"), // mikebrow TODO needed for console
		Stdout:   filepath.Join(sandboxDir, "stdout"),
		Stderr:   filepath.Join(sandboxDir, "stderr"),
	}
	stream, err := prepareStdio(create.Stdin, create.Stdout, create.Stderr, create.Terminal)
	if err != nil {
		return "", err
	}

	// mikebrow TODO proper console handling
	// TODO Do we need permanent namespace?
	glog.V(2).Infof("Create infra container for sandbox %s sandbox directory %s", sandboxID, sandboxDir)
	response, err := cs.containerService.Create(gocontext.Background(), create)
	if err != nil {
		return "", err
	}
	// We don't need any stream for the infra container, just close them
	stream.Close()

	sandboxStoreLock.Lock()
	defer sandboxStoreLock.Unlock()
	sandboxStore[sandboxID] = &runtimeapi.PodSandboxStatus{
		Id:          sandboxID,
		Metadata:    config.GetMetadata(),
		CreatedAt:   time.Now().UnixNano(),
		Labels:      config.GetLabels(),
		Annotations: config.GetAnnotations(),
	}

	// TODO Do we really need to start here?
	_, err = cs.containerService.Start(gocontext.Background(), &execution.StartRequest{ID: sandboxID})
	if err != nil {
		return "", err
	}

	return response.ID, nil
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be force terminated.
// P0
func (cs *containerdService) StopPodSandbox(sandboxID string) error {
	sandboxStoreLock.RLock()
	defer sandboxStoreLock.RUnlock()
	glog.V(2).Infof("StopPodSandbox called with %s", sandboxID)
	if _, ok := sandboxStore[sandboxID]; !ok {
		return fmt.Errorf("sandbox not found %s", sandboxID)
	}
	_, err := cs.containerService.Delete(gocontext.Background(), &execution.DeleteRequest{ID: sandboxID})
	return err
}

// RemovePodSandbox deletes the sandbox. If there are any running containers in the
// sandbox, they should be force deleted.
// P1
func (cs *containerdService) RemovePodSandbox(sandboxID string) error {
	sandboxStoreLock.RLock()
	defer sandboxStoreLock.RUnlock()
	glog.V(2).Infof("RemovePodSandbox called with %s", sandboxID)
	if _, ok := sandboxStore[sandboxID]; !ok {
		return fmt.Errorf("sandbox not found %s", sandboxID)
	}
	sandboxDir := getContainerDir(sandboxID)
	rootfsPath := filepath.Join(sandboxDir, "rootfs")
	if err := exec.Command("umount", rootfsPath).Run(); err != nil {
		return fmt.Errorf("failed to umount rootfs %s: %v", rootfsPath, err)
	}
	if err := os.RemoveAll(sandboxDir); err != nil {
		return err
	}
	delete(sandboxStore, sandboxID)
	return nil
}

// PodSandboxStatus returns the Status of the PodSandbox.
// P0
func (cs *containerdService) PodSandboxStatus(sandboxID string) (*runtimeapi.PodSandboxStatus, error) {
	sandboxStoreLock.RLock()
	defer sandboxStoreLock.RUnlock()
	glog.V(4).Infof("PodSandboxStatus called with %s", sandboxID)
	status, ok := sandboxStore[sandboxID]
	if !ok {
		return nil, fmt.Errorf("sandbox not found %v", sandboxID)
	}
	status.State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
	s, err := cs.containerService.Info(gocontext.Background(), &execution.InfoRequest{ID: sandboxID})
	if err != nil {
		if !strings.Contains(err.Error(), "container does not exist") {
			return nil, fmt.Errorf("failed to get sandbox info %q: %v", sandboxID, err)
		}
		return status, nil
	}
	if s.Status == container.Status_RUNNING {
		status.State = runtimeapi.PodSandboxState_SANDBOX_READY
	}
	// TODO Network plugin.
	status.Network = &runtimeapi.PodSandboxNetworkStatus{Ip: "127.0.0.1"}
	status.Linux = &runtimeapi.LinuxPodSandboxStatus{}
	return status, nil
}

// ListPodSandbox returns a list of SandBoxes.
// P0
func (cs *containerdService) ListPodSandbox(filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	sandboxStoreLock.RLock()
	defer sandboxStoreLock.RUnlock()
	resp, err := cs.containerService.List(gocontext.Background(), &execution.ListRequest{})
	if err != nil {
		return nil, fmt.Errorf("failed to list containers from containerd: %v", err)
	}

	var sandboxes []*runtimeapi.PodSandbox
	for _, status := range sandboxStore {
		sandbox := statusToSandbox(status)
		sandbox.State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
		for _, s := range resp.Containers {
			if s.ID == sandbox.Id {
				if s.Status == container.Status_RUNNING {
					sandbox.State = runtimeapi.PodSandboxState_SANDBOX_READY
				}
				break
			}
		}
		sandboxes = append(sandboxes, sandbox)
	}

	if filter != nil {
		if filter.Id != "" {
			var filtered []*runtimeapi.PodSandbox
			for _, s := range sandboxes {
				if filter.Id == s.Id {
					filtered = append(filtered, s)
				}
			}
			sandboxes = filtered
		}

		if filter.State != nil {
			var filtered []*runtimeapi.PodSandbox
			for _, s := range sandboxes {
				if s.State == filter.GetState().State {
					filtered = append(filtered, s)
				}
			}
			sandboxes = filtered
		}

		if filter.LabelSelector != nil {
			var filtered []*runtimeapi.PodSandbox
			for _, s := range sandboxes {
				match := true
				for k, v := range filter.LabelSelector {
					if s.Labels[k] != v {
						match = false
						break
					}
				}
				if match {
					filtered = append(filtered, s)
				}
			}
			sandboxes = filtered
		}
	}
	return sandboxes, nil
}
