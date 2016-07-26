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
	"fmt"
	"io"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

var (
	version                 = kubeRuntimeAPIVersion
	fakeImageSize    uint64 = 1
	fakeRuntimeName         = "fakeRuntime"
	fakePodSandboxIP        = "192.168.192.168"
)

type PodSandboxWithState struct {
	// creation timestamp
	createdAt int64
	// the config of this pod sandbox
	config *runtimeApi.PodSandboxConfig
	// the state of this pod sandbox
	state runtimeApi.PodSandBoxState
}

type ContainerWithState struct {
	// creation timestamp
	createdAt int64
	// the id of this container
	containerID string
	// the sandbox id of this container
	podSandBoxID string
	// the id of the image
	imageID string
	// the state of this container
	state runtimeApi.ContainerState
	// the config of this container
	containerConfig *runtimeApi.ContainerConfig
	// the sandbox config of this container
	sandboxConfig *runtimeApi.PodSandboxConfig
}

func (c *ContainerWithState) toKubeContainer() *kubecontainer.Container {
	_, _, _, cName, hash, _ := parseContainerName(c.containerConfig.GetName())
	return &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: fakeRuntimeName, ID: c.containerID},
		Name:  cName,
		Image: c.containerConfig.Image.GetImage(),
		Hash:  hash,
		State: toKubeContainerState(c.state),
	}
}

type fakeKubeRuntime struct {
	sync.Mutex

	Images     map[string]*runtimeApi.Image
	Containers map[string]*ContainerWithState
	Sandboxes  map[string]*PodSandboxWithState

	Called []string
}

func NewFakeKubeRuntime() *fakeKubeRuntime {
	s := &fakeKubeRuntime{
		Images:     make(map[string]*runtimeApi.Image),
		Containers: make(map[string]*ContainerWithState),
		Sandboxes:  make(map[string]*PodSandboxWithState),
	}

	return s
}

func stringInSlice(in string, list []string) bool {
	for _, v := range list {
		if v == in {
			return true
		}
	}

	return false
}

func makeFakeImage(image string) *runtimeApi.Image {
	return &runtimeApi.Image{
		Id:       &image,
		Size_:    &fakeImageSize,
		RepoTags: []string{image},
	}
}

func (r *fakeKubeRuntime) SetFakeSandboxes(sandboxes []*PodSandboxWithState) {
	r.Lock()
	defer r.Unlock()

	r.Sandboxes = make(map[string]*PodSandboxWithState)
	for _, sandbox := range sandboxes {
		sandboxID := sandbox.config.GetName()
		r.Sandboxes[sandboxID] = sandbox
	}
}

func (r *fakeKubeRuntime) SetFakeContainers(containers []*ContainerWithState) {
	r.Lock()
	defer r.Unlock()

	images := sets.NewString()
	r.Containers = make(map[string]*ContainerWithState)
	for _, c := range containers {
		containerID := c.containerConfig.GetName()
		r.Containers[containerID] = c
		images.Insert(c.containerConfig.Image.GetImage())
	}

	r.Images = make(map[string]*runtimeApi.Image)
	for _, image := range images.List() {
		r.Images[image] = makeFakeImage(image)
	}
}

func (r *fakeKubeRuntime) SetFakeImages(images []string) {
	r.Lock()
	defer r.Unlock()

	r.Images = make(map[string]*runtimeApi.Image)
	for _, image := range images {
		r.Images[image] = makeFakeImage(image)
	}
}

func (r *fakeKubeRuntime) ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListImages")

	images := make([]*runtimeApi.Image, 0)
	for _, img := range r.Images {
		if filter != nil && filter.Image != nil {
			if !stringInSlice(filter.Image.GetImage(), img.RepoTags) {
				continue
			}
		}

		images = append(images, img)
	}
	return images, nil
}

func (r *fakeKubeRuntime) ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ImageStatus")

	if img, ok := r.Images[image.GetImage()]; ok {
		return img, nil
	}

	return nil, fmt.Errorf("Image not found")
}

func (r *fakeKubeRuntime) PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "PullImage")

	imageID := image.GetImage()
	if _, ok := r.Images[imageID]; !ok {
		r.Images[imageID] = makeFakeImage(image.GetImage())
	}

	return nil
}

func (r *fakeKubeRuntime) RemoveImage(image *runtimeApi.ImageSpec) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RemoveImage")

	if _, ok := r.Images[image.GetImage()]; ok {
		delete(r.Images, image.GetImage())
	}

	return nil
}

func (r *fakeKubeRuntime) Version(apiVersion string) (*runtimeApi.VersionResponse, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Version")

	return &runtimeApi.VersionResponse{
		Version:           &version,
		RuntimeName:       &fakeRuntimeName,
		RuntimeVersion:    &version,
		RuntimeApiVersion: &version,
	}, nil
}

func (r *fakeKubeRuntime) CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "CreatePodSandbox")

	sandboxID := config.GetName()
	r.Sandboxes[sandboxID] = &PodSandboxWithState{
		config:    config,
		state:     runtimeApi.PodSandBoxState_READY,
		createdAt: time.Now().Unix(),
	}

	return sandboxID, nil
}

func (r *fakeKubeRuntime) StopPodSandbox(podSandBoxID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StopPodSandbox")

	if s, ok := r.Sandboxes[podSandBoxID]; ok {
		s.state = runtimeApi.PodSandBoxState_NOTREADY
	} else {
		return fmt.Errorf("PodSandbox not found")
	}

	return nil
}

func (r *fakeKubeRuntime) DeletePodSandbox(podSandBoxID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "DeletePodSandbox")

	if _, ok := r.Sandboxes[podSandBoxID]; ok {
		delete(r.Sandboxes, podSandBoxID)
	}

	return nil
}

func (r *fakeKubeRuntime) PodSandboxStatus(podSandBoxID string) (*runtimeApi.PodSandboxStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "PodSandboxStatus")

	s, ok := r.Sandboxes[podSandBoxID]
	if !ok {
		return nil, fmt.Errorf("PodSandbox not found")
	}

	return &runtimeApi.PodSandboxStatus{
		Id:        &podSandBoxID,
		Name:      s.config.Name,
		CreatedAt: &s.createdAt,
		State:     &s.state,
		Network: &runtimeApi.PodSandboxNetworkStatus{
			Ip: &fakePodSandboxIP,
		},
		Labels:      s.config.Labels,
		Annotations: s.config.Annotations,
	}, nil
}

func filterInLabels(filter, labels map[string]string) bool {
	for k, v := range filter {
		if value, ok := labels[k]; ok {
			if value != v {
				return false
			}
		} else {
			return false
		}
	}

	return true
}

func (r *fakeKubeRuntime) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListPodSandbox")

	result := make([]*runtimeApi.PodSandbox, 0)
	for id, s := range r.Sandboxes {
		if filter != nil {
			if filter.Id != nil && filter.GetId() != id {
				continue
			}
			if filter.Name != nil && filter.GetName() != s.config.GetName() {
				continue
			}
			if filter.State != nil && filter.GetState() != s.state {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, s.config.Labels) {
				continue
			}
		}

		result = append(result, &runtimeApi.PodSandbox{
			Id:        &id,
			Name:      s.config.Name,
			State:     &s.state,
			CreatedAt: &s.createdAt,
			Labels:    s.config.Labels,
		})
	}

	return result, nil
}

func (r *fakeKubeRuntime) CreateContainer(podSandBoxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "CreateContainer")

	imageID := ""
	for _, img := range r.Images {
		if stringInSlice(config.Image.GetImage(), img.RepoTags) {
			imageID = img.GetId()
			break
		}
	}
	if imageID == "" {
		return "", fmt.Errorf("Image %s not found", config.Image.GetImage())
	}

	containerID := config.GetName()
	r.Containers[containerID] = &ContainerWithState{
		createdAt:       time.Now().Unix(),
		containerID:     containerID,
		podSandBoxID:    podSandBoxID,
		containerConfig: config,
		sandboxConfig:   sandboxConfig,
		imageID:         imageID,
		state:           runtimeApi.ContainerState_CREATED,
	}

	return containerID, nil
}

func (r *fakeKubeRuntime) StartContainer(rawContainerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StartContainer")

	if c, ok := r.Containers[rawContainerID]; ok {
		c.state = runtimeApi.ContainerState_RUNNING
	} else {
		return fmt.Errorf("Container not found")
	}

	return nil
}

func (r *fakeKubeRuntime) StopContainer(rawContainerID string, timeout int64) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "StopContainer")

	if c, ok := r.Containers[rawContainerID]; ok {
		c.state = runtimeApi.ContainerState_EXITED
	} else {
		return fmt.Errorf("Container not found")
	}

	return nil
}

func (r *fakeKubeRuntime) RemoveContainer(rawContainerID string) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "RemoveContainer")

	if _, ok := r.Containers[rawContainerID]; ok {
		delete(r.Containers, rawContainerID)
	}

	return nil
}

func (r *fakeKubeRuntime) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ListContainers")

	result := make([]*runtimeApi.Container, 0)
	for _, s := range r.Containers {
		if filter != nil {
			if filter.Id != nil && filter.GetId() != s.containerID {
				continue
			}
			if filter.Name != nil && filter.GetName() != s.containerConfig.GetName() {
				continue
			}
			if filter.PodSandboxId != nil && filter.GetPodSandboxId() != s.podSandBoxID {
				continue
			}
			if filter.State != nil && filter.GetState() != s.state {
				continue
			}
			if filter.LabelSelector != nil && !filterInLabels(filter.LabelSelector, s.containerConfig.Labels) {
				continue
			}
		}

		result = append(result, &runtimeApi.Container{
			Id:       &s.containerID,
			Name:     s.containerConfig.Name,
			State:    &s.state,
			Image:    s.containerConfig.Image,
			ImageRef: &s.imageID,
			Labels:   s.containerConfig.Labels,
		})
	}

	return result, nil
}

func (r *fakeKubeRuntime) ContainerStatus(rawContainerID string) (*runtimeApi.ContainerStatus, error) {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "ContainerStatus")

	c, ok := r.Containers[rawContainerID]
	if !ok {
		return nil, fmt.Errorf("Not found")
	}

	return &runtimeApi.ContainerStatus{
		Id:        &c.containerID,
		Name:      c.containerConfig.Name,
		State:     &c.state,
		CreatedAt: &c.createdAt,
		Image:     c.containerConfig.Image,
		ImageRef:  &c.imageID,
		Labels:    c.containerConfig.Labels,
	}, nil
}

func (r *fakeKubeRuntime) Exec(rawContainerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	r.Lock()
	defer r.Unlock()

	r.Called = append(r.Called, "Exec")
	return nil
}

// getContainerByKubeUID gets a list of ContainerWithState by pod UID
func (r *fakeKubeRuntime) getContainersByKubeUID(uid string) []*ContainerWithState {
	result := []*ContainerWithState{}

	for _, c := range r.Containers {
		if c.containerConfig.Labels[types.KubernetesPodUIDLabel] == uid {
			result = append(result, c)
		}
	}

	return result
}

// getPodSandboxByKubeUID gets a list of PodSandboxWithState by pod UID
// It returns a list because there may be more than one pod sandboxes belonging to one Pod (e.g. stopped)
func (r *fakeKubeRuntime) getPodSandboxByKubeUID(uid string) []*PodSandboxWithState {
	result := []*PodSandboxWithState{}

	for _, s := range r.Sandboxes {
		if s.config.Labels[types.KubernetesPodUIDLabel] == uid {
			result = append(result, s)
		}
	}

	return result
}

// getExpectedKubePodStatus get kubecontainer.PodStatus by api.Pod, should only be used after syncPod
func (r *fakeKubeRuntime) getExpectedKubePodStatus(pod *api.Pod) (*kubecontainer.PodStatus, error) {
	sandboxes := r.getPodSandboxByKubeUID(string(pod.UID))
	containers := r.getContainersByKubeUID(string(pod.UID))

	podSandboxStateuses := []*runtimeApi.PodSandboxStatus{}
	for _, s := range sandboxes {
		status, err := r.PodSandboxStatus(s.config.GetName())
		if err != nil {
			return nil, err
		}

		podSandboxStateuses = append(podSandboxStateuses, status)
	}

	containerStateuses := []*kubecontainer.ContainerStatus{}
	for _, c := range containers {
		labeledContainerInfo := getContainerInfoFromLabels(c.containerConfig.Labels)
		annotatedContainerInfo := getContainerInfoFromAnnotations(c.containerConfig.Annotations)
		containerStateuses = append(containerStateuses, &kubecontainer.ContainerStatus{
			ID: kubecontainer.ContainerID{
				ID:   c.containerID,
				Type: fakeRuntimeName,
			},
			Name:      labeledContainerInfo.Name,
			Image:     c.containerConfig.GetImage().GetImage(),
			CreatedAt: time.Unix(c.createdAt, 0),
			Hash:      annotatedContainerInfo.Hash,
			ImageID:   c.imageID,
			State:     toKubeContainerState(c.state),
		})
	}

	return &kubecontainer.PodStatus{
		ID:                pod.UID,
		Name:              pod.Name,
		Namespace:         pod.Namespace,
		IP:                fakePodSandboxIP,
		SandboxStatus:     podSandboxStateuses,
		ContainerStatuses: containerStateuses,
	}, nil
}
