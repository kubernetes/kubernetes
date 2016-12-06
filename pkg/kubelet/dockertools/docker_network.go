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

package dockertools

import (
	"fmt"
	"sync"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
)

type dockerNetwork struct {
	// Network plugin.
	networkPlugin network.NetworkPlugin

	// Pod mutex map lock
	podsLock sync.Mutex
	pods     map[string]*podLock
}

func newDockerNetwork(networkPlugin network.NetworkPlugin) *dockerNetwork {
	return &dockerNetwork{
		networkPlugin: networkPlugin,
		pods:          make(map[string]*podLock),
	}
}

// container must not be nil
func getDockerNetworkMode(container *dockertypes.ContainerJSON) string {
	if container.HostConfig != nil {
		return string(container.HostConfig.NetworkMode)
	}
	return ""
}

func (dn *dockerNetwork) pluginDisablesDockerNetworking() bool {
	return dn.networkPlugin.Name() == "cni" || dn.networkPlugin.Name() == "kubenet"
}

type podLock struct {
	refcount uint
	mu       sync.Mutex
}

// Lock network operations for a specific pod.  If that pod is not yet in
// the pod map, it will be added.  The reference count for the pod will
// be increased.
func (dn *dockerNetwork) podLock(podNamespace, podName string) *sync.Mutex {
	dn.podsLock.Lock()
	defer dn.podsLock.Unlock()

	podKey := getPodKey(podNamespace, podName)
	lock, ok := dn.pods[podKey]
	if !ok {
		lock = &podLock{}
		dn.pods[podKey] = lock
	}
	lock.refcount++
	return &lock.mu
}

// Unlock netowrk operations for a specific pod. If this operation is the
// only outstanding operation for the pod, it will be removed from the
// pod list.
func (dn *dockerNetwork) podUnlock(podNamespace, podName string) {
	dn.podsLock.Lock()
	defer dn.podsLock.Unlock()

	podKey := getPodKey(podNamespace, podName)
	lock, ok := dn.pods[podKey]
	if !ok || lock.refcount == 0 {
		glog.Warningf("Unbalanced pod lock unref for %s/%s", podNamespace, podName)
		return
	}
	lock.refcount--
	lock.mu.Unlock()
	if lock.refcount == 0 {
		delete(dn.pods, podKey)
	}
}

func getPodKey(podNamespace, podName string) string {
	return fmt.Sprintf("%s/%s", podNamespace, podName)
}

// determineContainerIP determines the IP address of the given container.  It is expected
// that the container passed is the infrastructure container of a pod and the responsibility
// of the caller to ensure that the correct container is passed.
func (dn *dockerNetwork) determineContainerIP(podNamespace, podName string, container *dockertypes.ContainerJSON) (string, error) {
	dn.podLock(podNamespace, podName).Lock()
	defer dn.podUnlock(podNamespace, podName)

	result := getContainerIP(container)

	networkMode := getDockerNetworkMode(container)
	isHostNetwork := networkMode == namespaceModeHost

	// For host networking or default network plugin, GetPodNetworkStatus doesn't work
	if !isHostNetwork && dn.networkPlugin.Name() != network.DefaultPluginName {
		netStatus, err := dn.networkPlugin.GetPodNetworkStatus(podNamespace, podName, kubecontainer.DockerID(container.ID).ContainerID())
		if err != nil {
			return result, fmt.Errorf("NetworkPlugin %s failed on the status hook for pod '%s': %v", dn.networkPlugin.Name(), podName, err)
		} else if netStatus != nil {
			result = netStatus.IP.String()
		}
	}

	return result, nil
}

func (dn *dockerNetwork) setUpPod(podNamespace, podName string, containerID kubecontainer.ContainerID) error {
	dn.podLock(podNamespace, podName).Lock()
	defer dn.podUnlock(podNamespace, podName)

	fullPodName := kubecontainer.BuildPodFullName(podName, podNamespace)
	glog.V(3).Infof("Calling network plugin %s to setup pod for %s", dn.networkPlugin.Name(), fullPodName)
	if err := dn.networkPlugin.SetUpPod(podNamespace, podName, containerID); err != nil {
		// TODO: (random-liu) There shouldn't be "Skipping pod" in sync result message
		return fmt.Errorf("Failed to setup network for pod %q using network plugins %q: %v; Skipping pod", fullPodName, dn.networkPlugin.Name(), err)
	}

	return nil
}

func (dn *dockerNetwork) tearDownPod(podNamespace, podName string, containerID kubecontainer.ContainerID) error {
	dn.podLock(podNamespace, podName).Lock()
	defer dn.podUnlock(podNamespace, podName)

	fullPodName := kubecontainer.BuildPodFullName(podName, podNamespace)
	glog.V(3).Infof("Calling network plugin %s to tear down pod for %s", dn.networkPlugin.Name(), fullPodName)
	if err := dn.networkPlugin.TearDownPod(podNamespace, podName, containerID); err != nil {
		return fmt.Errorf("Failed to teardown network for pod %q using network plugins %q: %v", fullPodName, dn.networkPlugin.Name(), err)
	}

	return nil
}
