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

package dockershim

import (
	"errors"
	"fmt"
	"math/rand"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

// A helper to create a basic config.
func makeSandboxConfig(name, namespace, uid string, attempt uint32) *runtimeapi.PodSandboxConfig {
	return makeSandboxConfigWithLabelsAndAnnotations(name, namespace, uid, attempt, map[string]string{}, map[string]string{})
}

func makeSandboxConfigWithLabelsAndAnnotations(name, namespace, uid string, attempt uint32, labels, annotations map[string]string) *runtimeapi.PodSandboxConfig {
	return &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      name,
			Namespace: namespace,
			Uid:       uid,
			Attempt:   attempt,
		},
		Labels:      labels,
		Annotations: annotations,
	}
}

// TestListSandboxes creates several sandboxes and then list them to check
// whether the correct metadatas, states, and labels are returned.
func TestListSandboxes(t *testing.T) {
	ds, _, fakeClock := newTestDockerService()
	name, namespace := "foo", "bar"
	configs := []*runtimeapi.PodSandboxConfig{}
	for i := 0; i < 3; i++ {
		c := makeSandboxConfigWithLabelsAndAnnotations(fmt.Sprintf("%s%d", name, i),
			fmt.Sprintf("%s%d", namespace, i), fmt.Sprintf("%d", i), 0,
			map[string]string{"label": fmt.Sprintf("foo%d", i)},
			map[string]string{"annotation": fmt.Sprintf("bar%d", i)},
		)
		configs = append(configs, c)
	}

	expected := []*runtimeapi.PodSandbox{}
	state := runtimeapi.PodSandboxState_SANDBOX_READY
	var createdAt int64 = fakeClock.Now().UnixNano()
	for i := range configs {
		runResp, err := ds.RunPodSandbox(getTestCTX(), &runtimeapi.RunPodSandboxRequest{Config: configs[i]})
		require.NoError(t, err)
		// Prepend to the expected list because ListPodSandbox returns
		// the most recent sandbox first.
		expected = append([]*runtimeapi.PodSandbox{{
			Metadata:    configs[i].Metadata,
			Id:          runResp.PodSandboxId,
			State:       state,
			CreatedAt:   createdAt,
			Labels:      configs[i].Labels,
			Annotations: configs[i].Annotations,
		}}, expected...)
	}
	listResp, err := ds.ListPodSandbox(getTestCTX(), &runtimeapi.ListPodSandboxRequest{})
	require.NoError(t, err)
	assert.Len(t, listResp.Items, len(expected))
	assert.Equal(t, expected, listResp.Items)
}

// TestSandboxStatus tests the basic lifecycle operations and verify that
// the status returned reflects the operations performed.
func TestSandboxStatus(t *testing.T) {
	ds, fDocker, fClock := newTestDockerService()
	labels := map[string]string{"label": "foobar1"}
	annotations := map[string]string{"annotation": "abc"}
	config := makeSandboxConfigWithLabelsAndAnnotations("foo", "bar", "1", 0, labels, annotations)
	r := rand.New(rand.NewSource(0)).Uint32()
	podIP := fmt.Sprintf("10.%d.%d.%d", byte(r>>16), byte(r>>8), byte(r))

	state := runtimeapi.PodSandboxState_SANDBOX_READY
	ct := int64(0)
	expected := &runtimeapi.PodSandboxStatus{
		State:     state,
		CreatedAt: ct,
		Metadata:  config.Metadata,
		Network:   &runtimeapi.PodSandboxNetworkStatus{Ip: podIP, AdditionalIps: []*runtimeapi.PodIP{}},
		Linux: &runtimeapi.LinuxPodSandboxStatus{
			Namespaces: &runtimeapi.Namespace{
				Options: &runtimeapi.NamespaceOption{
					Pid: runtimeapi.NamespaceMode_CONTAINER,
				},
			},
		},
		Labels:      labels,
		Annotations: annotations,
	}

	// Create the sandbox.
	fClock.SetTime(time.Now())
	expected.CreatedAt = fClock.Now().UnixNano()
	runResp, err := ds.RunPodSandbox(getTestCTX(), &runtimeapi.RunPodSandboxRequest{Config: config})
	require.NoError(t, err)
	id := runResp.PodSandboxId

	// Check internal labels
	c, err := fDocker.InspectContainer(id)
	assert.NoError(t, err)
	assert.Equal(t, c.Config.Labels[containerTypeLabelKey], containerTypeLabelSandbox)
	assert.Equal(t, c.Config.Labels[types.KubernetesContainerNameLabel], sandboxContainerName)

	expected.Id = id // ID is only known after the creation.
	statusResp, err := ds.PodSandboxStatus(getTestCTX(), &runtimeapi.PodSandboxStatusRequest{PodSandboxId: id})
	require.NoError(t, err)
	assert.Equal(t, expected, statusResp.Status)

	// Stop the sandbox.
	expected.State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
	_, err = ds.StopPodSandbox(getTestCTX(), &runtimeapi.StopPodSandboxRequest{PodSandboxId: id})
	require.NoError(t, err)
	// IP not valid after sandbox stop
	expected.Network.Ip = ""
	expected.Network.AdditionalIps = []*runtimeapi.PodIP{}
	statusResp, err = ds.PodSandboxStatus(getTestCTX(), &runtimeapi.PodSandboxStatusRequest{PodSandboxId: id})
	require.NoError(t, err)
	assert.Equal(t, expected, statusResp.Status)

	// Remove the container.
	_, err = ds.RemovePodSandbox(getTestCTX(), &runtimeapi.RemovePodSandboxRequest{PodSandboxId: id})
	require.NoError(t, err)
	statusResp, err = ds.PodSandboxStatus(getTestCTX(), &runtimeapi.PodSandboxStatusRequest{PodSandboxId: id})
	assert.Error(t, err, fmt.Sprintf("status of sandbox: %+v", statusResp))
}

// TestSandboxStatusAfterRestart tests that retrieving sandbox status returns
// an IP address even if RunPodSandbox() was not yet called for this pod, as
// would happen on kubelet restart
func TestSandboxStatusAfterRestart(t *testing.T) {
	ds, _, fClock := newTestDockerService()
	config := makeSandboxConfig("foo", "bar", "1", 0)
	r := rand.New(rand.NewSource(0)).Uint32()
	podIP := fmt.Sprintf("10.%d.%d.%d", byte(r>>16), byte(r>>8), byte(r))
	state := runtimeapi.PodSandboxState_SANDBOX_READY
	ct := int64(0)
	expected := &runtimeapi.PodSandboxStatus{
		State:     state,
		CreatedAt: ct,
		Metadata:  config.Metadata,
		Network:   &runtimeapi.PodSandboxNetworkStatus{Ip: podIP, AdditionalIps: []*runtimeapi.PodIP{}},
		Linux: &runtimeapi.LinuxPodSandboxStatus{
			Namespaces: &runtimeapi.Namespace{
				Options: &runtimeapi.NamespaceOption{
					Pid: runtimeapi.NamespaceMode_CONTAINER,
				},
			},
		},
		Labels:      map[string]string{},
		Annotations: map[string]string{},
	}

	// Create the sandbox.
	fClock.SetTime(time.Now())
	expected.CreatedAt = fClock.Now().UnixNano()

	createConfig, err := ds.makeSandboxDockerConfig(config, defaultSandboxImage)
	assert.NoError(t, err)

	createResp, err := ds.client.CreateContainer(*createConfig)
	assert.NoError(t, err)
	err = ds.client.StartContainer(createResp.ID)
	assert.NoError(t, err)

	// Check status without RunPodSandbox() having set up networking
	expected.Id = createResp.ID // ID is only known after the creation.

	statusResp, err := ds.PodSandboxStatus(getTestCTX(), &runtimeapi.PodSandboxStatusRequest{PodSandboxId: createResp.ID})
	require.NoError(t, err)
	assert.Equal(t, expected, statusResp.Status)
}

// TestNetworkPluginInvocation checks that the right SetUpPod and TearDownPod
// calls are made when we run/stop a sandbox.
func TestNetworkPluginInvocation(t *testing.T) {
	ds, _, _ := newTestDockerService()
	mockPlugin := newTestNetworkPlugin(t)
	ds.network = network.NewPluginManager(mockPlugin)
	defer mockPlugin.Finish()

	name := "foo0"
	ns := "bar0"
	c := makeSandboxConfigWithLabelsAndAnnotations(
		name, ns, "0", 0,
		map[string]string{"label": name},
		map[string]string{"annotation": ns},
	)
	cID := kubecontainer.ContainerID{Type: runtimeName, ID: libdocker.GetFakeContainerID(fmt.Sprintf("/%v", makeSandboxName(c)))}

	mockPlugin.EXPECT().Name().Return("mockNetworkPlugin").AnyTimes()
	setup := mockPlugin.EXPECT().SetUpPod(ns, name, cID)
	mockPlugin.EXPECT().TearDownPod(ns, name, cID).After(setup)

	_, err := ds.RunPodSandbox(getTestCTX(), &runtimeapi.RunPodSandboxRequest{Config: c})
	require.NoError(t, err)
	_, err = ds.StopPodSandbox(getTestCTX(), &runtimeapi.StopPodSandboxRequest{PodSandboxId: cID.ID})
	require.NoError(t, err)
}

// TestHostNetworkPluginInvocation checks that *no* SetUp/TearDown calls happen
// for host network sandboxes.
func TestHostNetworkPluginInvocation(t *testing.T) {
	ds, _, _ := newTestDockerService()
	mockPlugin := newTestNetworkPlugin(t)
	ds.network = network.NewPluginManager(mockPlugin)
	defer mockPlugin.Finish()

	name := "foo0"
	ns := "bar0"
	c := makeSandboxConfigWithLabelsAndAnnotations(
		name, ns, "0", 0,
		map[string]string{"label": name},
		map[string]string{"annotation": ns},
	)
	c.Linux = &runtimeapi.LinuxPodSandboxConfig{
		SecurityContext: &runtimeapi.LinuxSandboxSecurityContext{
			NamespaceOptions: &runtimeapi.NamespaceOption{
				Network: runtimeapi.NamespaceMode_NODE,
			},
		},
	}
	cID := kubecontainer.ContainerID{Type: runtimeName, ID: libdocker.GetFakeContainerID(fmt.Sprintf("/%v", makeSandboxName(c)))}

	// No calls to network plugin are expected
	_, err := ds.RunPodSandbox(getTestCTX(), &runtimeapi.RunPodSandboxRequest{Config: c})
	require.NoError(t, err)

	_, err = ds.StopPodSandbox(getTestCTX(), &runtimeapi.StopPodSandboxRequest{PodSandboxId: cID.ID})
	require.NoError(t, err)
}

// TestSetUpPodFailure checks that the sandbox should be not ready when it
// hits a SetUpPod failure.
func TestSetUpPodFailure(t *testing.T) {
	ds, _, _ := newTestDockerService()
	mockPlugin := newTestNetworkPlugin(t)
	ds.network = network.NewPluginManager(mockPlugin)
	defer mockPlugin.Finish()

	name := "foo0"
	ns := "bar0"
	c := makeSandboxConfigWithLabelsAndAnnotations(
		name, ns, "0", 0,
		map[string]string{"label": name},
		map[string]string{"annotation": ns},
	)
	cID := kubecontainer.ContainerID{Type: runtimeName, ID: libdocker.GetFakeContainerID(fmt.Sprintf("/%v", makeSandboxName(c)))}
	mockPlugin.EXPECT().Name().Return("mockNetworkPlugin").AnyTimes()
	mockPlugin.EXPECT().SetUpPod(ns, name, cID).Return(errors.New("setup pod error")).AnyTimes()
	// If SetUpPod() fails, we expect TearDownPod() to immediately follow
	mockPlugin.EXPECT().TearDownPod(ns, name, cID)
	// Assume network plugin doesn't return error, dockershim should still be able to return not ready correctly.
	mockPlugin.EXPECT().GetPodNetworkStatus(ns, name, cID).Return(&network.PodNetworkStatus{IP: net.IP("127.0.0.01")}, nil).AnyTimes()

	t.Logf("RunPodSandbox should return error")
	_, err := ds.RunPodSandbox(getTestCTX(), &runtimeapi.RunPodSandboxRequest{Config: c})
	assert.Error(t, err)

	t.Logf("PodSandboxStatus should be not ready")
	statusResp, err := ds.PodSandboxStatus(getTestCTX(), &runtimeapi.PodSandboxStatusRequest{PodSandboxId: cID.ID})
	require.NoError(t, err)
	assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, statusResp.Status.State)

	t.Logf("ListPodSandbox should also show not ready")
	listResp, err := ds.ListPodSandbox(getTestCTX(), &runtimeapi.ListPodSandboxRequest{})
	require.NoError(t, err)
	var sandbox *runtimeapi.PodSandbox
	for _, s := range listResp.Items {
		if s.Id == cID.ID {
			sandbox = s
			break
		}
	}
	assert.NotNil(t, sandbox)
	assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, sandbox.State)
}
