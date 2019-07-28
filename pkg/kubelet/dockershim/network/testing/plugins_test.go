/*
Copyright 2014 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"net"
	"sync"
	"testing"

	utilsets "k8s.io/apimachinery/pkg/util/sets"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	sysctltest "k8s.io/kubernetes/pkg/util/sysctl/testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func TestSelectDefaultPlugin(t *testing.T) {
	all_plugins := []network.NetworkPlugin{}
	plug, err := network.InitNetworkPlugin(all_plugins, "", NewFakeHost(nil), kubeletconfig.HairpinNone, "10.0.0.0/8", network.UseDefaultMTU)
	if err != nil {
		t.Fatalf("Unexpected error in selecting default plugin: %v", err)
	}
	if plug == nil {
		t.Fatalf("Failed to select the default plugin.")
	}
	if plug.Name() != network.DefaultPluginName {
		t.Errorf("Failed to select the default plugin. Expected %s. Got %s", network.DefaultPluginName, plug.Name())
	}
}

func TestInit(t *testing.T) {
	tests := []struct {
		setting     string
		expectedLen int
	}{
		{
			setting:     "net/bridge/bridge-nf-call-iptables",
			expectedLen: 1,
		},
		{
			setting:     "net/bridge/bridge-nf-call-ip6tables",
			expectedLen: 2,
		},
	}
	for _, tt := range tests {
		sysctl := sysctltest.NewFake()
		sysctl.Settings[tt.setting] = 0
		plug := &network.NoopNetworkPlugin{}
		plug.Sysctl = sysctl
		plug.Init(NewFakeHost(nil), kubeletconfig.HairpinNone, "10.0.0.0/8", network.UseDefaultMTU)
		// Verify the sysctl specified is set
		assert.Equal(t, 1, sysctl.Settings[tt.setting], tt.setting+" sysctl should have been set")
		// Verify iptables is always set
		assert.Equal(t, 1, sysctl.Settings["net/bridge/bridge-nf-call-iptables"], "net/bridge/bridge-nf-call-iptables sysctl should have been set")
		// Verify ip6tables is only set if it existed
		assert.Len(t, sysctl.Settings, tt.expectedLen, "length wrong for "+tt.setting)
	}
}

func TestPluginManager(t *testing.T) {
	ctrl := gomock.NewController(t)
	fnp := NewMockNetworkPlugin(ctrl)
	defer fnp.Finish()
	pm := network.NewPluginManager(fnp)

	fnp.EXPECT().Name().Return("someNetworkPlugin").AnyTimes()

	allCreatedWg := sync.WaitGroup{}
	allCreatedWg.Add(1)
	allDoneWg := sync.WaitGroup{}

	// 10 pods, 4 setup/status/teardown runs each.  Ensure that network locking
	// works and the pod map isn't concurrently accessed
	for i := 0; i < 10; i++ {
		podName := fmt.Sprintf("pod%d", i)
		containerID := kubecontainer.ContainerID{ID: podName}

		fnp.EXPECT().SetUpPod("", podName, containerID).Return(nil).Times(4)
		fnp.EXPECT().GetPodNetworkStatus("", podName, containerID).Return(&network.PodNetworkStatus{IP: net.ParseIP("1.2.3.4")}, nil).Times(4)
		fnp.EXPECT().TearDownPod("", podName, containerID).Return(nil).Times(4)

		for x := 0; x < 4; x++ {
			allDoneWg.Add(1)
			go func(name string, id kubecontainer.ContainerID, num int) {
				defer allDoneWg.Done()

				// Block all goroutines from running until all have
				// been created and are ready.  This ensures we
				// have more pod network operations running
				// concurrently.
				allCreatedWg.Wait()

				if err := pm.SetUpPod("", name, id, nil, nil); err != nil {
					t.Errorf("Failed to set up pod %q: %v", name, err)
					return
				}

				if _, err := pm.GetPodNetworkStatus("", name, id); err != nil {
					t.Errorf("Failed to inspect pod %q: %v", name, err)
					return
				}

				if err := pm.TearDownPod("", name, id); err != nil {
					t.Errorf("Failed to tear down pod %q: %v", name, err)
					return
				}
			}(podName, containerID, x)
		}
	}
	// Block all goroutines from running until all have been created and started
	allCreatedWg.Done()

	// Wait for them all to finish
	allDoneWg.Wait()
}

type hookableFakeNetworkPluginSetupHook func(namespace, name string, id kubecontainer.ContainerID)

type hookableFakeNetworkPlugin struct {
	setupHook hookableFakeNetworkPluginSetupHook
}

func newHookableFakeNetworkPlugin(setupHook hookableFakeNetworkPluginSetupHook) *hookableFakeNetworkPlugin {
	return &hookableFakeNetworkPlugin{
		setupHook: setupHook,
	}
}

func (p *hookableFakeNetworkPlugin) Init(host network.Host, hairpinMode kubeletconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error {
	return nil
}

func (p *hookableFakeNetworkPlugin) Event(name string, details map[string]interface{}) {
}

func (p *hookableFakeNetworkPlugin) Name() string {
	return "fakeplugin"
}

func (p *hookableFakeNetworkPlugin) Capabilities() utilsets.Int {
	return utilsets.NewInt()
}

func (p *hookableFakeNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID, annotations, options map[string]string) error {
	if p.setupHook != nil {
		p.setupHook(namespace, name, id)
	}
	return nil
}

func (p *hookableFakeNetworkPlugin) TearDownPod(string, string, kubecontainer.ContainerID) error {
	return nil
}

func (p *hookableFakeNetworkPlugin) GetPodNetworkStatus(string, string, kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	return &network.PodNetworkStatus{IP: net.ParseIP("10.1.2.3")}, nil
}

func (p *hookableFakeNetworkPlugin) Status() error {
	return nil
}

// Ensure that one pod's network operations don't block another's.  If the
// test is successful (eg, first pod doesn't block on second) the test
// will complete.  If unsuccessful, it will hang and get killed.
func TestMultiPodParallelNetworkOps(t *testing.T) {
	podWg := sync.WaitGroup{}
	podWg.Add(1)

	// Can't do this with MockNetworkPlugin because the gomock controller
	// has its own locks which don't allow the parallel network operation
	// to proceed.
	didWait := false
	fakePlugin := newHookableFakeNetworkPlugin(func(podNamespace, podName string, id kubecontainer.ContainerID) {
		if podName == "waiter" {
			podWg.Wait()
			didWait = true
		}
	})
	pm := network.NewPluginManager(fakePlugin)

	opsWg := sync.WaitGroup{}

	// Start the pod that will wait for the other to complete
	opsWg.Add(1)
	go func() {
		defer opsWg.Done()

		podName := "waiter"
		containerID := kubecontainer.ContainerID{ID: podName}

		// Setup will block on the runner pod completing.  If network
		// operations locking isn't correct (eg pod network operations
		// block other pods) setUpPod() will never return.
		if err := pm.SetUpPod("", podName, containerID, nil, nil); err != nil {
			t.Errorf("Failed to set up waiter pod: %v", err)
			return
		}

		if err := pm.TearDownPod("", podName, containerID); err != nil {
			t.Errorf("Failed to tear down waiter pod: %v", err)
			return
		}
	}()

	opsWg.Add(1)
	go func() {
		defer opsWg.Done()
		// Let other pod proceed
		defer podWg.Done()

		podName := "runner"
		containerID := kubecontainer.ContainerID{ID: podName}

		if err := pm.SetUpPod("", podName, containerID, nil, nil); err != nil {
			t.Errorf("Failed to set up runner pod: %v", err)
			return
		}

		if err := pm.TearDownPod("", podName, containerID); err != nil {
			t.Errorf("Failed to tear down runner pod: %v", err)
			return
		}
	}()

	opsWg.Wait()

	if !didWait {
		t.Errorf("waiter pod didn't wait for runner pod!")
	}
}
