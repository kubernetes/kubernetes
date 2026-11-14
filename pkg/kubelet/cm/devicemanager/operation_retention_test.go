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

package devicemanager

import (
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/ktesting"
)

func TestAllocatedDevicesRetainPodOperation(t *testing.T) {
	for _, providerKind := range []string{"pending", "completed", "unaware"} {
		t.Run(providerKind, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			const resourceName = "example.com/device"
			manager := &ManagerImpl{
				activePods: func() []*v1.Pod { return nil }, sourcesReady: &sourcesReadyStub{},
				podDevices: newPodDevices(), allocatedDevices: map[string]sets.Set[string]{resourceName: sets.New("pending-device", "unrelated-device")},
			}
			manager.podDevices.insert("pending", "app", resourceName, constructDevices([]string{"pending-device"}), newContainerAllocateResponse())
			manager.podDevices.insert("unrelated", "app", resourceName, constructDevices([]string{"unrelated-device"}), newContainerAllocateResponse())
			retained := providerKind == "pending"
			if providerKind != "unaware" {
				manager.podOperationMayUsePodResources = func(uid types.UID) bool { return uid == "pending" && retained }
			}
			manager.UpdateAllocatedDevices(logger)
			require.Equal(t, retained, manager.podDevices.hasPod("pending"))
			require.False(t, manager.podDevices.hasPod("unrelated"))
			if retained {
				require.Equal(t, sets.New("pending-device"), manager.allocatedDevices[resourceName], "pending allocation cannot become available to another Pod")
			} else {
				require.Empty(t, manager.allocatedDevices[resourceName])
			}
			retained = false
			manager.UpdateAllocatedDevices(logger)
			require.False(t, manager.podDevices.hasPod("pending"))
			require.Empty(t, manager.allocatedDevices[resourceName])
		})
	}
}
