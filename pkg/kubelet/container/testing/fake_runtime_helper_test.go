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

package testing

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestFakeRuntimeHelperCapturesByPod(t *testing.T) {
	for _, mode := range []string{"sequential", "concurrent"} {
		t.Run(mode, func(t *testing.T) {
			helper := &FakeRuntimeHelper{}
			wantHandlers := map[types.UID]map[string]string{
				"first":  {"app": "first-handler"},
				"second": {"app": "second-handler"},
			}
			wantVolumes := map[types.UID]map[string]kubecontainer.ImageVolumes{
				"first":  {"app": {"data": &runtimeapi.ImageSpec{Image: "first-image"}}},
				"second": {"app": {"data": &runtimeapi.ImageSpec{Image: "second-image"}}},
			}
			wantIPs := map[types.UID][]string{
				"first":  {"10.0.0.1"},
				"second": {"10.0.0.2"},
			}

			var workers sync.WaitGroup
			start := make(chan struct{})
			for uid, handlers := range wantHandlers {
				capture := func() {
					pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: uid}}
					container := &v1.Container{Name: "app"}
					_, _, err := helper.GenerateRunContainerOptions(context.Background(), pod, container, "", nil, wantVolumes[uid][container.Name], handlers[container.Name])
					assert.NoError(t, err)
					assert.NoError(t, helper.UpdatePodRestoreHosts(context.Background(), pod, wantIPs[uid]))
				}
				if mode == "concurrent" {
					workers.Go(func() {
						<-start
						capture()
					})
				} else {
					capture()
				}
			}
			close(start)
			workers.Wait()

			assert.Equal(t, wantHandlers, helper.ContainerHandlers)
			assert.Equal(t, wantVolumes, helper.ContainerImageVolumes)
			assert.Equal(t, wantIPs, helper.PodRestoreIPs)
		})
	}
}
