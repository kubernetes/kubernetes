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

package metrics

import (
	"context"
	"testing"

	coordinationv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
)

func TestNewMetricsGrabberPicksLeader(t *testing.T) {
	staticPod := func(component, nodeName string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: component + "-" + nodeName, Namespace: metav1.NamespaceSystem},
			Spec:       v1.PodSpec{NodeName: nodeName, HostNetwork: true},
		}
	}
	snapshotPod := func(name, nodeName string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: metav1.NamespaceSystem},
			Spec:       v1.PodSpec{NodeName: nodeName},
		}
	}
	lease := func(name, holder string) *coordinationv1.Lease {
		return &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: metav1.NamespaceSystem},
			Spec:       coordinationv1.LeaseSpec{HolderIdentity: &holder},
		}
	}
	controlPlane := func(nodeNames ...string) []runtime.Object {
		var objs []runtime.Object
		for _, n := range nodeNames {
			objs = append(objs, staticPod("kube-controller-manager", n), staticPod("kube-scheduler", n))
		}
		return objs
	}

	tests := []struct {
		name                      string
		objects                   []runtime.Object
		wantKubeControllerManager string
		wantKubeScheduler         string
		wantSnapshotController    string
		wantLeaseGets             int
		onlyControllers           bool
		snapshotController        bool
	}{
		{
			name:                      "single control plane does not read leases",
			objects:                   controlPlane("cp-1"),
			wantKubeControllerManager: "kube-controller-manager-cp-1",
			wantKubeScheduler:         "kube-scheduler-cp-1",
			wantLeaseGets:             0,
		},
		{
			name:                      "no leases falls back to the first pod",
			objects:                   controlPlane("cp-1", "cp-2", "cp-3"),
			wantKubeControllerManager: "kube-controller-manager-cp-1",
			wantKubeScheduler:         "kube-scheduler-cp-1",
			wantLeaseGets:             2,
		},
		{
			name: "leaders are not the first pods",
			objects: append(controlPlane("cp-1", "cp-2", "cp-3"),
				lease("kube-controller-manager", "cp-3_5d0c4a5e-0f0c-4c5b-9a43-0d5c2b3e6f11"),
				lease("kube-scheduler", "cp-2_7b6f1d2a-3c4e-4f5a-8b9c-1d2e3f4a5b6c"),
			),
			wantKubeControllerManager: "kube-controller-manager-cp-3",
			wantKubeScheduler:         "kube-scheduler-cp-2",
			wantLeaseGets:             2,
		},
		{
			name: "scheduler metrics not requested keeps the first scheduler pod",
			objects: append(controlPlane("cp-1", "cp-2", "cp-3"),
				lease("kube-controller-manager", "cp-3_5d0c4a5e-0f0c-4c5b-9a43-0d5c2b3e6f11"),
				lease("kube-scheduler", "cp-2_7b6f1d2a-3c4e-4f5a-8b9c-1d2e3f4a5b6c"),
			),
			onlyControllers:           true,
			wantKubeControllerManager: "kube-controller-manager-cp-3",
			wantKubeScheduler:         "kube-scheduler-cp-1",
			wantLeaseGets:             1,
		},
		{
			name: "holder does not match any pod falls back to the first pod",
			objects: append(controlPlane("cp-1", "cp-2"),
				lease("kube-controller-manager", "other-host_5d0c4a5e-0f0c-4c5b-9a43-0d5c2b3e6f11"),
			),
			wantKubeControllerManager: "kube-controller-manager-cp-1",
			wantKubeScheduler:         "kube-scheduler-cp-1",
			wantLeaseGets:             2,
		},
		{
			name: "snapshot controller leader is matched by pod name",
			objects: []runtime.Object{
				snapshotPod("volume-snapshot-controller-0", "node-1"),
				snapshotPod("volume-snapshot-controller-1", "node-2"),
				lease("snapshot-controller-leader", "volume-snapshot-controller-1"),
			},
			snapshotController:     true,
			wantSnapshotController: "volume-snapshot-controller-1",
			wantLeaseGets:          1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := fake.NewClientset(tt.objects...)
			g, err := NewMetricsGrabber(context.Background(), c, &rest.Config{}, false, !tt.onlyControllers, true, false, false, tt.snapshotController)
			if err != nil {
				t.Fatalf("NewMetricsGrabber: %v", err)
			}
			if g.kubeControllerManager != tt.wantKubeControllerManager {
				t.Errorf("kube-controller-manager pod: got %q, want %q", g.kubeControllerManager, tt.wantKubeControllerManager)
			}
			if g.kubeScheduler != tt.wantKubeScheduler {
				t.Errorf("kube-scheduler pod: got %q, want %q", g.kubeScheduler, tt.wantKubeScheduler)
			}
			if g.snapshotController != tt.wantSnapshotController {
				t.Errorf("snapshot controller pod: got %q, want %q", g.snapshotController, tt.wantSnapshotController)
			}
			var leaseGets int
			for _, action := range c.Actions() {
				if action.Matches("get", "leases") {
					leaseGets++
				}
			}
			if leaseGets != tt.wantLeaseGets {
				t.Errorf("Lease reads: got %d, want %d", leaseGets, tt.wantLeaseGets)
			}
		})
	}
}
