/*
Copyright 2019 The Kubernetes Authors.

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

package volumebinding

import (
	"context"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/utils/pointer"
)

var (
	immediate            = storagev1.VolumeBindingImmediate
	waitForFirstConsumer = storagev1.VolumeBindingWaitForFirstConsumer
	immediateSC          = &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "immediate-sc",
		},
		VolumeBindingMode: &immediate,
	}
	waitSC = &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wait-sc",
		},
		VolumeBindingMode: &waitForFirstConsumer,
	}
)

func makePV(name string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
}

func addPVNodeAffinity(pv *v1.PersistentVolume, volumeNodeAffinity *v1.VolumeNodeAffinity) *v1.PersistentVolume {
	pv.Spec.NodeAffinity = volumeNodeAffinity
	return pv
}

func makePVC(name string, boundPVName string, storageClassName string) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: pointer.StringPtr(storageClassName),
		},
	}
	if boundPVName != "" {
		pvc.Spec.VolumeName = boundPVName
		metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnBindCompleted, "true")
	}
	return pvc
}

func makePod(name string, pvcNames []string) *v1.Pod {
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
		},
	}
	p.Spec.Volumes = make([]v1.Volume, 0)
	for _, pvcName := range pvcNames {
		p.Spec.Volumes = append(p.Spec.Volumes, v1.Volume{
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcName,
				},
			},
		})
	}
	return p
}

func TestVolumeBinding(t *testing.T) {
	table := []struct {
		name                    string
		pod                     *v1.Pod
		node                    *v1.Node
		pvcs                    []*v1.PersistentVolumeClaim
		pvs                     []*v1.PersistentVolume
		wantPreFilterStatus     *framework.Status
		wantStateAfterPreFilter *stateData
		wantFilterStatus        *framework.Status
	}{
		{
			name: "pod has not pvcs",
			pod:  makePod("pod-a", nil),
			node: &v1.Node{},
			wantStateAfterPreFilter: &stateData{
				skip: true,
			},
		},
		{
			name: "all bound",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			node: &v1.Node{},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
			},
			pvs: []*v1.PersistentVolume{
				makePV("pv-a"),
			},
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "pv-a", waitSC.Name),
				},
				claimsToBind:     []*v1.PersistentVolumeClaim{},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
		},
		{
			name: "immediate claims not bound",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			node: &v1.Node{},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "", immediateSC.Name),
			},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, "pod has unbound immediate PersistentVolumeClaims"),
		},
		{
			name: "unbound claims no matches",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			node: &v1.Node{},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "", waitSC.Name),
			},
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{},
				claimsToBind: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "", waitSC.Name),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, string(scheduling.ErrReasonBindConflict)),
		},
		{
			name: "bound and unbound unsatisfied",
			pod:  makePod("pod-a", []string{"pvc-a", "pvc-b"}),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "barbar",
					},
				},
			},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
				makePVC("pvc-b", "", waitSC.Name),
			},
			pvs: []*v1.PersistentVolume{
				addPVNodeAffinity(makePV("pv-a"), &v1.VolumeNodeAffinity{
					Required: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "foo",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"bar"},
									},
								},
							},
						},
					},
				}),
			},
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "pv-a", waitSC.Name),
				},
				claimsToBind: []*v1.PersistentVolumeClaim{
					makePVC("pvc-b", "", waitSC.Name),
				},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, string(scheduling.ErrReasonNodeConflict), string(scheduling.ErrReasonBindConflict)),
		},
		{
			name:                "pvc not found",
			pod:                 makePod("pod-a", []string{"pvc-a"}),
			node:                &v1.Node{},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "pvc-a" not found`),
			wantFilterStatus:    nil,
		},
		{
			name: "pv not found",
			pod:  makePod("pod-a", []string{"pvc-a"}),
			node: &v1.Node{},
			pvcs: []*v1.PersistentVolumeClaim{
				makePVC("pvc-a", "pv-a", waitSC.Name),
			},
			wantPreFilterStatus: nil,
			wantStateAfterPreFilter: &stateData{
				boundClaims: []*v1.PersistentVolumeClaim{
					makePVC("pvc-a", "pv-a", waitSC.Name),
				},
				claimsToBind:     []*v1.PersistentVolumeClaim{},
				podVolumesByNode: map[string]*scheduling.PodVolumes{},
			},
			wantFilterStatus: framework.NewStatus(framework.Error, `could not find v1.PersistentVolume "pv-a"`),
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			opts := []runtime.Option{
				runtime.WithClientSet(client),
				runtime.WithInformerFactory(informerFactory),
			}
			fh, err := runtime.NewFramework(nil, nil, nil, opts...)
			if err != nil {
				t.Fatal(err)
			}
			pl, err := New(&config.VolumeBindingArgs{
				BindTimeoutSeconds: 300,
			}, fh)
			if err != nil {
				t.Fatal(err)
			}

			t.Log("Feed testing data and wait for them to be synced")
			client.StorageV1().StorageClasses().Create(ctx, immediateSC, metav1.CreateOptions{})
			client.StorageV1().StorageClasses().Create(ctx, waitSC, metav1.CreateOptions{})
			if item.node != nil {
				client.CoreV1().Nodes().Create(ctx, item.node, metav1.CreateOptions{})
			}
			for _, pvc := range item.pvcs {
				client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, pvc, metav1.CreateOptions{})
			}
			for _, pv := range item.pvs {
				client.CoreV1().PersistentVolumes().Create(ctx, pv, metav1.CreateOptions{})
			}

			t.Log("Start informer factory after initialization")
			informerFactory.Start(ctx.Done())

			t.Log("Wait for all started informers' cache were synced")
			informerFactory.WaitForCacheSync(ctx.Done())

			t.Log("Verify")

			p := pl.(*VolumeBinding)
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(item.node)
			state := framework.NewCycleState()

			t.Logf("Verify: call PreFilter and check status")
			gotPreFilterStatus := p.PreFilter(ctx, state, item.pod)
			if !reflect.DeepEqual(gotPreFilterStatus, item.wantPreFilterStatus) {
				t.Errorf("filter prefilter status does not match: %v, want: %v", gotPreFilterStatus, item.wantPreFilterStatus)
			}
			if !gotPreFilterStatus.IsSuccess() {
				// scheduler framework will skip Filter if PreFilter fails
				return
			}

			t.Logf("Verify: check state after prefilter phase")
			stateData, err := getStateData(state)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(stateData, item.wantStateAfterPreFilter) {
				t.Errorf("state got after prefilter does not match: %v, want: %v", stateData, item.wantStateAfterPreFilter)
			}

			t.Logf("Verify: call Filter and check status")
			gotStatus := p.Filter(ctx, state, item.pod, nodeInfo)
			if !reflect.DeepEqual(gotStatus, item.wantFilterStatus) {
				t.Errorf("filter status does not match: %v, want: %v", gotStatus, item.wantFilterStatus)
			}
		})
	}
}
