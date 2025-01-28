/*
Copyright 2024 The Kubernetes Authors.

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

package leaderelection

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/coordination/v1"
	v1alpha2 "k8s.io/api/coordination/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"
)

func TestLeaseCandidateGCController(t *testing.T) {
	tests := []struct {
		name                 string
		leaseCandidates      []*v1alpha2.LeaseCandidate
		expectedDeletedCount int
	}{
		{
			name: "delete expired lease candidates",
			leaseCandidates: []*v1alpha2.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * leaseCandidateValidDuration)},
					},
					Spec: v1alpha2.LeaseCandidateSpec{
						LeaseName:        "component-A",
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						RenewTime:        ptr.To(metav1.NewMicroTime(time.Now().Add(-1 * leaseCandidateValidDuration))),
						Strategy:         v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate2",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * leaseCandidateValidDuration)},
					},
					Spec: v1alpha2.LeaseCandidateSpec{
						LeaseName:        "component-B",
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						RenewTime:        ptr.To(metav1.NewMicroTime(time.Now().Add(-1 * leaseCandidateValidDuration))),
						Strategy:         v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate3",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1alpha2.LeaseCandidateSpec{
						LeaseName:        "component-C",
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						RenewTime:        ptr.To(metav1.NewMicroTime(time.Now())),
						Strategy:         v1.OldestEmulationVersion,
					},
				},
			},
			expectedDeletedCount: 2,
		},
		{
			name: "no expired lease candidates",
			leaseCandidates: []*v1alpha2.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1alpha2.LeaseCandidateSpec{
						LeaseName:        "component-A",
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						RenewTime:        ptr.To(metav1.NewMicroTime(time.Now())),
						Strategy:         v1.OldestEmulationVersion,
					},
				},
			},
			expectedDeletedCount: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			leaseCandidateInformer := informerFactory.Coordination().V1alpha2().LeaseCandidates()
			controller := NewLeaseCandidateGC(client, 10*time.Millisecond, leaseCandidateInformer)

			// Create lease candidates
			for _, lc := range tc.leaseCandidates {
				_, err := client.CoordinationV1alpha2().LeaseCandidates(lc.Namespace).Create(ctx, lc, metav1.CreateOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			go controller.Run(ctx)
			err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 600*time.Second, true, func(ctx context.Context) (done bool, err error) {
				lcs, err := client.CoordinationV1alpha2().LeaseCandidates("default").List(ctx, metav1.ListOptions{})
				if err != nil {
					return true, err
				}
				return len(lcs.Items) == len(tc.leaseCandidates)-tc.expectedDeletedCount, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
