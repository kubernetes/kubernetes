/*
Copyright 2023 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/pointer"
)

func TestCompare(t *testing.T) {
	cases := []struct {
		name           string
		lhs            *v1.Lease
		rhs            *v1.Lease
		expectedResult int
	}{
		{
			name: "identical versions",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			expectedResult: 0,
		},
		{
			name: "no lhs version",
			lhs:  &v1.Lease{},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			expectedResult: -1,
		},
		{
			name: "no rhs version",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			rhs:            &v1.Lease{},
			expectedResult: 1,
		},
		{
			name: "invalid lhs version",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "xyz",
						BinaryVersionAnnotationName:        "xyz",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			expectedResult: -1,
		},
		{
			name: "invalid rhs version",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "xyz",
						BinaryVersionAnnotationName:        "xyz",
					},
				},
			},
			expectedResult: 1,
		},
		{
			name: "lhs less than rhs",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			expectedResult: -1,
		},
		{
			name: "rhs less than lhs",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			expectedResult: 1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := compare(tc.lhs, tc.rhs)
			if result != tc.expectedResult {
				t.Errorf("Expected comparison result of %d but got %d", tc.expectedResult, result)
			}
		})
	}
}

func TestPickLeader(t *testing.T) {
	cases := []struct {
		name               string
		candidates         []*v1.Lease
		expectedLeaderName string
		expectNoLeader     bool
	}{
		{
			name: "same compatibility version, newer binary version",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
			},
			expectedLeaderName: "component-identity-1",
		},
		{
			name: "same binary version, newer compatibility version",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.20",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
			},
			expectedLeaderName: "component-identity-2",
		},
		{
			name: "one candidate",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
			},
			expectedLeaderName: "component-identity-1",
		},
		{
			name:           "no candidates",
			candidates:     []*v1.Lease{},
			expectNoLeader: true,
		},
		// TODO: Add test cases where candidates have invalid version numbers
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			leader := pickLeader(tc.candidates)
			if tc.expectNoLeader == true {
				if leader != nil {
					t.Errorf("Expected no leader but got %s", leader.Name)
				}
			} else {
				if leader == nil {
					t.Errorf("Expected leader %s, but got nil leader response", tc.expectedLeaderName)
				} else if leader.Name != tc.expectedLeaderName {
					t.Errorf("Expected leader to be %s but got %s", tc.expectedLeaderName, leader.Name)
				}
			}
		})
	}
}

func TestShouldReelect(t *testing.T) {
	cases := []struct {
		name          string
		candidates    []*v1.Lease
		currentLeader *v1.Lease
		expectResult  bool
	}{
		{
			name: "candidate with newer binary version",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
			},
			currentLeader: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.19",
					},
				},
			},
			expectResult: true,
		},
		{
			name: "no newer candidates",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
			},
			currentLeader: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.19",
					},
				},
			},
			expectResult: false,
		},
		{
			name:       "no candidates",
			candidates: []*v1.Lease{},
			currentLeader: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.19",
					},
				},
			},
			expectResult: false,
		},
		// TODO: Add test cases where candidates have invalid version numbers
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := shouldReelect(tc.candidates, tc.currentLeader)
			if tc.expectResult != result {
				t.Errorf("Expected %t but got %t", tc.expectResult, result)
			}
		})
	}
}

func TestController(t *testing.T) {
	cases := []struct {
		name                       string
		leaderLeaseId              leaderLeaseId
		createAfterControllerStart []*v1.Lease
		deleteAfterControllerStart []leaderLeaseId
		expectedLeaderLeases       []*v1.Lease
	}{
		{
			name:          "single candidate leader election",
			leaderLeaseId: leaderLeaseId{namespace: "kube-system", name: "component-A"},
			createAfterControllerStart: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
						Annotations: map[string]string{
							ElectedByAnnotationName: controllerName,
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
			},
		},
		{
			name:          "multiple candidate leader election",
			leaderLeaseId: leaderLeaseId{namespace: "kube-system", name: "component-A"},
			createAfterControllerStart: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-2"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-3",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.20",
							BinaryVersionAnnotationName:        "1.20",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-3"),
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
						Annotations: map[string]string{
							ElectedByAnnotationName: controllerName,
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-3"),
					},
				},
			},
		},
		{
			name:          "deletion of lease triggers reelection",
			leaderLeaseId: leaderLeaseId{namespace: "kube-system", name: "component-A"},
			createAfterControllerStart: []*v1.Lease{
				{
					// Leader lease
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
						Annotations: map[string]string{
							ElectedByAnnotationName: controllerName,
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-9"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
			},
			deleteAfterControllerStart: []leaderLeaseId{
				{namespace: "kube-system", name: "component-A"},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
						Annotations: map[string]string{
							ElectedByAnnotationName: controllerName,
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
			},
		},
		{
			name:          "better candidate triggers reelection",
			leaderLeaseId: leaderLeaseId{namespace: "kube-system", name: "component-A"},
			createAfterControllerStart: []*v1.Lease{
				{
					// Leader lease
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
						Annotations: map[string]string{
							ElectedByAnnotationName: controllerName,
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-1"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.20",
							BinaryVersionAnnotationName:        "1.20",
						},
						Labels: map[string]string{
							CanLeadLeasesLabelName: "kube-system/component-A",
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-2"),
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
						Annotations: map[string]string{
							ElectedByAnnotationName: controllerName,
						},
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: pointer.String("component-identity-2"),
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()

			//client := fake.NewSimpleClientset(tc.createAfterControllerStart...)
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			controller, err := NewController(
				informerFactory.Coordination().V1().Leases(),
				client.CoordinationV1(),
			)
			if err != nil {
				t.Fatal(err)
			}

			go informerFactory.Start(ctx.Done())
			go controller.Run(ctx, 1)

			go func() {
				ticker := time.NewTicker(10 * time.Millisecond)

				// Mock out the removal of end-of-term leases.
				// When controllers are running, they are expected to do this voluntarily
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						for _, expectedLease := range tc.expectedLeaderLeases {
							lease, err := client.CoordinationV1().Leases(expectedLease.Namespace).Get(ctx, expectedLease.Name, metav1.GetOptions{})
							if err == nil {
								if endOfTerm := lease.Annotations[EndOfTermAnnotationName]; endOfTerm == "true" {
									err = client.CoordinationV1().Leases(expectedLease.Namespace).Delete(ctx, expectedLease.Name, metav1.DeleteOptions{})
									if err != nil {
										runtime.HandleError(err)
									}
								}
							}
						}
					}
				}
			}()

			for _, obj := range tc.createAfterControllerStart {
				_, err := client.CoordinationV1().Leases(obj.Namespace).Create(ctx, obj, metav1.CreateOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			for _, obj := range tc.deleteAfterControllerStart {
				err := client.CoordinationV1().Leases(obj.namespace).Delete(ctx, obj.name, metav1.DeleteOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			for _, expectedLease := range tc.expectedLeaderLeases {
				var lease *v1.Lease
				err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 600*time.Second, true, func(ctx context.Context) (done bool, err error) {
					lease, err = client.CoordinationV1().Leases(expectedLease.Namespace).Get(ctx, expectedLease.Name, metav1.GetOptions{})
					if err != nil {
						if errors.IsNotFound(err) {
							return false, nil
						}
						return true, err
					}
					if expectedLease.Spec.HolderIdentity == nil || lease.Spec.HolderIdentity == nil {
						return expectedLease.Spec.HolderIdentity == nil && lease.Spec.HolderIdentity == nil, nil
					}
					if expectedLease.Spec.HolderIdentity != nil && lease.Spec.HolderIdentity != nil && *expectedLease.Spec.HolderIdentity != *lease.Spec.HolderIdentity {
						return false, nil
					}
					return true, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if lease.Spec.HolderIdentity == nil {
					t.Fatalf("Expected HolderIdentity of %s but got nil", expectedLease.Name)
				}
				if *lease.Spec.HolderIdentity != *expectedLease.Spec.HolderIdentity {
					t.Errorf("Expected HolderIdentity of %s but got %s", *expectedLease.Spec.HolderIdentity, *lease.Spec.HolderIdentity)
				}
			}
		})
	}
}
