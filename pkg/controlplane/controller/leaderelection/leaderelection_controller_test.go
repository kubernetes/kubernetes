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
	"fmt"
	"math/rand"
	"testing"
	"time"

	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestReconcileElectionStep(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())

	tests := []struct {
		name                    string
		leaseNN                 types.NamespacedName
		candidates              []*v1alpha1.LeaseCandidate
		existingLease           *v1.Lease
		expectLease             bool
		expectedHolderIdentity  *string
		expectedPreferredHolder *string
		expectedRequeue         bool
		expectedError           bool
		expectedStrategy        *v1.CoordinatedLeaseStrategy
		candidatesPinged        bool
	}{
		{
			name:                   "no candidates, no lease, noop",
			leaseNN:                types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates:             []*v1alpha1.LeaseCandidate{},
			existingLease:          nil,
			expectLease:            false,
			expectedHolderIdentity: nil,
			expectedStrategy:       nil,
			expectedRequeue:        false,
			expectedError:          false,
		},
		{
			name:                   "no candidates, lease exists. noop, not managed by CLE",
			leaseNN:                types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates:             []*v1alpha1.LeaseCandidate{},
			existingLease:          &v1.Lease{},
			expectLease:            false,
			expectedHolderIdentity: nil,
			expectedStrategy:       nil,
			expectedRequeue:        false,
			expectedError:          false,
		},
		{
			name:    "candidates exist, no existing lease should create lease",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease:          nil,
			expectLease:            true,
			expectedHolderIdentity: ptr.To("component-identity-1"),
			expectedStrategy:       ptr.To[v1.CoordinatedLeaseStrategy]("OldestEmulationVersion"),
			expectedRequeue:        true,
			expectedError:          false,
		},
		{
			name:    "candidates exist, lease exists, unoptimal should set preferredHolder",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-2",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.18.0",
						BinaryVersion:       "1.18.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "component-A",
				},
				Spec: v1.LeaseSpec{
					HolderIdentity:       ptr.To("component-identity-1"),
					LeaseDurationSeconds: ptr.To(int32(10)),
					RenewTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now())),
				},
			},
			expectLease:             true,
			expectedHolderIdentity:  ptr.To("component-identity-1"),
			expectedPreferredHolder: ptr.To("component-identity-2"),
			expectedStrategy:        ptr.To[v1.CoordinatedLeaseStrategy]("OldestEmulationVersion"),
			expectedRequeue:         true,
			expectedError:           false,
		},
		{
			name:    "candidates exist, should only elect leader from acked candidates",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						PingTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-2 * electionDuration))),
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-4 * electionDuration))),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-2",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.20.0",
						BinaryVersion:       "1.20.0",
						PingTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-1 * time.Millisecond))),
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease:          nil,
			expectLease:            true,
			expectedHolderIdentity: ptr.To("component-identity-2"),
			expectedStrategy:       ptr.To[v1.CoordinatedLeaseStrategy]("OldestEmulationVersion"),
			expectedRequeue:        true,
			expectedError:          false,
		},
		{
			name:    "candidates exist, lease exists, lease expired",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "component-A",
				},
				Spec: v1.LeaseSpec{
					HolderIdentity:       ptr.To("component-identity-expired"),
					LeaseDurationSeconds: ptr.To(int32(10)),
					RenewTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-1 * time.Minute))),
				},
			},
			expectLease:            true,
			expectedHolderIdentity: ptr.To("component-identity-1"),
			expectedStrategy:       ptr.To[v1.CoordinatedLeaseStrategy]("OldestEmulationVersion"),
			expectedRequeue:        true,
			expectedError:          false,
		},
		{
			name:    "candidates exist, no acked candidates should return error",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						PingTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-1 * time.Minute))),
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-2 * time.Minute))),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease:          nil,
			expectLease:            false,
			expectedHolderIdentity: nil,
			expectedRequeue:        false,
			expectedError:          true,
		},
		{
			name:    "candidates exist, should ping on election",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-2 * electionDuration))),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease:          nil,
			expectLease:            false,
			expectedHolderIdentity: nil,
			expectedStrategy:       nil,
			expectedRequeue:        true,
			expectedError:          false,
			candidatesPinged:       true,
		},
		{
			name:    "candidate exist, pinged candidate should have until electionDuration until election decision is made",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						PingTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-1 * time.Minute))),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			existingLease:          nil,
			expectLease:            false,
			expectedHolderIdentity: nil,
			expectedRequeue:        true,
			expectedError:          false,
		},
		{
			name:    "candidates exist, lease exists, lease expired, 3rdparty strategy",
			leaseNN: types.NamespacedName{Namespace: "default", Name: "component-A"},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{"foo.com/bar"},
					},
				},
			},
			existingLease: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "component-A",
				},
				Spec: v1.LeaseSpec{
					HolderIdentity:       ptr.To("component-identity-expired"),
					LeaseDurationSeconds: ptr.To(int32(10)),
					RenewTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-1 * time.Minute))),
				},
			},
			expectLease:            true,
			expectedHolderIdentity: ptr.To("component-identity-expired"),
			expectedStrategy:       ptr.To[v1.CoordinatedLeaseStrategy]("foo.com/bar"),
			expectedRequeue:        true,
			expectedError:          false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			controller, err := NewController(
				informerFactory.Coordination().V1().Leases(),
				informerFactory.Coordination().V1alpha1().LeaseCandidates(),
				client.CoordinationV1(),
				client.CoordinationV1alpha1(),
			)
			controller.clock = fakeClock
			if err != nil {
				t.Fatal(err)
			}

			// Set up the fake client with the existing lease
			if tc.existingLease != nil {
				_, err = client.CoordinationV1().Leases(tc.existingLease.Namespace).Create(ctx, tc.existingLease, metav1.CreateOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			// Set up the fake client with the candidates
			for _, candidate := range tc.candidates {
				_, err = client.CoordinationV1alpha1().LeaseCandidates(candidate.Namespace).Create(ctx, candidate, metav1.CreateOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			requeue, err := controller.reconcileElectionStep(ctx, tc.leaseNN)

			if (requeue != 0) != tc.expectedRequeue {
				t.Errorf("reconcileElectionStep() requeue = %v, want %v", requeue, tc.expectedRequeue)
			}
			if tc.expectedError && err == nil {
				t.Errorf("reconcileElectionStep() error = %v, want error", err)
			} else if !tc.expectedError && err != nil {
				t.Errorf("reconcileElectionStep() error = %v, want nil", err)
			}

			lease, err := client.CoordinationV1().Leases(tc.leaseNN.Namespace).Get(ctx, tc.leaseNN.Name, metav1.GetOptions{})
			if tc.expectLease {
				if err != nil {
					t.Fatal(err)
				}

				// Check the lease holder identity
				if tc.expectedHolderIdentity != nil && (lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != *tc.expectedHolderIdentity) {
					t.Errorf("reconcileElectionStep() holderIdentity = %s, want %s", strOrNil(lease.Spec.HolderIdentity), *tc.expectedHolderIdentity)
				} else if tc.expectedHolderIdentity == nil && lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != "" {
					t.Errorf("reconcileElectionStep() holderIdentity = %s, want nil", *lease.Spec.HolderIdentity)
				}
				if tc.expectedPreferredHolder != nil && (lease.Spec.PreferredHolder == nil || *lease.Spec.PreferredHolder != *tc.expectedPreferredHolder) {
					t.Errorf("reconcileElectionStep() preferredHolder = %s, want %s", strOrNil(lease.Spec.PreferredHolder), *tc.expectedPreferredHolder)
				} else if tc.expectedPreferredHolder == nil && lease.Spec.PreferredHolder != nil && *lease.Spec.PreferredHolder != "" {
					t.Errorf("reconcileElectionStep() preferredHolder = %s, want nil", *lease.Spec.PreferredHolder)
				}

				// Check chosen strategy in the Lease
				if tc.expectedStrategy != nil && (lease.Spec.Strategy == nil || *lease.Spec.Strategy != *tc.expectedStrategy) {
					t.Errorf("reconcileElectionStep() strategy = %s, want %s", strOrNil(lease.Spec.Strategy), *tc.expectedStrategy)
				} else if tc.expectedStrategy == nil && lease.Spec.Strategy != nil && *lease.Spec.Strategy != "" {
					t.Errorf("reconcileElectionStep() strategy = %s, want nil", *lease.Spec.Strategy)
				}
			} else if err == nil {
				t.Errorf("reconcileElectionStep() expected no lease to be created")
			}

			// Verify that ping to candidate was issued
			if tc.candidatesPinged {
				pinged := false
				candidatesList, err := client.CoordinationV1alpha1().LeaseCandidates(tc.leaseNN.Namespace).List(ctx, metav1.ListOptions{})
				if err != nil {
					t.Fatal(err)
				}
				oldCandidateMap := make(map[string]*v1alpha1.LeaseCandidate)
				for _, candidate := range tc.candidates {
					oldCandidateMap[candidate.Name] = candidate
				}
				for _, candidate := range candidatesList.Items {
					if candidate.Spec.PingTime != nil {
						if oldCandidateMap[candidate.Name].Spec.PingTime == nil {
							pinged = true
							break
						}
					}
				}
				if !pinged {
					t.Errorf("reconcileElectionStep() expected candidates to be pinged")
				}
			}

		})
	}
}

func TestController(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())

	cases := []struct {
		name                            string
		leases                          []*v1.Lease
		candidates                      []*v1alpha1.LeaseCandidate
		createAfterControllerStart      []*v1alpha1.LeaseCandidate
		deleteLeaseAfterControllerStart []types.NamespacedName
		expectedLeaderLeases            []*v1.Lease
	}{
		{
			name: "single candidate leader election",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: ptr.To("component-identity-1"),
					},
				},
			},
		},
		{
			name: "multiple candidate leader election",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-2",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.20.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-3",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.20.0",
						BinaryVersion:       "1.20.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: ptr.To("component-identity-1"),
					},
				},
			},
		},
		{
			name: "deletion of lease triggers reelection",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity:       ptr.To("some-other-component"),
						RenewTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(time.Second))),
						LeaseDurationSeconds: ptr.To(int32(10)),
					},
				},
			},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			deleteLeaseAfterControllerStart: []types.NamespacedName{
				{Namespace: "kube-system", Name: "component-A"},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: ptr.To("component-identity-1"),
					},
				},
			},
		},
		{
			name: "expired lease is replaced",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity:       ptr.To("some-other-component"),
						RenewTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(-11 * time.Second))),
						LeaseDurationSeconds: ptr.To(int32(10)),
						Strategy:             ptr.To[v1.CoordinatedLeaseStrategy]("OldestEmulationVersion"),
					},
				},
			},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: ptr.To("component-identity-1"),
					},
				},
			},
		},
		{
			name: "better candidate triggers reelection",
			leases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity:       ptr.To("component-identity-1"),
						Strategy:             ptr.To[v1.CoordinatedLeaseStrategy]("OldestEmulationVersion"),
						RenewTime:            ptr.To(metav1.NewMicroTime(fakeClock.Now().Add(time.Second))),
						LeaseDurationSeconds: ptr.To(int32(10)),
					},
				},
			},
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.20.0",
						BinaryVersion:       "1.20.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			createAfterControllerStart: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-identity-2",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						LeaseName:           "component-A",
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						RenewTime:           ptr.To(metav1.NewMicroTime(fakeClock.Now())),
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			expectedLeaderLeases: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "kube-system",
						Name:      "component-A",
					},
					Spec: v1.LeaseSpec{
						HolderIdentity: ptr.To("component-identity-2"),
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()

			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			controller, err := NewController(
				informerFactory.Coordination().V1().Leases(),
				informerFactory.Coordination().V1alpha1().LeaseCandidates(),
				client.CoordinationV1(),
				client.CoordinationV1alpha1(),
			)
			if err != nil {
				t.Fatal(err)
			}

			for _, obj := range tc.leases {
				t.Logf("Pre-creating lease %s/%s", obj.Namespace, obj.Name)
				_, err := client.CoordinationV1().Leases(obj.Namespace).Create(ctx, obj, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error pre-creating lease %s/%s: %v", obj.Namespace, obj.Name, err)
				}
			}
			for _, obj := range tc.candidates {
				t.Logf("Pre-creating lease candidate %s/%s", obj.Namespace, obj.Name)
				_, err := client.CoordinationV1alpha1().LeaseCandidates(obj.Namespace).Create(ctx, obj, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error pre-creating lease candidate %s/%s: %v", obj.Namespace, obj.Name, err)
				}
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			go controller.Run(ctx, 1)

			if rand.Intn(2) == 0 {
				t.Logf("Giving controller a chance to run")
				time.Sleep(time.Second)
			}

			go func() {
				ticker := time.NewTicker(10 * time.Millisecond)
				// Mock out the removal of preferredHolder leases.
				// When controllers are running, they are expected to do this voluntarily
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						for _, expectedLease := range tc.expectedLeaderLeases {
							l, err := client.CoordinationV1().Leases(expectedLease.Namespace).Get(ctx, expectedLease.Name, metav1.GetOptions{})
							if err != nil {
								continue
							}
							ph := l.Spec.PreferredHolder
							if ph == nil || l.Spec.HolderIdentity == nil {
								continue
							}
							if *ph == *l.Spec.HolderIdentity {
								continue
							}
							if _, err := client.CoordinationV1alpha1().LeaseCandidates(expectedLease.Namespace).Get(ctx, *l.Spec.HolderIdentity, metav1.GetOptions{}); err != nil {
								continue // only candidate-aware controllers will follow preferredHolder
							}

							t.Logf("Deleting lease %s/%s because of preferredHolder %q != %q", l.Namespace, l.Name, *ph, *l.Spec.HolderIdentity)
							if err = client.CoordinationV1().Leases(expectedLease.Namespace).Delete(ctx, expectedLease.Name, metav1.DeleteOptions{}); err != nil {
								t.Logf("Error deleting lease %s/%s: %v", l.Namespace, l.Name, err)
							}
						}
					}
				}
			}()

			go func() {
				ticker := time.NewTicker(10 * time.Millisecond)
				// Mock out leasecandidate ack.
				// When controllers are running, they are expected to watch and ack
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						for _, lc := range tc.createAfterControllerStart {
							c, err := client.CoordinationV1alpha1().LeaseCandidates(lc.Namespace).Get(ctx, lc.Name, metav1.GetOptions{})
							if err == nil {
								if c.Spec.PingTime != nil {
									t.Logf("Answering ping for %s/%s", c.Namespace, c.Name)
									c.Spec.RenewTime = &metav1.MicroTime{Time: fakeClock.Now()}
									_, err = client.CoordinationV1alpha1().LeaseCandidates(lc.Namespace).Update(ctx, c, metav1.UpdateOptions{})
									if err != nil {
										t.Logf("Error updating lease candidate %s/%s: %v", c.Namespace, c.Name, err)
									}
								}
							}
						}
					}
				}
			}()

			for _, obj := range tc.createAfterControllerStart {
				t.Logf("Post-creating lease candidate %s/%s", obj.Namespace, obj.Name)
				_, err := client.CoordinationV1alpha1().LeaseCandidates(obj.Namespace).Create(ctx, obj, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error post-creating lease candidate %s/%s: %v", obj.Namespace, obj.Name, err)
				}
			}

			for _, obj := range tc.deleteLeaseAfterControllerStart {
				t.Logf("Post-deleting lease %s/%s", obj.Namespace, obj.Name)
				err := client.CoordinationV1().Leases(obj.Namespace).Delete(ctx, obj.Name, metav1.DeleteOptions{})
				if err != nil {
					t.Fatalf("Error post-deleting lease %s/%s: %v", obj.Namespace, obj.Name, err)
				}
			}

			for _, expectedLease := range tc.expectedLeaderLeases {
				var lease *v1.Lease
				t.Logf("Waiting for lease %s/%s with holder %q", expectedLease.Namespace, expectedLease.Name, strOrNil(expectedLease.Spec.HolderIdentity))
				err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (done bool, err error) {
					lease, err = client.CoordinationV1().Leases(expectedLease.Namespace).Get(ctx, expectedLease.Name, metav1.GetOptions{})
					if err != nil {
						if errors.IsNotFound(err) {
							return false, nil
						}
						return true, err
					}
					if expectedLease.Spec.HolderIdentity == nil {
						return lease.Spec.HolderIdentity == nil, nil
					}
					if lease.Spec.HolderIdentity == nil {
						return false, nil
					}
					if *expectedLease.Spec.HolderIdentity != *lease.Spec.HolderIdentity {
						return false, nil
					}
					return true, nil
				})
				if err != nil {
					if lease == nil {
						t.Fatalf("Error waiting for lease %s/%s to exist: %v", expectedLease.Namespace, expectedLease.Name, err)
					}
					t.Fatalf("Error waiting for lease %s/%s with holder %q, but got %q: %v", expectedLease.Namespace, expectedLease.Name, strOrNil(expectedLease.Spec.HolderIdentity), strOrNil(lease.Spec.HolderIdentity), err)
				}
			}
		})
	}
}

func strOrNil[T any](s *T) string {
	if s == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%v", *s)
}
