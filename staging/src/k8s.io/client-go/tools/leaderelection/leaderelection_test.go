/*
Copyright 2015 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/fake"
	fakeclient "k8s.io/client-go/testing"
	rl "k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
)

func createLockObject(objectType, namespace, name string, record rl.LeaderElectionRecord) (obj runtime.Object) {
	objectMeta := metav1.ObjectMeta{
		Namespace: namespace,
		Name:      name,
	}
	switch objectType {
	case "endpoints":
		recordBytes, _ := json.Marshal(record)
		objectMeta.Annotations = map[string]string{
			rl.LeaderElectionRecordAnnotationKey: string(recordBytes),
		}
		obj = &corev1.Endpoints{ObjectMeta: objectMeta}
	case "configmaps":
		recordBytes, _ := json.Marshal(record)
		objectMeta.Annotations = map[string]string{
			rl.LeaderElectionRecordAnnotationKey: string(recordBytes),
		}
		obj = &corev1.ConfigMap{ObjectMeta: objectMeta}
	case "leases":
		spec := rl.LeaderElectionRecordToLeaseSpec(&record)
		obj = &coordinationv1.Lease{ObjectMeta: objectMeta, Spec: spec}
	default:
		panic("unexpected objType:" + objectType)
	}
	return
}

// Will test leader election using endpoints as the resource
func TestTryAcquireOrRenewEndpoints(t *testing.T) {
	testTryAcquireOrRenew(t, "endpoints")
}

func testTryAcquireOrRenew(t *testing.T, objectType string) {
	future := time.Now().Add(1000 * time.Hour)
	past := time.Now().Add(-1000 * time.Hour)

	tests := []struct {
		name           string
		observedRecord rl.LeaderElectionRecord
		observedTime   time.Time
		reactors       []struct {
			verb     string
			reaction fakeclient.ReactionFunc
		}

		expectSuccess    bool
		transitionLeader bool
		outHolder        string
	}{
		{
			name: "acquire from no object",
			reactors: []struct {
				verb     string
				reaction fakeclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
				{
					verb: "create",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},
			expectSuccess: true,
			outHolder:     "baz",
		},
		{
			name: "acquire from unled object",
			reactors: []struct {
				verb     string
				reaction fakeclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), rl.LeaderElectionRecord{}), nil
					},
				},
				{
					verb: "update",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "acquire from led, unacked object",
			reactors: []struct {
				verb     string
				reaction fakeclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb: "update",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},
			observedRecord: rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedTime:   past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "acquire from empty led, acked object",
			reactors: []struct {
				verb     string
				reaction fakeclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), rl.LeaderElectionRecord{HolderIdentity: ""}), nil
					},
				},
				{
					verb: "update",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},
			observedTime: future,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "don't acquire from led, acked object",
			reactors: []struct {
				verb     string
				reaction fakeclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
			},
			observedTime: future,

			expectSuccess: false,
			outHolder:     "bing",
		},
		{
			name: "renew already acquired object",
			reactors: []struct {
				verb     string
				reaction fakeclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
					},
				},
				{
					verb: "update",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},
			observedTime:   future,
			observedRecord: rl.LeaderElectionRecord{HolderIdentity: "baz"},

			expectSuccess: true,
			outHolder:     "baz",
		},
	}

	for i := range tests {
		test := &tests[i]
		t.Run(test.name, func(t *testing.T) {
			// OnNewLeader is called async so we have to wait for it.
			var wg sync.WaitGroup
			wg.Add(1)
			var reportedLeader string
			var lock rl.Interface

			objectMeta := metav1.ObjectMeta{Namespace: "foo", Name: "bar"}
			resourceLockConfig := rl.ResourceLockConfig{
				Identity:      "baz",
				EventRecorder: &record.FakeRecorder{},
			}
			c := &fake.Clientset{}
			for _, reactor := range test.reactors {
				c.AddReactor(reactor.verb, objectType, reactor.reaction)
			}
			c.AddReactor("*", "*", func(action fakeclient.Action) (bool, runtime.Object, error) {
				t.Errorf("unreachable action. testclient called too many times: %+v", action)
				return true, nil, fmt.Errorf("unreachable action")
			})

			switch objectType {
			case "endpoints":
				lock = &rl.EndpointsLock{
					EndpointsMeta: objectMeta,
					LockConfig:    resourceLockConfig,
					Client:        c.CoreV1(),
				}
			case "configmaps":
				lock = &rl.ConfigMapLock{
					ConfigMapMeta: objectMeta,
					LockConfig:    resourceLockConfig,
					Client:        c.CoreV1(),
				}
			case "leases":
				lock = &rl.LeaseLock{
					LeaseMeta:  objectMeta,
					LockConfig: resourceLockConfig,
					Client:     c.CoordinationV1(),
				}
			}

			lec := LeaderElectionConfig{
				Lock:          lock,
				LeaseDuration: 10 * time.Second,
				Callbacks: LeaderCallbacks{
					OnNewLeader: func(l string) {
						defer wg.Done()
						reportedLeader = l
					},
				},
			}
			le := &LeaderElector{
				config:         lec,
				observedRecord: test.observedRecord,
				observedTime:   test.observedTime,
				clock:          clock.RealClock{},
			}

			if test.expectSuccess != le.tryAcquireOrRenew() {
				t.Errorf("unexpected result of tryAcquireOrRenew: [succeeded=%v]", !test.expectSuccess)
			}

			le.observedRecord.AcquireTime = metav1.Time{}
			le.observedRecord.RenewTime = metav1.Time{}
			if le.observedRecord.HolderIdentity != test.outHolder {
				t.Errorf("expected holder:\n\t%+v\ngot:\n\t%+v", test.outHolder, le.observedRecord.HolderIdentity)
			}
			if len(test.reactors) != len(c.Actions()) {
				t.Errorf("wrong number of api interactions")
			}
			if test.transitionLeader && le.observedRecord.LeaderTransitions != 1 {
				t.Errorf("leader should have transitioned but did not")
			}
			if !test.transitionLeader && le.observedRecord.LeaderTransitions != 0 {
				t.Errorf("leader should not have transitioned but did")
			}

			le.maybeReportTransition()
			wg.Wait()
			if reportedLeader != test.outHolder {
				t.Errorf("reported leader was not the new leader. expected %q, got %q", test.outHolder, reportedLeader)
			}
		})
	}
}

// Will test leader election using configmap as the resource
func TestTryAcquireOrRenewConfigMaps(t *testing.T) {
	testTryAcquireOrRenew(t, "configmaps")
}

// Will test leader election using lease as the resource
func TestTryAcquireOrRenewLeases(t *testing.T) {
	testTryAcquireOrRenew(t, "leases")
}

func TestLeaseSpecToLeaderElectionRecordRoundTrip(t *testing.T) {
	holderIdentity := "foo"
	leaseDurationSeconds := int32(10)
	leaseTransitions := int32(1)
	oldSpec := coordinationv1.LeaseSpec{
		HolderIdentity:       &holderIdentity,
		LeaseDurationSeconds: &leaseDurationSeconds,
		AcquireTime:          &metav1.MicroTime{time.Now()},
		RenewTime:            &metav1.MicroTime{time.Now()},
		LeaseTransitions:     &leaseTransitions,
	}

	oldRecord := rl.LeaseSpecToLeaderElectionRecord(&oldSpec)
	newSpec := rl.LeaderElectionRecordToLeaseSpec(oldRecord)

	if !equality.Semantic.DeepEqual(oldSpec, newSpec) {
		t.Errorf("diff: %v", diff.ObjectReflectDiff(oldSpec, newSpec))
	}

	newRecord := rl.LeaseSpecToLeaderElectionRecord(&newSpec)

	if !equality.Semantic.DeepEqual(oldRecord, newRecord) {
		t.Errorf("diff: %v", diff.ObjectReflectDiff(oldRecord, newRecord))
	}
}
