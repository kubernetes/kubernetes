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
	"context"
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

func createLockObject(t *testing.T, objectType, namespace, name string, record *rl.LeaderElectionRecord) (obj runtime.Object) {
	objectMeta := metav1.ObjectMeta{
		Namespace: namespace,
		Name:      name,
	}
	if record != nil {
		recordBytes, _ := json.Marshal(record)
		objectMeta.Annotations = map[string]string{
			rl.LeaderElectionRecordAnnotationKey: string(recordBytes),
		}
	}
	switch objectType {
	case "endpoints":
		obj = &corev1.Endpoints{ObjectMeta: objectMeta}
	case "configmaps":
		obj = &corev1.ConfigMap{ObjectMeta: objectMeta}
	case "leases":
		var spec coordinationv1.LeaseSpec
		if record != nil {
			spec = rl.LeaderElectionRecordToLeaseSpec(record)
		}
		obj = &coordinationv1.Lease{ObjectMeta: objectMeta, Spec: spec}
	default:
		t.Fatal("unexpected objType:" + objectType)
	}
	return
}

// Will test leader election using endpoints as the resource
func TestTryAcquireOrRenewEndpoints(t *testing.T) {
	testTryAcquireOrRenew(t, "endpoints")
}

type Reactor struct {
	verb       string
	objectType string
	reaction   fakeclient.ReactionFunc
}

func testTryAcquireOrRenew(t *testing.T, objectType string) {
	future := time.Now().Add(1000 * time.Hour)
	past := time.Now().Add(-1000 * time.Hour)

	tests := []struct {
		name           string
		observedRecord rl.LeaderElectionRecord
		observedTime   time.Time
		reactors       []Reactor

		expectSuccess    bool
		transitionLeader bool
		outHolder        string
	}{
		{
			name: "acquire from no object",
			reactors: []Reactor{
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
			name: "acquire from object without annotations",
			reactors: []Reactor{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), nil), nil
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
			name: "acquire from unled object",
			reactors: []Reactor{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{}), nil
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
			reactors: []Reactor{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
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
			reactors: []Reactor{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: ""}), nil
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
			reactors: []Reactor{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
			},
			observedTime: future,

			expectSuccess: false,
			outHolder:     "bing",
		},
		{
			name: "renew already acquired object",
			reactors: []Reactor{
				{
					verb: "get",
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, objectType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
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
			observedRawRecord := GetRawRecordOrDie(t, objectType, test.observedRecord)
			le := &LeaderElector{
				config:            lec,
				observedRecord:    test.observedRecord,
				observedRawRecord: observedRawRecord,
				observedTime:      test.observedTime,
				clock:             clock.RealClock{},
			}
			if test.expectSuccess != le.tryAcquireOrRenew(context.Background()) {
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

func multiLockType(t *testing.T, objectType string) (primaryType, secondaryType string) {
	switch objectType {
	case rl.EndpointsLeasesResourceLock:
		return rl.EndpointsResourceLock, rl.LeasesResourceLock
	case rl.ConfigMapsLeasesResourceLock:
		return rl.ConfigMapsResourceLock, rl.LeasesResourceLock
	default:
		t.Fatal("unexpected objType:" + objectType)
	}
	return
}

func GetRawRecordOrDie(t *testing.T, objectType string, ler rl.LeaderElectionRecord) (ret []byte) {
	var err error
	switch objectType {
	case "endpoints", "configmaps", "leases":
		ret, err = json.Marshal(ler)
		if err != nil {
			t.Fatalf("lock %s get raw record %v failed: %v", objectType, ler, err)
		}
	case "endpointsleases", "configmapsleases":
		recordBytes, err := json.Marshal(ler)
		if err != nil {
			t.Fatalf("lock %s get raw record %v failed: %v", objectType, ler, err)
		}
		ret = rl.ConcatRawRecord(recordBytes, recordBytes)
	default:
		t.Fatal("unexpected objType:" + objectType)
	}
	return
}

func testTryAcquireOrRenewMultiLock(t *testing.T, objectType string) {
	future := time.Now().Add(1000 * time.Hour)
	past := time.Now().Add(-1000 * time.Hour)
	primaryType, secondaryType := multiLockType(t, objectType)
	tests := []struct {
		name              string
		observedRecord    rl.LeaderElectionRecord
		observedRawRecord []byte
		observedTime      time.Time
		reactors          []Reactor

		expectSuccess    bool
		transitionLeader bool
		outHolder        string
	}{
		{
			name: "acquire from no object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
				{
					verb:       "create",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
				{
					verb:       "create",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},
			expectSuccess: true,
			outHolder:     "baz",
		},
		{
			name: "acquire from unled old object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
				{
					verb:       "update",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
				{
					verb:       "create",
					objectType: secondaryType,
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
			name: "acquire from unled transition object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{}), nil
					},
				},
				{
					verb:       "update",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{}), nil
					},
				},
				{
					verb:       "update",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
			},
			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "acquire from led, unack old object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
				{
					verb:       "update",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "create",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedRawRecord: GetRawRecordOrDie(t, primaryType, rl.LeaderElectionRecord{HolderIdentity: "bing"}),
			observedTime:      past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "acquire from led, unack transition object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "update",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "update",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedRawRecord: GetRawRecordOrDie(t, objectType, rl.LeaderElectionRecord{HolderIdentity: "bing"}),
			observedTime:      past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "acquire from conflict led, ack transition object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedRawRecord: GetRawRecordOrDie(t, objectType, rl.LeaderElectionRecord{HolderIdentity: "bing"}),
			observedTime:      future,

			expectSuccess: false,
			outHolder:     rl.UnknownLeader,
		},
		{
			name: "acquire from led, unack unknown object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: rl.UnknownLeader}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: rl.UnknownLeader}), nil
					},
				},
				{
					verb:       "update",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: rl.UnknownLeader}), nil
					},
				},
				{
					verb:       "update",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: rl.UnknownLeader},
			observedRawRecord: GetRawRecordOrDie(t, objectType, rl.LeaderElectionRecord{HolderIdentity: rl.UnknownLeader}),
			observedTime:      past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		{
			name: "don't acquire from led, ack old object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedRawRecord: GetRawRecordOrDie(t, primaryType, rl.LeaderElectionRecord{HolderIdentity: "bing"}),
			observedTime:      future,

			expectSuccess: false,
			outHolder:     "bing",
		},
		{
			name: "don't acquire from led, acked new object, observe new record",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedRawRecord: GetRawRecordOrDie(t, secondaryType, rl.LeaderElectionRecord{HolderIdentity: "bing"}),
			observedTime:      future,

			expectSuccess: false,
			outHolder:     rl.UnknownLeader,
		},
		{
			name: "don't acquire from led, acked new object, observe transition record",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "bing"}), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedRawRecord: GetRawRecordOrDie(t, objectType, rl.LeaderElectionRecord{HolderIdentity: "bing"}),
			observedTime:      future,

			expectSuccess: false,
			outHolder:     "bing",
		},
		{
			name: "renew already required object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, primaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
					},
				},
				{
					verb:       "update",
					objectType: primaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
				{
					verb:       "get",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, createLockObject(t, secondaryType, action.GetNamespace(), action.(fakeclient.GetAction).GetName(), &rl.LeaderElectionRecord{HolderIdentity: "baz"}), nil
					},
				},
				{
					verb:       "update",
					objectType: secondaryType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
			},
			observedRecord:    rl.LeaderElectionRecord{HolderIdentity: "baz"},
			observedRawRecord: GetRawRecordOrDie(t, objectType, rl.LeaderElectionRecord{HolderIdentity: "baz"}),
			observedTime:      future,

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
				c.AddReactor(reactor.verb, reactor.objectType, reactor.reaction)
			}
			c.AddReactor("*", "*", func(action fakeclient.Action) (bool, runtime.Object, error) {
				t.Errorf("unreachable action. testclient called too many times: %+v", action)
				return true, nil, fmt.Errorf("unreachable action")
			})

			switch objectType {
			case rl.EndpointsLeasesResourceLock:
				lock = &rl.MultiLock{
					Primary: &rl.EndpointsLock{
						EndpointsMeta: objectMeta,
						LockConfig:    resourceLockConfig,
						Client:        c.CoreV1(),
					},
					Secondary: &rl.LeaseLock{
						LeaseMeta:  objectMeta,
						LockConfig: resourceLockConfig,
						Client:     c.CoordinationV1(),
					},
				}
			case rl.ConfigMapsLeasesResourceLock:
				lock = &rl.MultiLock{
					Primary: &rl.ConfigMapLock{
						ConfigMapMeta: objectMeta,
						LockConfig:    resourceLockConfig,
						Client:        c.CoreV1(),
					},
					Secondary: &rl.LeaseLock{
						LeaseMeta:  objectMeta,
						LockConfig: resourceLockConfig,
						Client:     c.CoordinationV1(),
					},
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
				config:            lec,
				observedRecord:    test.observedRecord,
				observedRawRecord: test.observedRawRecord,
				observedTime:      test.observedTime,
				clock:             clock.RealClock{},
			}
			if test.expectSuccess != le.tryAcquireOrRenew(context.Background()) {
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

// Will test leader election using endpointsleases as the resource
func TestTryAcquireOrRenewEndpointsLeases(t *testing.T) {
	testTryAcquireOrRenewMultiLock(t, "endpointsleases")
}

// Will test leader election using configmapsleases as the resource
func TestTryAcquireOrRenewConfigMapsLeases(t *testing.T) {
	testTryAcquireOrRenewMultiLock(t, "configmapsleases")
}

func testReleaseLease(t *testing.T, objectType string) {
	tests := []struct {
		name           string
		observedRecord rl.LeaderElectionRecord
		observedTime   time.Time
		reactors       []Reactor

		expectSuccess    bool
		transitionLeader bool
		outHolder        string
	}{
		{
			name: "release acquired lock from no object",
			reactors: []Reactor{
				{
					verb:       "get",
					objectType: objectType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
					},
				},
				{
					verb:       "create",
					objectType: objectType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.CreateAction).GetObject(), nil
					},
				},
				{
					verb:       "update",
					objectType: objectType,
					reaction: func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(fakeclient.UpdateAction).GetObject(), nil
					},
				},
			},
			expectSuccess: true,
			outHolder:     "",
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
			observedRawRecord := GetRawRecordOrDie(t, objectType, test.observedRecord)
			le := &LeaderElector{
				config:            lec,
				observedRecord:    test.observedRecord,
				observedRawRecord: observedRawRecord,
				observedTime:      test.observedTime,
				clock:             clock.RealClock{},
			}
			if !le.tryAcquireOrRenew(context.Background()) {
				t.Errorf("unexpected result of tryAcquireOrRenew: [succeeded=%v]", true)
			}

			le.maybeReportTransition()

			// Wait for a response to the leader transition, and add 1 so that we can track the final transition.
			wg.Wait()
			wg.Add(1)

			if test.expectSuccess != le.release() {
				t.Errorf("unexpected result of release: [succeeded=%v]", !test.expectSuccess)
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

// Will test leader election using endpoints as the resource
func TestReleaseLeaseEndpoints(t *testing.T) {
	testReleaseLease(t, "endpoints")
}

// Will test leader election using endpoints as the resource
func TestReleaseLeaseConfigMaps(t *testing.T) {
	testReleaseLease(t, "configmaps")
}

// Will test leader election using endpoints as the resource
func TestReleaseLeaseLeases(t *testing.T) {
	testReleaseLease(t, "leases")
}

func TestReleaseOnCancellation_Endpoints(t *testing.T) {
	testReleaseOnCancellation(t, "endpoints")
}

func TestReleaseOnCancellation_ConfigMaps(t *testing.T) {
	testReleaseOnCancellation(t, "configmaps")
}

func TestReleaseOnCancellation_Leases(t *testing.T) {
	testReleaseOnCancellation(t, "leases")
}

func testReleaseOnCancellation(t *testing.T, objectType string) {
	var (
		onNewLeader   = make(chan struct{})
		onRenewCalled = make(chan struct{})
		onRenewResume = make(chan struct{})
		onRelease     = make(chan struct{})

		lockObj runtime.Object
		updates int
	)

	resourceLockConfig := rl.ResourceLockConfig{
		Identity:      "baz",
		EventRecorder: &record.FakeRecorder{},
	}
	c := &fake.Clientset{}

	c.AddReactor("get", objectType, func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
		if lockObj != nil {
			return true, lockObj, nil
		}
		return true, nil, errors.NewNotFound(action.(fakeclient.GetAction).GetResource().GroupResource(), action.(fakeclient.GetAction).GetName())
	})

	// create lock
	c.AddReactor("create", objectType, func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
		lockObj = action.(fakeclient.CreateAction).GetObject()
		return true, lockObj, nil
	})

	c.AddReactor("update", objectType, func(action fakeclient.Action) (handled bool, ret runtime.Object, err error) {
		updates++

		// Second update (first renew) should return our canceled error
		// FakeClient doesn't do anything with the context so we're doing this ourselves
		if updates == 2 {
			close(onRenewCalled)
			<-onRenewResume
			return true, nil, context.Canceled
		} else if updates == 3 {
			close(onRelease)
		}

		lockObj = action.(fakeclient.UpdateAction).GetObject()
		return true, lockObj, nil

	})

	c.AddReactor("*", "*", func(action fakeclient.Action) (bool, runtime.Object, error) {
		t.Errorf("unreachable action. testclient called too many times: %+v", action)
		return true, nil, fmt.Errorf("unreachable action")
	})

	lock, err := rl.New(objectType, "foo", "bar", c.CoreV1(), c.CoordinationV1(), resourceLockConfig)
	if err != nil {
		t.Fatal("resourcelock.New() = ", err)
	}

	lec := LeaderElectionConfig{
		Lock:          lock,
		LeaseDuration: 15 * time.Second,
		RenewDeadline: 2 * time.Second,
		RetryPeriod:   1 * time.Second,

		// This is what we're testing
		ReleaseOnCancel: true,

		Callbacks: LeaderCallbacks{
			OnNewLeader:      func(identity string) {},
			OnStoppedLeading: func() {},
			OnStartedLeading: func(context.Context) {
				close(onNewLeader)
			},
		},
	}

	elector, err := NewLeaderElector(lec)
	if err != nil {
		t.Fatal("Failed to create leader elector: ", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	go elector.Run(ctx)

	// Wait for us to become the leader
	select {
	case <-onNewLeader:
	case <-time.After(10 * time.Second):
		t.Fatal("failed to become the leader")
	}

	// Wait for renew (update) to be invoked
	select {
	case <-onRenewCalled:
	case <-time.After(10 * time.Second):
		t.Fatal("the elector failed to renew the lock")
	}

	// Cancel the context - stopping the elector while
	// it's running
	cancel()

	// Resume the update call to return the cancellation
	// which should trigger the release flow
	close(onRenewResume)

	select {
	case <-onRelease:
	case <-time.After(10 * time.Second):
		t.Fatal("the lock was not released")
	}
}
