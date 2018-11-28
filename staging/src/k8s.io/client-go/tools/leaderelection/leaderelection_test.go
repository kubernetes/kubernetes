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
	"fmt"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	fakecorev1 "k8s.io/client-go/kubernetes/typed/core/v1/fake"
	core "k8s.io/client-go/testing"
	rl "k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
)

func createLockObject(objectType string, objectMeta metav1.ObjectMeta) (obj runtime.Object) {
	switch objectType {
	case "endpoints":
		obj = &v1.Endpoints{ObjectMeta: objectMeta}
	case "configmaps":
		obj = &v1.ConfigMap{ObjectMeta: objectMeta}
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
			reaction core.ReactionFunc
		}

		expectSuccess    bool
		transitionLeader bool
		outHolder        string
	}{
		{
			name: "acquire from no object",
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(action.(core.GetAction).GetResource().GroupResource(), action.(core.GetAction).GetName())
					},
				},
				{
					verb: "create",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject(), nil
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
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						objectMeta := metav1.ObjectMeta{
							Namespace: action.GetNamespace(),
							Name:      action.(core.GetAction).GetName(),
						}
						return true, createLockObject(objectType, objectMeta), nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject(), nil
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
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						objectMeta := metav1.ObjectMeta{
							Namespace: action.GetNamespace(),
							Name:      action.(core.GetAction).GetName(),
							Annotations: map[string]string{
								rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
							},
						}
						return true, createLockObject(objectType, objectMeta), nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject(), nil
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
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						objectMeta := metav1.ObjectMeta{
							Namespace: action.GetNamespace(),
							Name:      action.(core.GetAction).GetName(),
							Annotations: map[string]string{
								rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":""}`,
							},
						}
						return true, createLockObject(objectType, objectMeta), nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject(), nil
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
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						objectMeta := metav1.ObjectMeta{
							Namespace: action.GetNamespace(),
							Name:      action.(core.GetAction).GetName(),
							Annotations: map[string]string{
								rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
							},
						}
						return true, createLockObject(objectType, objectMeta), nil
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
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						objectMeta := metav1.ObjectMeta{
							Namespace: action.GetNamespace(),
							Name:      action.(core.GetAction).GetName(),
							Annotations: map[string]string{
								rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"baz"}`,
							},
						}
						return true, createLockObject(objectType, objectMeta), nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject(), nil
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
			c := &fakecorev1.FakeCoreV1{Fake: &core.Fake{}}
			for _, reactor := range test.reactors {
				c.AddReactor(reactor.verb, objectType, reactor.reaction)
			}
			c.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
				t.Errorf("unreachable action. testclient called too many times: %+v", action)
				return true, nil, fmt.Errorf("unreachable action")
			})

			switch objectType {
			case "endpoints":
				lock = &rl.EndpointsLock{
					EndpointsMeta: objectMeta,
					LockConfig:    resourceLockConfig,
					Client:        c,
				}
			case "configmaps":
				lock = &rl.ConfigMapLock{
					ConfigMapMeta: objectMeta,
					LockConfig:    resourceLockConfig,
					Client:        c,
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
