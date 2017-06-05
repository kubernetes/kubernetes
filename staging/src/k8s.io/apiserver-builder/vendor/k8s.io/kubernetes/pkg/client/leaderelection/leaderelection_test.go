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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/v1"
	fakeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	rl "k8s.io/kubernetes/pkg/client/leaderelection/resourcelock"
)

// Will test leader election using endpoints as the resource
func TestTryAcquireOrRenewEndpoints(t *testing.T) {
	future := time.Now().Add(1000 * time.Hour)
	past := time.Now().Add(-1000 * time.Hour)

	tests := []struct {
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
		// acquire from no endpoints
		{
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
						return true, action.(core.CreateAction).GetObject().(*v1.Endpoints), nil
					},
				},
			},
			expectSuccess: true,
			outHolder:     "baz",
		},
		// acquire from unled endpoints
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.Endpoints{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject().(*v1.Endpoints), nil
					},
				},
			},

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		// acquire from led, unacked endpoints
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.Endpoints{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
								Annotations: map[string]string{
									rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
								},
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject().(*v1.Endpoints), nil
					},
				},
			},
			observedRecord: rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedTime:   past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		// don't acquire from led, acked endpoints
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.Endpoints{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
								Annotations: map[string]string{
									rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
								},
							},
						}, nil
					},
				},
			},
			observedTime: future,

			expectSuccess: false,
			outHolder:     "bing",
		},
		// renew already acquired endpoints
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.Endpoints{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
								Annotations: map[string]string{
									rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"baz"}`,
								},
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject().(*v1.Endpoints), nil
					},
				},
			},
			observedTime:   future,
			observedRecord: rl.LeaderElectionRecord{HolderIdentity: "baz"},

			expectSuccess: true,
			outHolder:     "baz",
		},
	}

	for i, test := range tests {
		// OnNewLeader is called async so we have to wait for it.
		var wg sync.WaitGroup
		wg.Add(1)
		var reportedLeader string

		lock := rl.EndpointsLock{
			EndpointsMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
			LockConfig: rl.ResourceLockConfig{
				Identity:      "baz",
				EventRecorder: &record.FakeRecorder{},
			},
		}

		lec := LeaderElectionConfig{
			Lock:          &lock,
			LeaseDuration: 10 * time.Second,
			Callbacks: LeaderCallbacks{
				OnNewLeader: func(l string) {
					defer wg.Done()
					reportedLeader = l
				},
			},
		}
		c := &fakeclientset.Clientset{Fake: core.Fake{}}
		for _, reactor := range test.reactors {
			c.AddReactor(reactor.verb, "endpoints", reactor.reaction)
		}
		c.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
			t.Errorf("[%v] unreachable action. testclient called too many times: %+v", i, action)
			return true, nil, fmt.Errorf("uncreachable action")
		})

		le := &LeaderElector{
			config:         lec,
			observedRecord: test.observedRecord,
			observedTime:   test.observedTime,
		}
		lock.Client = c

		if test.expectSuccess != le.tryAcquireOrRenew() {
			t.Errorf("[%v]unexpected result of tryAcquireOrRenew: [succeded=%v]", i, !test.expectSuccess)
		}

		le.observedRecord.AcquireTime = metav1.Time{}
		le.observedRecord.RenewTime = metav1.Time{}
		if le.observedRecord.HolderIdentity != test.outHolder {
			t.Errorf("[%v]expected holder:\n\t%+v\ngot:\n\t%+v", i, test.outHolder, le.observedRecord.HolderIdentity)
		}
		if len(test.reactors) != len(c.Actions()) {
			t.Errorf("[%v]wrong number of api interactions", i)
		}
		if test.transitionLeader && le.observedRecord.LeaderTransitions != 1 {
			t.Errorf("[%v]leader should have transitioned but did not", i)
		}
		if !test.transitionLeader && le.observedRecord.LeaderTransitions != 0 {
			t.Errorf("[%v]leader should not have transitioned but did", i)
		}

		le.maybeReportTransition()
		wg.Wait()
		if reportedLeader != test.outHolder {
			t.Errorf("[%v]reported leader was not the new leader. expected %q, got %q", i, test.outHolder, reportedLeader)
		}
	}
}

// Will test leader election using configmap as the resource
func TestTryAcquireOrRenewConfigMap(t *testing.T) {
	future := time.Now().Add(1000 * time.Hour)
	past := time.Now().Add(-1000 * time.Hour)

	tests := []struct {
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
		// acquire from no onfigmap
		{
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
						return true, action.(core.CreateAction).GetObject().(*v1.ConfigMap), nil
					},
				},
			},
			expectSuccess: true,
			outHolder:     "baz",
		},
		// acquire from unled configmap
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.ConfigMap{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject().(*v1.ConfigMap), nil
					},
				},
			},

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		// acquire from led, unacked configmap
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.ConfigMap{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
								Annotations: map[string]string{
									rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
								},
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject().(*v1.ConfigMap), nil
					},
				},
			},
			observedRecord: rl.LeaderElectionRecord{HolderIdentity: "bing"},
			observedTime:   past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		// don't acquire from led, acked configmap
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.ConfigMap{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
								Annotations: map[string]string{
									rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
								},
							},
						}, nil
					},
				},
			},
			observedTime: future,

			expectSuccess: false,
			outHolder:     "bing",
		},
		// renew already acquired configmap
		{
			reactors: []struct {
				verb     string
				reaction core.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, &v1.ConfigMap{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(core.GetAction).GetName(),
								Annotations: map[string]string{
									rl.LeaderElectionRecordAnnotationKey: `{"holderIdentity":"baz"}`,
								},
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(core.CreateAction).GetObject().(*v1.ConfigMap), nil
					},
				},
			},
			observedTime:   future,
			observedRecord: rl.LeaderElectionRecord{HolderIdentity: "baz"},

			expectSuccess: true,
			outHolder:     "baz",
		},
	}

	for i, test := range tests {
		// OnNewLeader is called async so we have to wait for it.
		var wg sync.WaitGroup
		wg.Add(1)
		var reportedLeader string

		lock := rl.ConfigMapLock{
			ConfigMapMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
			LockConfig: rl.ResourceLockConfig{
				Identity:      "baz",
				EventRecorder: &record.FakeRecorder{},
			},
		}

		lec := LeaderElectionConfig{
			Lock:          &lock,
			LeaseDuration: 10 * time.Second,
			Callbacks: LeaderCallbacks{
				OnNewLeader: func(l string) {
					defer wg.Done()
					reportedLeader = l
				},
			},
		}
		c := &fakeclientset.Clientset{Fake: core.Fake{}}
		for _, reactor := range test.reactors {
			c.AddReactor(reactor.verb, "configmaps", reactor.reaction)
		}
		c.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
			t.Errorf("[%v] unreachable action. testclient called too many times: %+v", i, action)
			return true, nil, fmt.Errorf("uncreachable action")
		})

		le := &LeaderElector{
			config:         lec,
			observedRecord: test.observedRecord,
			observedTime:   test.observedTime,
		}
		lock.Client = c

		if test.expectSuccess != le.tryAcquireOrRenew() {
			t.Errorf("[%v]unexpected result of tryAcquireOrRenew: [succeded=%v]", i, !test.expectSuccess)
		}

		le.observedRecord.AcquireTime = metav1.Time{}
		le.observedRecord.RenewTime = metav1.Time{}
		if le.observedRecord.HolderIdentity != test.outHolder {
			t.Errorf("[%v]expected holder:\n\t%+v\ngot:\n\t%+v", i, test.outHolder, le.observedRecord.HolderIdentity)
		}
		if len(test.reactors) != len(c.Actions()) {
			t.Errorf("[%v]wrong number of api interactions", i)
		}
		if test.transitionLeader && le.observedRecord.LeaderTransitions != 1 {
			t.Errorf("[%v]leader should have transitioned but did not", i)
		}
		if !test.transitionLeader && le.observedRecord.LeaderTransitions != 0 {
			t.Errorf("[%v]leader should not have transitioned but did", i)
		}

		le.maybeReportTransition()
		wg.Wait()
		if reportedLeader != test.outHolder {
			t.Errorf("[%v]reported leader was not the new leader. expected %q, got %q", i, test.outHolder, reportedLeader)
		}
	}
}
