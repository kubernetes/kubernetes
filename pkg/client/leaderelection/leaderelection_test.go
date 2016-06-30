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

// Package leaderelection implements leader election of a set of endpoints.
// It uses an annotation in the endpoints object to store the record of the
// election state.

package leaderelection

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestTryAcquireOrRenew(t *testing.T) {
	future := time.Now().Add(1000 * time.Hour)
	past := time.Now().Add(-1000 * time.Hour)

	tests := []struct {
		observedRecord LeaderElectionRecord
		observedTime   time.Time
		reactors       []struct {
			verb     string
			reaction testclient.ReactionFunc
		}

		expectSuccess    bool
		transitionLeader bool
		outHolder        string
	}{
		// acquire from no endpoints
		{
			reactors: []struct {
				verb     string
				reaction testclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.NewNotFound(api.Resource(action.(testclient.GetAction).GetResource()), action.(testclient.GetAction).GetName())
					},
				},
				{
					verb: "create",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(testclient.CreateAction).GetObject().(*api.Endpoints), nil
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
				reaction testclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(testclient.GetAction).GetName(),
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(testclient.CreateAction).GetObject().(*api.Endpoints), nil
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
				reaction testclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(testclient.GetAction).GetName(),
								Annotations: map[string]string{
									LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
								},
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(testclient.CreateAction).GetObject().(*api.Endpoints), nil
					},
				},
			},
			observedRecord: LeaderElectionRecord{HolderIdentity: "bing"},
			observedTime:   past,

			expectSuccess:    true,
			transitionLeader: true,
			outHolder:        "baz",
		},
		// don't acquire from led, acked endpoints
		{
			reactors: []struct {
				verb     string
				reaction testclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(testclient.GetAction).GetName(),
								Annotations: map[string]string{
									LeaderElectionRecordAnnotationKey: `{"holderIdentity":"bing"}`,
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
				reaction testclient.ReactionFunc
			}{
				{
					verb: "get",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, &api.Endpoints{
							ObjectMeta: api.ObjectMeta{
								Namespace: action.GetNamespace(),
								Name:      action.(testclient.GetAction).GetName(),
								Annotations: map[string]string{
									LeaderElectionRecordAnnotationKey: `{"holderIdentity":"baz"}`,
								},
							},
						}, nil
					},
				},
				{
					verb: "update",
					reaction: func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
						return true, action.(testclient.CreateAction).GetObject().(*api.Endpoints), nil
					},
				},
			},
			observedTime:   future,
			observedRecord: LeaderElectionRecord{HolderIdentity: "baz"},

			expectSuccess: true,
			outHolder:     "baz",
		},
	}

	for i, test := range tests {
		// OnNewLeader is called async so we have to wait for it.
		var wg sync.WaitGroup
		wg.Add(1)
		var reportedLeader string

		lec := LeaderElectionConfig{
			EndpointsMeta: api.ObjectMeta{Namespace: "foo", Name: "bar"},
			Identity:      "baz",
			EventRecorder: &record.FakeRecorder{},
			LeaseDuration: 10 * time.Second,
			Callbacks: LeaderCallbacks{
				OnNewLeader: func(l string) {
					defer wg.Done()
					reportedLeader = l
				},
			},
		}
		c := &testclient.Fake{}
		for _, reactor := range test.reactors {
			c.AddReactor(reactor.verb, "endpoints", reactor.reaction)
		}
		c.AddReactor("*", "*", func(action testclient.Action) (bool, runtime.Object, error) {
			t.Errorf("[%v] unreachable action. testclient called too many times: %+v", i, action)
			return true, nil, fmt.Errorf("uncreachable action")
		})

		le := &LeaderElector{
			config:         lec,
			observedRecord: test.observedRecord,
			observedTime:   test.observedTime,
		}
		le.config.Client = c

		if test.expectSuccess != le.tryAcquireOrRenew() {
			t.Errorf("[%v]unexpected result of tryAcquireOrRenew: [succeded=%v]", i, !test.expectSuccess)
		}

		le.observedRecord.AcquireTime = unversioned.Time{}
		le.observedRecord.RenewTime = unversioned.Time{}
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
