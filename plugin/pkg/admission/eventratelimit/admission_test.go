/*
Copyright 2017 The Kubernetes Authors.

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

package eventratelimit

import (
	"net/http"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	api "k8s.io/kubernetes/pkg/apis/core"
	eventratelimitapi "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit"
)

const (
	qps          = 1
	eventKind    = "Event"
	nonEventKind = "NonEvent"
)

// attributesForRequest generates the admission.Attributes that for the specified request
func attributesForRequest(rq request) admission.Attributes {
	return admission.NewAttributesRecord(
		rq.event,
		nil,
		api.Kind(rq.kind).WithVersion("version"),
		rq.namespace,
		"name",
		api.Resource("resource").WithVersion("version"),
		"",
		admission.Create,
		&metav1.CreateOptions{},
		rq.dryRun,
		&user.DefaultInfo{Name: rq.username})
}

type request struct {
	kind      string
	namespace string
	username  string
	event     *api.Event
	delay     time.Duration
	accepted  bool
	dryRun    bool
}

func newRequest(kind string) request {
	return request{
		kind:     kind,
		accepted: true,
	}
}

func newEventRequest() request {
	return newRequest(eventKind)
}

func newNonEventRequest() request {
	return newRequest(nonEventKind)
}

func (r request) withNamespace(namespace string) request {
	r.namespace = namespace
	return r
}

func (r request) withEvent(event *api.Event) request {
	r.event = event
	return r
}

func (r request) withEventComponent(component string) request {
	return r.withEvent(&api.Event{
		Source: api.EventSource{
			Component: component,
		},
	})
}

func (r request) withDryRun(dryRun bool) request {
	r.dryRun = dryRun
	return r
}

func (r request) withUser(name string) request {
	r.username = name
	return r
}

func (r request) blocked() request {
	r.accepted = false
	return r
}

// withDelay will adjust the clock to simulate the specified delay, in seconds
func (r request) withDelay(delayInSeconds int) request {
	r.delay = time.Duration(delayInSeconds) * time.Second
	return r
}

// createSourceAndObjectKeyInclusionRequests creates a series of requests that can be used
// to test that a particular part of the event is included in the source+object key
func createSourceAndObjectKeyInclusionRequests(eventFactory func(label string) *api.Event) []request {
	return []request{
		newEventRequest().withEvent(eventFactory("A")),
		newEventRequest().withEvent(eventFactory("A")).blocked(),
		newEventRequest().withEvent(eventFactory("B")),
	}
}

func TestEventRateLimiting(t *testing.T) {
	cases := []struct {
		name                     string
		serverBurst              int32
		namespaceBurst           int32
		namespaceCacheSize       int32
		sourceAndObjectBurst     int32
		sourceAndObjectCacheSize int32
		userBurst                int32
		userCacheSize            int32
		requests                 []request
	}{
		{
			name:        "event not blocked when tokens available",
			serverBurst: 3,
			requests: []request{
				newEventRequest(),
			},
		},
		{
			name:        "non-event not blocked",
			serverBurst: 3,
			requests: []request{
				newNonEventRequest(),
			},
		},
		{
			name:        "event blocked after tokens exhausted",
			serverBurst: 3,
			requests: []request{
				newEventRequest(),
				newEventRequest(),
				newEventRequest(),
				newEventRequest().blocked(),
			},
		},
		{
			name:        "event not blocked by dry-run requests",
			serverBurst: 3,
			requests: []request{
				newEventRequest(),
				newEventRequest(),
				newEventRequest().withDryRun(true),
				newEventRequest().withDryRun(true),
				newEventRequest().withDryRun(true),
				newEventRequest().withDryRun(true),
				newEventRequest(),
				newEventRequest().blocked(),
				newEventRequest().withDryRun(true),
			},
		},
		{
			name:        "non-event not blocked after tokens exhausted",
			serverBurst: 3,
			requests: []request{
				newEventRequest(),
				newEventRequest(),
				newEventRequest(),
				newNonEventRequest(),
			},
		},
		{
			name:        "non-events should not count against limit",
			serverBurst: 3,
			requests: []request{
				newEventRequest(),
				newEventRequest(),
				newNonEventRequest(),
				newEventRequest(),
			},
		},
		{
			name:        "event accepted after token refill",
			serverBurst: 3,
			requests: []request{
				newEventRequest(),
				newEventRequest(),
				newEventRequest(),
				newEventRequest().blocked(),
				newEventRequest().withDelay(1),
			},
		},
		{
			name:               "event blocked by namespace limits",
			serverBurst:        100,
			namespaceBurst:     3,
			namespaceCacheSize: 10,
			requests: []request{
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A").blocked(),
			},
		},
		{
			name:               "event from other namespace not blocked",
			serverBurst:        100,
			namespaceBurst:     3,
			namespaceCacheSize: 10,
			requests: []request{
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("B"),
			},
		},
		{
			name:               "events from other namespaces should not count against limit",
			serverBurst:        100,
			namespaceBurst:     3,
			namespaceCacheSize: 10,
			requests: []request{
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("B"),
				newEventRequest().withNamespace("A"),
			},
		},
		{
			name:               "event accepted after namespace token refill",
			serverBurst:        100,
			namespaceBurst:     3,
			namespaceCacheSize: 10,
			requests: []request{
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A").blocked(),
				newEventRequest().withNamespace("A").withDelay(1),
			},
		},
		{
			name:               "event from other namespaces should not clear namespace limits",
			serverBurst:        100,
			namespaceBurst:     3,
			namespaceCacheSize: 10,
			requests: []request{
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("B"),
				newEventRequest().withNamespace("A").blocked(),
			},
		},
		{
			name:               "namespace limits from lru namespace should clear when cache size exceeded",
			serverBurst:        100,
			namespaceBurst:     3,
			namespaceCacheSize: 2,
			requests: []request{
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("B"),
				newEventRequest().withNamespace("B"),
				newEventRequest().withNamespace("B"),
				newEventRequest().withNamespace("A"),
				newEventRequest().withNamespace("B").blocked(),
				newEventRequest().withNamespace("A").blocked(),
				// This should clear out namespace B from the lru cache
				newEventRequest().withNamespace("C"),
				newEventRequest().withNamespace("A").blocked(),
				newEventRequest().withNamespace("B"),
			},
		},
		{
			name:                     "event blocked by source+object limits",
			serverBurst:              100,
			sourceAndObjectBurst:     3,
			sourceAndObjectCacheSize: 10,
			requests: []request{
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A").blocked(),
			},
		},
		{
			name:                     "event from other source+object not blocked",
			serverBurst:              100,
			sourceAndObjectBurst:     3,
			sourceAndObjectCacheSize: 10,
			requests: []request{
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("B"),
			},
		},
		{
			name:                     "events from other source+object should not count against limit",
			serverBurst:              100,
			sourceAndObjectBurst:     3,
			sourceAndObjectCacheSize: 10,
			requests: []request{
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("B"),
				newEventRequest().withEventComponent("A"),
			},
		},
		{
			name:                     "event accepted after source+object token refill",
			serverBurst:              100,
			sourceAndObjectBurst:     3,
			sourceAndObjectCacheSize: 10,
			requests: []request{
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A").blocked(),
				newEventRequest().withEventComponent("A").withDelay(1),
			},
		},
		{
			name:                     "event from other source+object should not clear source+object limits",
			serverBurst:              100,
			sourceAndObjectBurst:     3,
			sourceAndObjectCacheSize: 10,
			requests: []request{
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("B"),
				newEventRequest().withEventComponent("A").blocked(),
			},
		},
		{
			name:                     "source+object limits from lru source+object should clear when cache size exceeded",
			serverBurst:              100,
			sourceAndObjectBurst:     3,
			sourceAndObjectCacheSize: 2,
			requests: []request{
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("B"),
				newEventRequest().withEventComponent("B"),
				newEventRequest().withEventComponent("B"),
				newEventRequest().withEventComponent("A"),
				newEventRequest().withEventComponent("B").blocked(),
				newEventRequest().withEventComponent("A").blocked(),
				// This should clear out component B from the lru cache
				newEventRequest().withEventComponent("C"),
				newEventRequest().withEventComponent("A").blocked(),
				newEventRequest().withEventComponent("B"),
			},
		},
		{
			name:                     "source host should be included in source+object key",
			serverBurst:              100,
			sourceAndObjectBurst:     1,
			sourceAndObjectCacheSize: 10,
			requests: createSourceAndObjectKeyInclusionRequests(func(label string) *api.Event {
				return &api.Event{Source: api.EventSource{Host: label}}
			}),
		},
		{
			name:                     "involved object kind should be included in source+object key",
			serverBurst:              100,
			sourceAndObjectBurst:     1,
			sourceAndObjectCacheSize: 10,
			requests: createSourceAndObjectKeyInclusionRequests(func(label string) *api.Event {
				return &api.Event{InvolvedObject: api.ObjectReference{Kind: label}}
			}),
		},
		{
			name:                     "involved object namespace should be included in source+object key",
			serverBurst:              100,
			sourceAndObjectBurst:     1,
			sourceAndObjectCacheSize: 10,
			requests: createSourceAndObjectKeyInclusionRequests(func(label string) *api.Event {
				return &api.Event{InvolvedObject: api.ObjectReference{Namespace: label}}
			}),
		},
		{
			name:                     "involved object name should be included in source+object key",
			serverBurst:              100,
			sourceAndObjectBurst:     1,
			sourceAndObjectCacheSize: 10,
			requests: createSourceAndObjectKeyInclusionRequests(func(label string) *api.Event {
				return &api.Event{InvolvedObject: api.ObjectReference{Name: label}}
			}),
		},
		{
			name:                     "involved object UID should be included in source+object key",
			serverBurst:              100,
			sourceAndObjectBurst:     1,
			sourceAndObjectCacheSize: 10,
			requests: createSourceAndObjectKeyInclusionRequests(func(label string) *api.Event {
				return &api.Event{InvolvedObject: api.ObjectReference{UID: types.UID(label)}}
			}),
		},
		{
			name:                     "involved object APIVersion should be included in source+object key",
			serverBurst:              100,
			sourceAndObjectBurst:     1,
			sourceAndObjectCacheSize: 10,
			requests: createSourceAndObjectKeyInclusionRequests(func(label string) *api.Event {
				return &api.Event{InvolvedObject: api.ObjectReference{APIVersion: label}}
			}),
		},
		{
			name:          "event blocked by user limits",
			userBurst:     3,
			userCacheSize: 10,
			requests: []request{
				newEventRequest().withUser("A"),
				newEventRequest().withUser("A"),
				newEventRequest().withUser("A"),
				newEventRequest().withUser("A").blocked(),
			},
		},
		{
			name: "event from other user not blocked",
			requests: []request{
				newEventRequest().withUser("A"),
				newEventRequest().withUser("A"),
				newEventRequest().withUser("A"),
				newEventRequest().withUser("B"),
			},
		},
		{
			name: "events from other user should not count against limit",
			requests: []request{
				newEventRequest().withUser("A"),
				newEventRequest().withUser("A"),
				newEventRequest().withUser("B"),
				newEventRequest().withUser("A"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clock := clock.NewFakeClock(time.Now())
			config := &eventratelimitapi.Configuration{}
			if tc.serverBurst > 0 {
				serverLimit := eventratelimitapi.Limit{
					Type:  eventratelimitapi.ServerLimitType,
					QPS:   qps,
					Burst: tc.serverBurst,
				}
				config.Limits = append(config.Limits, serverLimit)
			}
			if tc.namespaceBurst > 0 {
				namespaceLimit := eventratelimitapi.Limit{
					Type:      eventratelimitapi.NamespaceLimitType,
					Burst:     tc.namespaceBurst,
					QPS:       qps,
					CacheSize: tc.namespaceCacheSize,
				}
				config.Limits = append(config.Limits, namespaceLimit)
			}
			if tc.userBurst > 0 {
				userLimit := eventratelimitapi.Limit{
					Type:      eventratelimitapi.UserLimitType,
					Burst:     tc.userBurst,
					QPS:       qps,
					CacheSize: tc.userCacheSize,
				}
				config.Limits = append(config.Limits, userLimit)
			}
			if tc.sourceAndObjectBurst > 0 {
				sourceAndObjectLimit := eventratelimitapi.Limit{
					Type:      eventratelimitapi.SourceAndObjectLimitType,
					Burst:     tc.sourceAndObjectBurst,
					QPS:       qps,
					CacheSize: tc.sourceAndObjectCacheSize,
				}
				config.Limits = append(config.Limits, sourceAndObjectLimit)
			}
			eventratelimit, err := newEventRateLimit(config, clock)
			if err != nil {
				t.Fatalf("%v: Could not create EventRateLimit: %v", tc.name, err)
			}

			for rqIndex, rq := range tc.requests {
				if rq.delay > 0 {
					clock.Step(rq.delay)
				}
				attributes := attributesForRequest(rq)
				err = eventratelimit.Validate(attributes, nil)
				if rq.accepted != (err == nil) {
					expectedAction := "admitted"
					if !rq.accepted {
						expectedAction = "blocked"
					}
					t.Fatalf("%v: Request %v should have been %v: %v", tc.name, rqIndex, expectedAction, err)
				}
				if err != nil {
					statusErr, ok := err.(*errors.StatusError)
					if ok && statusErr.ErrStatus.Code != http.StatusTooManyRequests {
						t.Fatalf("%v: Request %v should yield a 429 response: %v", tc.name, rqIndex, err)
					}
				}
			}
		})
	}
}
