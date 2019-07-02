/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	rmtypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apimachinery/pkg/util/sets"
)

func createRequestManagementServer(
	delegate http.Handler, flowSchemas []*rmtypesv1a1.FlowSchema, priorityLevelConfigurations []*rmtypesv1a1.PriorityLevelConfiguration,
	serverConcurrencyLimit int, requestWaitLimit time.Duration) *httptest.Server {
	longRunningRequestCheck := BasicLongRunningRequestCheck(sets.NewString("watch"), sets.NewString("proxy"))

	// requestInfoFactory := &apirequest.RequestInfoFactory{APIPrefixes: sets.NewString("apis", "api"), GrouplessAPIPrefixes: sets.NewString("api")}

	handler := WithRequestManagement(
		delegate,
		flowSchemas,
		priorityLevelConfigurations,
		serverConcurrencyLimit,
		requestWaitLimit,
		longRunningRequestCheck,
	)
	// TODO(aaron-prindle) add back if needed...
	// handler = withFakeUser(handler)
	// handler = apifilters.WithRequestInfo(handler, requestInfoFactory)

	return httptest.NewServer(handler)
}

func newFalse() *bool {
	b := false
	return &b
}

func newHandSize(x int32) *int32 {
	return &x
}

// current tests assume serverConcurrencyLimit >= # of flowschemas
func TestRequestManagementTwoRequests(t *testing.T) {
	flowSchemas := []*rmtypesv1a1.FlowSchema{
		&rmtypesv1a1.FlowSchema{
			// metav1.TypeMeta
			// metav1.ObjectMeta
			Spec: rmtypesv1a1.FlowSchemaSpec{
				PriorityLevelConfiguration: rmtypesv1a1.PriorityLevelConfigurationReference{
					"plc-0",
				},
				// MatchingPrecedence: 1, // TODO(aaron-prindle) currently ignored
				// DistinguisherMethod: *rmtypesv1a1.FlowDistinguisherMethodByUserType, // TODO(aaron-prindle) currently ignored
				// Rules: []rmtypesv1a1.PolicyRuleWithSubjects{}, // TODO(aaron-prindle) currently ignored
			},
			// Status: rmtypesv1a1.FlowSchemaStatus{},
		},
	}

	priorityLevelConfigurations := []*rmtypesv1a1.PriorityLevelConfiguration{
		&rmtypesv1a1.PriorityLevelConfiguration{
			Spec: rmtypesv1a1.PriorityLevelConfigurationSpec{
				AssuredConcurrencyShares: 10,
				Exempt:                   newFalse(),
				GlobalDefault:            newFalse(),
				HandSize:                 newHandSize(8),
				QueueLengthLimit:         int32(65536),
				Queues:                   int32(128),
			},
		},
	}
	for i := range priorityLevelConfigurations {
		priorityLevelConfigurations[i].Name = fmt.Sprintf("plc-%d", i)
	}

	serverConcurrencyLimit := 2
	// serverConcurrencyLimit := 100
	requestWaitLimit := 1 * time.Millisecond
	requests := 2
	// requests := 5000

	var count int64
	delegate := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&count, int64(1))
	})

	server := createRequestManagementServer(
		delegate, flowSchemas, priorityLevelConfigurations,
		serverConcurrencyLimit, requestWaitLimit,
	)
	defer server.Close()

	header := make(http.Header)
	header.Add("PRIORITY", "0")
	req, _ := http.NewRequest("GET", server.URL, nil)
	req.Header = header

	w := &Work{
		Request: req,
		N:       requests,
		C:       2, // C > serverConcurrencyLimit - must be or else nothing
		// C:       5, // C > serverConcurrencyLimit - must be or else nothing
	}
	// Run blocks until all work is done.
	w.Run()
	if count != int64(requests) {
		t.Errorf("Expected to send %d requests, found %v", requests, count)
	}
}

// current tests assume serverConcurrencyLimit >= # of flowschemas
func TestRequestManagement(t *testing.T) {
	flowSchemas := []*rmtypesv1a1.FlowSchema{
		&rmtypesv1a1.FlowSchema{
			// metav1.TypeMeta
			// metav1.ObjectMeta
			Spec: rmtypesv1a1.FlowSchemaSpec{
				PriorityLevelConfiguration: rmtypesv1a1.PriorityLevelConfigurationReference{
					"plc-0",
				},
				// MatchingPrecedence: 1, // TODO(aaron-prindle) currently ignored
				// DistinguisherMethod: *rmtypesv1a1.FlowDistinguisherMethodByUserType, // TODO(aaron-prindle) currently ignored
				// Rules: []rmtypesv1a1.PolicyRuleWithSubjects{}, // TODO(aaron-prindle) currently ignored
			},
			// Status: rmtypesv1a1.FlowSchemaStatus{},
		},
	}

	priorityLevelConfigurations := []*rmtypesv1a1.PriorityLevelConfiguration{
		&rmtypesv1a1.PriorityLevelConfiguration{
			Spec: rmtypesv1a1.PriorityLevelConfigurationSpec{
				AssuredConcurrencyShares: 10,
				Exempt:                   newFalse(),
				GlobalDefault:            newFalse(),
				HandSize:                 newHandSize(8),
				QueueLengthLimit:         int32(65536),
				Queues:                   int32(128),
			},
		},
	}
	for i := range priorityLevelConfigurations {
		priorityLevelConfigurations[i].Name = fmt.Sprintf("plc-%d", i)
	}

	serverConcurrencyLimit := 100
	requestWaitLimit := 1 * time.Millisecond
	requests := 5000

	var count int64
	delegate := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&count, int64(1))
	})

	server := createRequestManagementServer(
		delegate, flowSchemas, priorityLevelConfigurations,
		serverConcurrencyLimit, requestWaitLimit,
	)
	defer server.Close()

	header := make(http.Header)
	header.Add("PRIORITY", "0")
	req, _ := http.NewRequest("GET", server.URL, nil)
	req.Header = header

	w := &Work{
		Request: req,
		N:       requests,
		C:       100, // C > serverConcurrencyLimit - must be or else nothing
	}
	// Run blocks until all work is done.
	w.Run()

	if count != int64(requests) {
		t.Errorf("Expected to send %d requests, found %v", requests, count)
	}
}

// TestRequestMultiple verifies that fairness is preserved across 5 flowschemas
// and that the serverConcurrencyLimit/assuredConcurrencyShares are correctly
// distributed
func TestRequestMultiple(t *testing.T) {
	// init objects
	groups := 5
	flowSchemas := []*rmtypesv1a1.FlowSchema{}
	for i := 0; i < groups; i++ {
		flowSchemas = append(flowSchemas,
			&rmtypesv1a1.FlowSchema{
				// metav1.TypeMeta
				// metav1.ObjectMeta
				Spec: rmtypesv1a1.FlowSchemaSpec{
					PriorityLevelConfiguration: rmtypesv1a1.PriorityLevelConfigurationReference{
						fmt.Sprintf("plc-%d", i),
					},
					MatchingPrecedence: int32(i), // TODO(aaron-prindle) currently ignored
					// DistinguisherMethod: *rmtypesv1a1.FlowDistinguisherMethodByUserType, // TODO(aaron-prindle) currently ignored
					// Rules: []rmtypesv1a1.PolicyRuleWithSubjects{}, // TODO(aaron-prindle) currently ignored
				},
				// Status: rmtypesv1a1.FlowSchemaStatus{},
			},
		)
	}
	priorityLevelConfigurations := []*rmtypesv1a1.PriorityLevelConfiguration{}

	for i := 0; i < groups; i++ {
		priorityLevelConfigurations = append(priorityLevelConfigurations,
			&rmtypesv1a1.PriorityLevelConfiguration{
				Spec: rmtypesv1a1.PriorityLevelConfigurationSpec{
					AssuredConcurrencyShares: 1,
					Exempt:                   newFalse(),
					GlobalDefault:            newFalse(),
					HandSize:                 newHandSize(8),
					QueueLengthLimit:         int32(65536),
					Queues:                   int32(128),
				},
			},
		)
	}
	// can't init nested struct in initializer
	for i := range priorityLevelConfigurations {
		priorityLevelConfigurations[i].Name = fmt.Sprintf("plc-%d", i)
	}

	serverConcurrencyLimit := groups
	requestWaitLimit := 5 * time.Second
	requests := 100

	countnum := len(flowSchemas)
	counts := []int64{}
	for range flowSchemas {
		counts = append(counts, int64(0))
	}

	delegate := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		for i := 0; i < countnum; i++ {
			if r.Header.Get("PRIORITY") == strconv.Itoa(i) {
				atomic.AddInt64(&counts[i], int64(1))
			}
		}
	})

	server := createRequestManagementServer(
		delegate, flowSchemas, priorityLevelConfigurations,
		serverConcurrencyLimit, requestWaitLimit,
	)
	defer server.Close()

	ws := []*Work{}
	for i := 0; i < countnum; i++ {
		header := make(http.Header)
		header.Add("PRIORITY", strconv.Itoa(i))
		req, _ := http.NewRequest("GET", server.URL, nil)
		req.Header = header

		ws = append(ws, &Work{
			Request: req,
			N:       requests,
			C:       serverConcurrencyLimit,
		})

	}
	for _, w := range ws {
		w := w
		// fmt.Printf("w %d info: %s\n", i, w.Request.Header.Get("PRIORITY"))
		go func() { w.Run() }()
	}

	time.Sleep(1 * time.Second)
	for i, count := range counts {
		if count != 1 {
			t.Errorf("Expected to dispatch 1 request for Group %d, found %v", i, count)
		}
	}
}

// TestRequestTimeout
func TestRequestTimeout(t *testing.T) {
	flowSchemas := []*rmtypesv1a1.FlowSchema{
		&rmtypesv1a1.FlowSchema{
			// metav1.TypeMeta
			// metav1.ObjectMeta
			Spec: rmtypesv1a1.FlowSchemaSpec{
				PriorityLevelConfiguration: rmtypesv1a1.PriorityLevelConfigurationReference{
					"plc-0",
				},
				// MatchingPrecedence: 1, // TODO(aaron-prindle) currently ignored
				// DistinguisherMethod: *rmtypesv1a1.FlowDistinguisherMethodByUserType, // TODO(aaron-prindle) currently ignored
				// Rules: []rmtypesv1a1.PolicyRuleWithSubjects{}, // TODO(aaron-prindle) currently ignored
			},
			// Status: rmtypesv1a1.FlowSchemaStatus{},
		},
	}

	priorityLevelConfigurations := []*rmtypesv1a1.PriorityLevelConfiguration{
		&rmtypesv1a1.PriorityLevelConfiguration{
			Spec: rmtypesv1a1.PriorityLevelConfigurationSpec{
				AssuredConcurrencyShares: 10,
				Exempt:                   newFalse(),
				GlobalDefault:            newFalse(),
				HandSize:                 newHandSize(1),
				QueueLengthLimit:         int32(65536),
				Queues:                   int32(1),
			},
		},
	}
	for i := range priorityLevelConfigurations {
		priorityLevelConfigurations[i].Name = fmt.Sprintf("plc-%d", i)
	}

	serverConcurrencyLimit := 1000
	// TODO(aaron-prindle) currently ignored
	requestWaitLimit := 1 * time.Nanosecond
	requests := 5000

	var count int64
	delegate := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&count, int64(1))
		time.Sleep(1 * time.Second)
	})

	server := createRequestManagementServer(
		delegate, flowSchemas, priorityLevelConfigurations,
		serverConcurrencyLimit, requestWaitLimit,
	)
	defer server.Close()

	header := make(http.Header)
	header.Add("PRIORITY", "0")
	req, _ := http.NewRequest("GET", server.URL, nil)
	req.Header = header

	w := &Work{
		Request: req,
		N:       requests,
		C:       1000, // C > serverConcurrencyLimit - must be or else nothing
	}

	// Run blocks until all work is done.
	w.Run()
	// TODO(aaron-prindle) make this more exact
	// perhaps check for timeout text in stdout/stderr
	if int64(requests) == count {
		t.Errorf("Expected some requests to timeout, recieved all requests %v/%v",
			requests, count)
	}
}

// func TestRequestQueueLengthLimit(t *testing.T) {
// 	flowSchemas := []*rmtypesv1a1.FlowSchema{
// 		&rmtypesv1a1.FlowSchema{
// 			// metav1.TypeMeta
// 			// metav1.ObjectMeta
// 			Spec: rmtypesv1a1.FlowSchemaSpec{
// 				PriorityLevelConfiguration: rmtypesv1a1.PriorityLevelConfigurationReference{
// 					"plc-0",
// 				},
// 				// MatchingPrecedence: 1, // TODO(aaron-prindle) currently ignored
// 				// DistinguisherMethod: *rmtypesv1a1.FlowDistinguisherMethodByUserType, // TODO(aaron-prindle) currently ignored
// 				// Rules: []rmtypesv1a1.PolicyRuleWithSubjects{}, // TODO(aaron-prindle) currently ignored
// 			},
// 			// Status: rmtypesv1a1.FlowSchemaStatus{},
// 		},
// 	}

// 	priorityLevelConfigurations := []*rmtypesv1a1.PriorityLevelConfiguration{
// 		&rmtypesv1a1.PriorityLevelConfiguration{
// 			Spec: rmtypesv1a1.PriorityLevelConfigurationSpec{
// 				AssuredConcurrencyShares: 10,
// 				Exempt:                   newFalse(),
// 				GlobalDefault:            newFalse(),
// 				HandSize:                 newHandSize(1),
// 				QueueLengthLimit:         int32(0),
// 				Queues:                   int32(1),
// 			},
// 		},
// 	}
// 	for i := range priorityLevelConfigurations {
// 		priorityLevelConfigurations[i].Name = fmt.Sprintf("plc-%d", i)
// 	}

// 	serverConcurrencyLimit := 1000
// 	// TODO(aaron-prindle) currently ignored
// 	requestWaitLimit := 1 * time.Millisecond
// 	requests := 5000

// 	var count int64
// 	delegate := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
// 		atomic.AddInt64(&count, int64(1))
// 		time.Sleep(1 * time.Second)
// 	})

// 	server := createRequestManagementServer(
// 		delegate, flowSchemas, priorityLevelConfigurations,
// 		serverConcurrencyLimit, requestWaitLimit,
// 	)
// 	defer server.Close()

// 	time.Sleep(1 * time.Second)
// 	header := make(http.Header)
// 	header.Add("PRIORITY", "0")
// 	req, _ := http.NewRequest("GET", server.URL, nil)
// 	req.Header = header

// 	w := &Work{
// 		Request: req,
// 		N:       requests,
// 		C:       1000, // C > serverConcurrencyLimit - must be or else nothing
// 	}
//	// Run blocks until all work is done.
// 	w.Run()

// 	// TODO(aaron-prindle) make this more exact
// 	// perhaps check for timeout text in stdout/stderr
// 	if int64(requests) == count {
// 		t.Errorf("Expected some requests to timeout, recieved all requests %v/%v",
// 			request, count)
// 	}
// }
