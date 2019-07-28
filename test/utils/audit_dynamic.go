/*
Copyright 2018 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/apiserver/pkg/audit"
)

// AuditTestServer is a helper server for dynamic audit testing
type AuditTestServer struct {
	Name            string
	LockedEventList *LockedEventList
	Server          *httptest.Server
	t               *testing.T
}

// LockedEventList is an event list with a lock for concurrent access
type LockedEventList struct {
	*sync.RWMutex
	EventList auditinternal.EventList
}

// NewLockedEventList returns a new LockedEventList
func NewLockedEventList() *LockedEventList {
	return &LockedEventList{
		RWMutex:   &sync.RWMutex{},
		EventList: auditinternal.EventList{},
	}
}

// NewAuditTestServer returns a new audit test server
func NewAuditTestServer(t *testing.T, name string) *AuditTestServer {
	s := &AuditTestServer{
		Name:            name,
		LockedEventList: NewLockedEventList(),
		t:               t,
	}
	s.buildServer()
	return s
}

// GetEventList safely returns the internal event list
func (a *AuditTestServer) GetEventList() auditinternal.EventList {
	a.LockedEventList.RLock()
	defer a.LockedEventList.RUnlock()
	return a.LockedEventList.EventList
}

// ResetEventList resets the internal event list
func (a *AuditTestServer) ResetEventList() {
	a.LockedEventList.Lock()
	defer a.LockedEventList.Unlock()
	a.LockedEventList.EventList = auditinternal.EventList{}
}

// AppendEvents will add the given events to the internal event list
func (a *AuditTestServer) AppendEvents(events []auditinternal.Event) {
	a.LockedEventList.Lock()
	defer a.LockedEventList.Unlock()
	a.LockedEventList.EventList.Items = append(a.LockedEventList.EventList.Items, events...)
}

// WaitForEvents waits for the given events to arrive in the server or the 30s timeout is reached
func (a *AuditTestServer) WaitForEvents(expected []AuditEvent) ([]AuditEvent, error) {
	var missing []AuditEvent
	err := wait.PollImmediate(50*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		var err error
		a.LockedEventList.RLock()
		defer a.LockedEventList.RUnlock()
		el := a.GetEventList()
		if len(el.Items) < 1 {
			return false, nil
		}
		missing, err = CheckAuditList(el, expected)
		if err != nil {
			return false, nil
		}
		return true, nil
	})
	return missing, err
}

// WaitForNumEvents checks that at least the given number of events has arrived or the 30s timeout is reached
func (a *AuditTestServer) WaitForNumEvents(numEvents int) error {
	err := wait.PollImmediate(50*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		el := a.GetEventList()
		if len(el.Items) < numEvents {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("%v: %d events failed to arrive in %v", err, numEvents, wait.ForeverTestTimeout)
	}
	return nil
}

// Health polls the server healthcheck until successful or the 30s timeout has been reached
func (a *AuditTestServer) Health() error {
	err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		resp, err := http.Get(fmt.Sprintf("%s/health", a.Server.URL))
		if err != nil {
			return false, nil
		}
		if resp.StatusCode != 200 {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("server %s permanently failed health check: %v", a.Server.URL, err)
	}
	return nil
}

// Close the server
func (a *AuditTestServer) Close() {
	a.Server.Close()
}

// BuildSinkConfiguration creates a generic audit sink configuration for this server
func (a *AuditTestServer) BuildSinkConfiguration() *auditregv1alpha1.AuditSink {
	return &auditregv1alpha1.AuditSink{
		ObjectMeta: metav1.ObjectMeta{
			Name: a.Name,
		},
		Spec: auditregv1alpha1.AuditSinkSpec{
			Policy: auditregv1alpha1.Policy{
				Level: auditregv1alpha1.LevelRequestResponse,
				Stages: []auditregv1alpha1.Stage{
					auditregv1alpha1.StageResponseStarted,
					auditregv1alpha1.StageResponseComplete,
				},
			},
			Webhook: auditregv1alpha1.Webhook{
				ClientConfig: auditregv1alpha1.WebhookClientConfig{
					URL: &a.Server.URL,
				},
			},
		},
	}
}

// buildServer creates an http test server that will update the internal event list
// with the value it receives
func (a *AuditTestServer) buildServer() {
	decoder := audit.Codecs.UniversalDecoder(auditv1.SchemeGroupVersion)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		body, err := ioutil.ReadAll(r.Body)
		require.NoError(a.t, err, "could not read request body")
		el := auditinternal.EventList{}
		err = runtime.DecodeInto(decoder, body, &el)
		r.Body.Close()
		require.NoError(a.t, err, "failed decoding buf: %b, apiVersion: %s", body, auditv1.SchemeGroupVersion)
		a.AppendEvents(el.Items)
		w.WriteHeader(200)
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	})
	a.Server = httptest.NewServer(mux)
}
