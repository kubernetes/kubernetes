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

package dynamic

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/apiserver/pkg/audit"
	webhook "k8s.io/apiserver/pkg/util/webhook"
	informers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
)

func TestDynamic(t *testing.T) {
	eventList1 := &atomic.Value{}
	eventList1.Store(auditinternal.EventList{})
	eventList2 := &atomic.Value{}
	eventList2.Store(auditinternal.EventList{})

	// start test servers
	server1 := httptest.NewServer(buildTestHandler(t, eventList1))
	defer server1.Close()
	server2 := httptest.NewServer(buildTestHandler(t, eventList2))
	defer server2.Close()

	testPolicy := auditregv1alpha1.Policy{
		Level: auditregv1alpha1.LevelMetadata,
		Stages: []auditregv1alpha1.Stage{
			auditregv1alpha1.StageResponseStarted,
		},
	}
	testEvent := auditinternal.Event{
		Level:      auditinternal.LevelMetadata,
		Stage:      auditinternal.StageResponseStarted,
		Verb:       "get",
		RequestURI: "/test/path",
	}
	testConfig1 := &auditregv1alpha1.AuditSink{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test1",
			UID:  types.UID("test1"),
		},
		Spec: auditregv1alpha1.AuditSinkSpec{
			Policy: testPolicy,
			Webhook: auditregv1alpha1.Webhook{
				ClientConfig: auditregv1alpha1.WebhookClientConfig{
					URL: &server1.URL,
				},
			},
		},
	}
	testConfig2 := &auditregv1alpha1.AuditSink{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test2",
			UID:  types.UID("test2"),
		},
		Spec: auditregv1alpha1.AuditSinkSpec{
			Policy: testPolicy,
			Webhook: auditregv1alpha1.Webhook{
				ClientConfig: auditregv1alpha1.WebhookClientConfig{
					URL: &server2.URL,
				},
			},
		},
	}

	config, stopChan := defaultTestConfig()
	config.BufferedConfig.MaxBatchWait = 10 * time.Millisecond

	b, err := NewBackend(config)
	require.NoError(t, err)
	d := b.(*backend)
	err = b.Run(stopChan)
	require.NoError(t, err)

	t.Run("find none", func(t *testing.T) {
		require.Len(t, d.GetDelegates(), 0)
	})

	t.Run("find one", func(t *testing.T) {
		d.addSink(testConfig1)
		delegates := d.GetDelegates()
		require.Len(t, delegates, 1)
		require.Contains(t, delegates, types.UID("test1"))
		require.Equal(t, testConfig1, delegates["test1"].configuration)

		// send event and check that it arrives
		b.ProcessEvents(&testEvent)
		err := checkForEvent(eventList1, testEvent)
		require.NoError(t, err, "unable to find events sent to sink")
	})

	t.Run("find two", func(t *testing.T) {
		eventList1.Store(auditinternal.EventList{})
		d.addSink(testConfig2)
		delegates := d.GetDelegates()
		require.Len(t, delegates, 2)
		require.Contains(t, delegates, types.UID("test1"))
		require.Contains(t, delegates, types.UID("test2"))
		require.Equal(t, testConfig1, delegates["test1"].configuration)
		require.Equal(t, testConfig2, delegates["test2"].configuration)

		// send event to both delegates and check that it arrives in both places
		b.ProcessEvents(&testEvent)
		err := checkForEvent(eventList1, testEvent)
		require.NoError(t, err, "unable to find events sent to sink 1")
		err = checkForEvent(eventList2, testEvent)
		require.NoError(t, err, "unable to find events sent to sink 2")
	})

	t.Run("delete one", func(t *testing.T) {
		eventList2.Store(auditinternal.EventList{})
		d.deleteSink(testConfig1)
		delegates := d.GetDelegates()
		require.Len(t, delegates, 1)
		require.Contains(t, delegates, types.UID("test2"))
		require.Equal(t, testConfig2, delegates["test2"].configuration)

		// send event and check that it arrives to remaining sink
		b.ProcessEvents(&testEvent)
		err := checkForEvent(eventList2, testEvent)
		require.NoError(t, err, "unable to find events sent to sink")
	})

	t.Run("update one", func(t *testing.T) {
		eventList1.Store(auditinternal.EventList{})
		oldConfig := *testConfig2
		testConfig2.Spec.Webhook.ClientConfig.URL = &server1.URL
		testConfig2.UID = types.UID("test2.1")
		d.updateSink(&oldConfig, testConfig2)
		delegates := d.GetDelegates()
		require.Len(t, delegates, 1)
		require.Contains(t, delegates, types.UID("test2.1"))
		require.Equal(t, testConfig2, delegates["test2.1"].configuration)

		// send event and check that it arrives to updated sink
		b.ProcessEvents(&testEvent)
		err := checkForEvent(eventList1, testEvent)
		require.NoError(t, err, "unable to find events sent to sink")
	})

	t.Run("update meta only", func(t *testing.T) {
		eventList1.Store(auditinternal.EventList{})
		oldConfig := *testConfig2
		testConfig2.UID = types.UID("test2.2")
		testConfig2.Labels = map[string]string{"my": "label"}
		d.updateSink(&oldConfig, testConfig2)
		delegates := d.GetDelegates()
		require.Len(t, delegates, 1)
		require.Contains(t, delegates, types.UID("test2.2"))

		// send event and check that it arrives to same sink
		b.ProcessEvents(&testEvent)
		err := checkForEvent(eventList1, testEvent)
		require.NoError(t, err, "unable to find events sent to sink")
	})

	t.Run("shutdown", func(t *testing.T) {
		// if the stop signal is not propagated correctly the buffers will not
		// close down gracefully, and the shutdown method will hang causing
		// the test will timeout.
		timeoutChan := make(chan struct{})
		successChan := make(chan struct{})
		go func() {
			time.Sleep(1 * time.Second)
			timeoutChan <- struct{}{}
		}()
		go func() {
			close(stopChan)
			d.Shutdown()
			successChan <- struct{}{}
		}()
		for {
			select {
			case <-timeoutChan:
				t.Error("shutdown timed out")
				return
			case <-successChan:
				return
			}
		}
	})
}

// checkForEvent will poll to check for an audit event in an atomic event list
func checkForEvent(a *atomic.Value, evSent auditinternal.Event) error {
	return wait.Poll(100*time.Millisecond, 1*time.Second, func() (bool, error) {
		el := a.Load().(auditinternal.EventList)
		if len(el.Items) != 1 {
			return false, nil
		}
		evFound := el.Items[0]
		eq := reflect.DeepEqual(evSent, evFound)
		if !eq {
			return false, fmt.Errorf("event mismatch -- sent: %+v found: %+v", evSent, evFound)
		}
		return true, nil
	})
}

// buildTestHandler returns a handler that will update the atomic value passed in
// with the event list it receives
func buildTestHandler(t *testing.T, a *atomic.Value) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("could not read request body: %v", err)
		}
		el := auditinternal.EventList{}
		decoder := audit.Codecs.UniversalDecoder(auditv1.SchemeGroupVersion)
		if err := runtime.DecodeInto(decoder, body, &el); err != nil {
			t.Fatalf("failed decoding buf: %b, apiVersion: %s", body, auditv1.SchemeGroupVersion)
		}
		defer r.Body.Close()
		a.Store(el)
		w.WriteHeader(200)
	})
}

// defaultTestConfig returns a Config object suitable for testing along with its
// associated stopChan
func defaultTestConfig() (*Config, chan struct{}) {
	authWrapper := webhook.AuthenticationInfoResolverWrapper(
		func(a webhook.AuthenticationInfoResolver) webhook.AuthenticationInfoResolver { return a },
	)
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	stop := make(chan struct{})

	eventSink := &v1core.EventSinkImpl{Interface: client.CoreV1().Events("")}

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)
	informer := informerFactory.Auditregistration().V1alpha1().AuditSinks()
	return &Config{
		Informer:       informer,
		EventConfig:    EventConfig{Sink: eventSink},
		BufferedConfig: NewDefaultWebhookBatchConfig(),
		WebhookConfig: WebhookConfig{
			AuthInfoResolverWrapper: authWrapper,
			ServiceResolver:         webhook.NewDefaultServiceResolver(),
		},
	}, stop
}
