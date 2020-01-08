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

package master

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
)

// TestDynamicAudit ensures that v1alpha of the auditregistration api works
func TestDynamicAudit(t *testing.T) {
	// start api server
	stopCh := make(chan struct{})
	defer close(stopCh)

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicAuditing, true)()
	kubeclient, _ := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Audit.DynamicOptions.Enabled = true
			// set max batch size so the buffers flush immediately
			opts.Audit.DynamicOptions.BatchConfig.MaxBatchSize = 1
			opts.APIEnablement.RuntimeConfig.Set("auditregistration.k8s.io/v1alpha1=true")
		},
	})

	// create test sinks
	testServer1 := utils.NewAuditTestServer(t, "test1")
	defer testServer1.Close()
	testServer2 := utils.NewAuditTestServer(t, "test2")
	defer testServer2.Close()

	// check that servers are healthy
	require.NoError(t, testServer1.Health(), "server1 never became healthy")
	require.NoError(t, testServer2.Health(), "server2 never became healthy")

	// build AuditSink configurations
	sinkConfig1 := testServer1.BuildSinkConfiguration()
	sinkConfig2 := testServer2.BuildSinkConfiguration()

	// test creates a single audit sink, generates audit events, and ensures they arrive at the server
	success := t.Run("one sink", func(t *testing.T) {
		_, err := kubeclient.AuditregistrationV1alpha1().AuditSinks().Create(sinkConfig1)
		require.NoError(t, err, "failed to create audit sink1")
		t.Log("created audit sink1")

		// verify sink is ready
		sinkHealth(t, kubeclient, testServer1)

		// perform configmap ops
		configMapOperations(t, kubeclient)

		// check for corresponding events
		missing, err := testServer1.WaitForEvents(expectedEvents)
		require.NoError(t, err, "failed to match all expected events for server1, events %#v not found", missing)
	})
	require.True(t, success) // propagate failure

	// test creates a second audit sink, generates audit events, and ensures events arrive in both servers
	success = t.Run("two sink", func(t *testing.T) {
		_, err := kubeclient.AuditregistrationV1alpha1().AuditSinks().Create(sinkConfig2)
		require.NoError(t, err, "failed to create audit sink2")
		t.Log("created audit sink2")

		// verify both sinks are ready
		sinkHealth(t, kubeclient, testServer1, testServer2)

		// perform configmap ops
		configMapOperations(t, kubeclient)

		// check for corresponding events in both sinks
		missing, err := testServer1.WaitForEvents(expectedEvents)
		require.NoError(t, err, "failed to match all expected events for server1, events %#v not found", missing)
		missing, err = testServer2.WaitForEvents(expectedEvents)
		require.NoError(t, err, "failed to match all expected events for server2, events %#v not found", missing)
	})
	require.True(t, success) // propagate failure

	// test deletes an audit sink, generates audit events, and ensures they don't arrive in the corresponding server
	success = t.Run("delete sink", func(t *testing.T) {
		err := kubeclient.AuditregistrationV1alpha1().AuditSinks().Delete(sinkConfig2.Name, &metav1.DeleteOptions{})
		require.NoError(t, err, "failed to delete audit sink2")
		t.Log("deleted audit sink2")

		var finalErr error
		err = wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
			// reset event lists
			testServer1.ResetEventList()
			testServer2.ResetEventList()

			// perform configmap ops
			configMapOperations(t, kubeclient)

			// check for corresponding events in server1
			missing, err := testServer1.WaitForEvents(expectedEvents)
			if err != nil {
				finalErr = fmt.Errorf("%v: failed to match all expected events for server1, events %#v not found", err, missing)
				return false, nil
			}

			// check that server2 is empty
			if len(testServer2.GetEventList().Items) != 0 {
				finalErr = fmt.Errorf("server2 event list should be empty")
				return false, nil
			}
			return true, nil
		})
		require.NoError(t, err, finalErr)
	})
	require.True(t, success) // propagate failure

	// This test will run a background process that generates audit events sending them to a sink.
	// Whilst that generation is occurring, the sink is updated to point to a different server.
	// The test checks that no events are lost or duplicated during the update.
	t.Run("update sink", func(t *testing.T) {
		// fetch sink1 config
		sink1, err := kubeclient.AuditregistrationV1alpha1().AuditSinks().Get(sinkConfig1.Name, metav1.GetOptions{})
		require.NoError(t, err)

		// reset event lists
		testServer1.ResetEventList()
		testServer2.ResetEventList()

		// run operations in background
		stopChan := make(chan struct{})
		expectedEvents := &atomic.Value{}
		expectedEvents.Store([]utils.AuditEvent{})
		wg := &sync.WaitGroup{}
		wg.Add(1)
		go asyncOps(stopChan, wg, kubeclient, expectedEvents)

		// check to see that at least 20 events have arrived in server1
		err = testServer1.WaitForNumEvents(20)
		require.NoError(t, err, "failed to find enough events in server1")

		// check that no events are in server 2 yet
		require.Len(t, testServer2.GetEventList().Items, 0, "server2 should not have events yet")

		// update the url
		sink1.Spec.Webhook.ClientConfig.URL = &testServer2.Server.URL
		_, err = kubeclient.AuditregistrationV1alpha1().AuditSinks().Update(sink1)
		require.NoError(t, err, "failed to update audit sink1")
		t.Log("updated audit sink1 to point to server2")

		// check that at least 20 events have arrived in server2
		err = testServer2.WaitForNumEvents(20)
		require.NoError(t, err, "failed to find enough events in server2")

		// stop the operations and ensure they have finished
		close(stopChan)
		wg.Wait()

		// check that the final events have arrived
		expected := expectedEvents.Load().([]utils.AuditEvent)
		missing, err := testServer2.WaitForEvents(expected[len(expected)-4:])
		require.NoError(t, err, "failed to find the final events in server2, events %#v not found", missing)

		// combine the event lists
		el1 := testServer1.GetEventList()
		el2 := testServer2.GetEventList()
		combinedList := auditinternal.EventList{}
		combinedList.Items = append(el1.Items, el2.Items...)

		// check that there are no duplicate events
		dups, err := utils.CheckForDuplicates(combinedList)
		require.NoError(t, err, "duplicate events found: %#v", dups)

		// check that no events are missing
		missing, err = utils.CheckAuditList(combinedList, expected)
		require.NoError(t, err, "failed to match all expected events: %#v not found", missing)
	})
}

// sinkHealth checks if sinks are running by verifying that uniquely identified events are found
// in the given servers
func sinkHealth(t *testing.T, kubeclient kubernetes.Interface, servers ...*utils.AuditTestServer) {
	var missing []utils.AuditEvent
	i := 0
	var finalErr error
	err := wait.PollImmediate(50*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		i++
		name := fmt.Sprintf("health-%d-%d", i, time.Now().UnixNano())
		expected, err := simpleOp(name, kubeclient)
		require.NoError(t, err, "could not perform config map operations")

		// check that all given servers have received events
		for _, server := range servers {
			missing, err = server.WaitForEvents(expected)
			if err != nil {
				finalErr = fmt.Errorf("not all events found in %s health check: missing %#v", server.Name, missing)
				return false, nil
			}
			server.ResetEventList()
		}
		return true, nil
	})
	require.NoError(t, err, finalErr)
}

// simpleOp is a function that simply tries to get a configmap with the given name and returns the
// corresponding expected audit event
func simpleOp(name string, kubeclient kubernetes.Interface) ([]utils.AuditEvent, error) {
	_, err := kubeclient.CoreV1().ConfigMaps(namespace).Get(name, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}

	expectedEvents := []utils.AuditEvent{
		{
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/%s", namespace, name),
			Verb:              "get",
			Code:              404,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     false,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		},
	}
	return expectedEvents, nil
}

// asyncOps runs the simpleOp function until the stopChan is closed updating
// the expected atomic events list
func asyncOps(
	stopChan <-chan struct{},
	wg *sync.WaitGroup,
	kubeclient kubernetes.Interface,
	expected *atomic.Value,
) {
	for i := 0; ; i++ {
		select {
		case <-stopChan:
			wg.Done()
			return
		default:
			name := fmt.Sprintf("health-%d-%d", i, time.Now().UnixNano())
			exp, err := simpleOp(name, kubeclient)
			if err != nil {
				// retry on errors
				continue
			}
			e := expected.Load().([]utils.AuditEvent)
			evList := append(e, exp...)
			expected.Store(evList)
		}
	}
}
