// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package integration

// This file tests the scheduler.

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"
)

func init() {
	requireEtcd()
}

func TestUnschedulableNodes(t *testing.T) {
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("Couldn't create etcd helper: %v", err)
	}
	deleteAllEtcdKeys()

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		EnableIndex:       true,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	client := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Version()})

	schedulerConfigFactory := factory.NewConfigFactory(client)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		t.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"})
	eventBroadcaster.StartRecordingToSink(client.Events(""))
	scheduler.New(schedulerConfig).Run()

	defer close(schedulerConfig.StopEverything)

	DoTestUnschedulableNodes(t, client)
}

func podScheduled(c *client.Client, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.Pods(podNamespace).Get(podName)
		if errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		if pod.Spec.Host == "" {
			return false, nil
		}
		return true, nil
	}
}

func DoTestUnschedulableNodes(t *testing.T, client *client.Client) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "node"},
		Spec:       api.NodeSpec{Unschedulable: true},
	}
	if _, err := client.Nodes().Create(node); err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "my-pod"},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "container", Image: "kubernetes/pause:go"}},
		},
	}
	myPod, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}
	// There are no schedulable nodes - the pod shouldn't be scheduled.
	err = wait.Poll(time.Second, time.Second*10, podScheduled(client, myPod.Namespace, myPod.Name))
	if err == nil {
		t.Errorf("Pod scheduled successfully on unschedulable nodes")
	}
	if err != wait.ErrWaitTimeout {
		t.Errorf("Failed while waiting for scheduled pod: %v", err)
	}

	// Make the node schedulable and wait until the pod is scheduled.
	newNode, err := client.Nodes().Get(node.Name)
	if err != nil {
		t.Fatalf("Failed to get node: %v", err)
	}
	newNode.Spec.Unschedulable = false
	if _, err = client.Nodes().Update(newNode); err != nil {
		t.Fatalf("Failed to update node: %v", err)
	}
	err = wait.Poll(time.Second, time.Second*10, podScheduled(client, myPod.Namespace, myPod.Name))
	if err != nil {
		t.Errorf("Failed to schedule a pod: %v", err)
	}

	err = client.Pods(api.NamespaceDefault).Delete(myPod.Name, nil)
	if err != nil {
		t.Errorf("Failed to delete pod: %v", err)
	}
}
