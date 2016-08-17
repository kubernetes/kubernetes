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

package benchmark

import (
	"net/http"
	"net/http/httptest"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	e2e "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/integration/framework"
)

// mustSetupScheduler starts the following components:
// - k8s api server (a.k.a. master)
// - scheduler
// It returns scheduler config factory and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupScheduler() (schedulerConfigFactory *factory.ConfigFactory, destroyFunc func()) {
	framework.DeleteAllEtcdKeys()

	var m *master.Master
	masterConfig := framework.NewIntegrationTestMasterConfig()
	m, err := master.New(masterConfig)
	if err != nil {
		panic("error in brining up the master: " + err.Error())
	}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))

	c := client.NewOrDie(&restclient.Config{
		Host:          s.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()},
		QPS:           5000.0,
		Burst:         5000,
	})

	schedulerConfigFactory = factory.NewConfigFactory(c, api.DefaultSchedulerName, api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		panic("Couldn't create scheduler config")
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"})
	eventBroadcaster.StartRecordingToSink(c.Events(""))
	scheduler.New(schedulerConfig).Run()

	destroyFunc = func() {
		glog.Infof("destroying")
		close(schedulerConfig.StopEverything)
		s.Close()
		glog.Infof("destroyed")
	}
	return
}

func makeNodes(c client.Interface, nodeCount int) {
	glog.Infof("making %d nodes", nodeCount)
	baseNode := &api.Node{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "scheduler-test-node-",
		},
		Spec: api.NodeSpec{
			ExternalID: "foobar",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				api.ResourceCPU:    resource.MustParse("4"),
				api.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: api.NodeRunning,
			Conditions: []api.NodeCondition{
				{Type: api.NodeReady, Status: api.ConditionTrue},
			},
		},
	}
	for i := 0; i < nodeCount; i++ {
		if _, err := c.Nodes().Create(baseNode); err != nil {
			panic("error creating node: " + err.Error())
		}
	}
}

func makePodSpec() api.PodSpec {
	return api.PodSpec{
		Containers: []api.Container{{
			Name:  "pause",
			Image: e2e.GetPauseImageNameForHostArch(),
			Ports: []api.ContainerPort{{ContainerPort: 80}},
			Resources: api.ResourceRequirements{
				Limits: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("100m"),
					api.ResourceMemory: resource.MustParse("500Mi"),
				},
				Requests: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("100m"),
					api.ResourceMemory: resource.MustParse("500Mi"),
				},
			},
		}},
	}
}

// makePodsFromRC will create a ReplicationController object and
// a given number of pods (imitating the controller).
func makePodsFromRC(c client.Interface, name string, podCount int) {
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: int32(podCount),
			Selector: map[string]string{"name": name},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: makePodSpec(),
			},
		},
	}
	if _, err := c.ReplicationControllers("default").Create(rc); err != nil {
		glog.Fatalf("unexpected error: %v", err)
	}

	basePod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "scheduler-test-pod-",
			Labels:       map[string]string{"name": name},
		},
		Spec: makePodSpec(),
	}
	createPod := func(i int) {
		for {
			if _, err := c.Pods("default").Create(basePod); err == nil {
				break
			}
		}
	}
	workqueue.Parallelize(30, podCount, createPod)
}
