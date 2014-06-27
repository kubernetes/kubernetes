/*
Copyright 2014 Google Inc. All rights reserved.

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

// A basic integration test for the service.
// Assumes that there is a pre-existing etcd server running on localhost.
package main

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"runtime"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

func main() {
	runtime.GOMAXPROCS(4)
	util.InitLogs()
	defer util.FlushLogs()

	manifestUrl := ServeCachedManifestFile()
	// Setup
	servers := []string{"http://localhost:4001"}
	glog.Infof("Creating etcd client pointing to %v", servers)
	machineList := []string{"localhost", "machine"}

	// Master
	m := master.New(servers, machineList, nil)
	apiserver := httptest.NewServer(m.ConstructHandler("/api/v1beta1"))

	controllerManager := controller.MakeReplicationManager(etcd.NewClient(servers), client.New(apiserver.URL, nil))

	controllerManager.Run(1 * time.Second)

	// Kublet
	fakeDocker1 := &kubelet.FakeDockerClient{}
	myKubelet := kubelet.Kubelet{
		Hostname:           machineList[0],
		DockerClient:       fakeDocker1,
		DockerPuller:       &kubelet.FakeDockerPuller{},
		FileCheckFrequency: 5 * time.Second,
		SyncFrequency:      5 * time.Second,
		HTTPCheckFrequency: 5 * time.Second,
	}
	go myKubelet.RunKubelet("", manifestUrl, servers[0], "localhost", 0)

	// Create a second kublet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	fakeDocker2 := &kubelet.FakeDockerClient{}
	otherKubelet := kubelet.Kubelet{
		Hostname:           machineList[1],
		DockerClient:       fakeDocker2,
		DockerPuller:       &kubelet.FakeDockerPuller{},
		FileCheckFrequency: 5 * time.Second,
		SyncFrequency:      5 * time.Second,
		HTTPCheckFrequency: 5 * time.Second,
	}
	go otherKubelet.RunKubelet("", "", servers[0], "localhost", 0)

	// Ok. we're good to go.
	glog.Infof("API Server started on %s", apiserver.URL)
	// Wait for the synchronization threads to come up.
	time.Sleep(time.Second * 10)

	kubeClient := client.New(apiserver.URL, nil)
	data, err := ioutil.ReadFile("api/examples/controller.json")
	if err != nil {
		glog.Fatalf("Unexpected error: %#v", err)
	}
	var controllerRequest api.ReplicationController
	if err = json.Unmarshal(data, &controllerRequest); err != nil {
		glog.Fatalf("Unexpected error: %#v", err)
	}

	if _, err = kubeClient.CreateReplicationController(controllerRequest); err != nil {
		glog.Fatalf("Unexpected error: %#v", err)
	}
	// Give the controllers some time to actually create the pods
	time.Sleep(time.Second * 10)

	// Validate that they're truly up.
	pods, err := kubeClient.ListPods(nil)
	if err != nil || len(pods.Items) != 2 {
		glog.Fatal("FAILED: %#v", pods.Items)
	}

	// Check that kubelet tried to make the pods.
	// Using a set to list unique creation attempts. Our fake is
	// really stupid, so kubelet tries to create these multiple times.
	createdPods := map[string]struct{}{}
	for _, p := range fakeDocker1.Created {
		// The last 8 characters are random, so slice them off.
		if n := len(p); n > 8 {
			createdPods[p[:n-8]] = struct{}{}
		}
	}
	for _, p := range fakeDocker2.Created {
		// The last 8 characters are random, so slice them off.
		if n := len(p); n > 8 {
			createdPods[p[:n-8]] = struct{}{}
		}
	}
	// We expect 5: 2 net containers + 2 pods from the replication controller +
	//              1 net container + 2 pods from the URL.
	if len(createdPods) != 7 {
		glog.Fatalf("Unexpected list of created pods: %#v %#v %#v\n", createdPods, fakeDocker1.Created, fakeDocker2.Created)
	}
	glog.Infof("OK")
}

// Serve a file for kubelet to read.
func ServeCachedManifestFile() (servingAddress string) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manifest" {
			w.Write([]byte(testManifestFile))
			return
		}
		glog.Fatalf("Got request: %#v\n", r)
		http.NotFound(w, r)
	}))
	return server.URL + "/manifest"
}

const (
	// This is copied from, and should be kept in sync with:
	// https://raw.githubusercontent.com/GoogleCloudPlatform/container-vm-guestbook-redis-python/master/manifest.yaml
	testManifestFile = `version: v1beta1
id: web-test
containers:
  - name: redis
    image: dockerfile/redis
    volumeMounts:
      - name: redis-data
        path: /data

  - name: guestbook
    image: google/guestbook-python-redis
    ports:
      - name: www
        hostPort: 80
        containerPort: 80

volumes:
  - name: redis-data`
)
