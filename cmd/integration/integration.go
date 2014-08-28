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
	"reflect"
	"runtime"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

var (
	fakeDocker1, fakeDocker2 kubelet.FakeDockerClient
)

type fakePodInfoGetter struct{}

func (fakePodInfoGetter) GetPodInfo(host, podID string) (api.PodInfo, error) {
	// This is a horrible hack to get around the fact that we can't provide
	// different port numbers per kubelet...
	var c client.PodInfoGetter
	switch host {
	case "localhost":
		c = &client.HTTPPodInfoGetter{
			Client: http.DefaultClient,
			Port:   10250,
		}
	case "machine":
		c = &client.HTTPPodInfoGetter{
			Client: http.DefaultClient,
			Port:   10251,
		}
	default:
		glog.Fatalf("Can't get info for: '%v', '%v'", host, podID)
	}
	return c.GetPodInfo("localhost", podID)
}

type delegateHandler struct {
	delegate http.Handler
}

func (h *delegateHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if h.delegate != nil {
		h.delegate.ServeHTTP(w, req)
		return
	}
	w.WriteHeader(http.StatusNotFound)
}

func startComponents(manifestURL string) (apiServerURL string) {
	// Setup
	servers := []string{"http://localhost:4001"}
	glog.Infof("Creating etcd client pointing to %v", servers)
	machineList := []string{"localhost", "machine"}

	handler := delegateHandler{}
	apiServer := httptest.NewServer(&handler)

	etcdClient := etcd.NewClient(servers)
	keys, err := etcdClient.Get("/", false, false)
	if err != nil {
		glog.Fatalf("Unable to list root etcd keys: %v", err)
	}
	for _, node := range keys.Node.Nodes {
		if _, err := etcdClient.Delete(node.Key, true); err != nil {
			glog.Fatalf("Unable delete key: %v", err)
		}
	}

	cl := client.NewOrDie(apiServer.URL, nil)
	cl.PollPeriod = time.Second * 1
	cl.Sync = true

	// Master
	m := master.New(&master.Config{
		Client:        cl,
		EtcdServers:   servers,
		Minions:       machineList,
		PodInfoGetter: fakePodInfoGetter{},
	})
	storage, codec := m.API_v1beta1()
	handler.delegate = apiserver.Handle(storage, codec, "/api/v1beta1")

	// Scheduler
	scheduler.New((&factory.ConfigFactory{cl}).Create()).Run()

	controllerManager := controller.NewReplicationManager(cl)

	// Prove that controllerManager's watch works by making it not sync until after this
	// test is over. (Hopefully we don't take 10 minutes!)
	controllerManager.Run(10 * time.Minute)

	// Kubelet (localhost)
	cfg1 := config.NewPodConfig(config.PodConfigNotificationSnapshotAndUpdates)
	config.NewSourceEtcd(config.EtcdKeyForHost(machineList[0]), etcdClient, cfg1.Channel("etcd"))
	config.NewSourceURL(manifestURL, 5*time.Second, cfg1.Channel("url"))
	myKubelet := kubelet.NewIntegrationTestKubelet(machineList[0], &fakeDocker1)
	go util.Forever(func() { myKubelet.Run(cfg1.Updates()) }, 0)
	go util.Forever(func() {
		kubelet.ListenAndServeKubeletServer(myKubelet, cfg1.Channel("http"), "localhost", 10250)
	}, 0)

	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	cfg2 := config.NewPodConfig(config.PodConfigNotificationSnapshotAndUpdates)
	config.NewSourceEtcd(config.EtcdKeyForHost(machineList[1]), etcdClient, cfg2.Channel("etcd"))
	otherKubelet := kubelet.NewIntegrationTestKubelet(machineList[1], &fakeDocker2)
	go util.Forever(func() { otherKubelet.Run(cfg2.Updates()) }, 0)
	go util.Forever(func() {
		kubelet.ListenAndServeKubeletServer(otherKubelet, cfg2.Channel("http"), "localhost", 10251)
	}, 0)

	return apiServer.URL
}

// podsOnMinions returns true when all of the selected pods exist on a minion.
func podsOnMinions(c *client.Client, pods api.PodList) wait.ConditionFunc {
	podInfo := fakePodInfoGetter{}
	return func() (bool, error) {
		for i := range pods.Items {
			host, id := pods.Items[i].CurrentState.Host, pods.Items[i].ID
			if len(host) == 0 {
				return false, nil
			}
			if _, err := podInfo.GetPodInfo(host, id); err != nil {
				return false, nil
			}
		}
		return true, nil
	}
}

func runReplicationControllerTest(c *client.Client) {
	data, err := ioutil.ReadFile("api/examples/controller.json")
	if err != nil {
		glog.Fatalf("Unexpected error: %#v", err)
	}
	var controllerRequest api.ReplicationController
	if err := json.Unmarshal(data, &controllerRequest); err != nil {
		glog.Fatalf("Unexpected error: %#v", err)
	}

	glog.Infof("Creating replication controllers")
	if _, err := c.CreateReplicationController(controllerRequest); err != nil {
		glog.Fatalf("Unexpected error: %#v", err)
	}
	glog.Infof("Done creating replication controllers")

	// Give the controllers some time to actually create the pods
	if err := wait.Poll(time.Second, time.Second*10, c.ControllerHasDesiredReplicas(controllerRequest)); err != nil {
		glog.Fatalf("FAILED: pods never created %v", err)
	}

	// wait for minions to indicate they have info about the desired pods
	pods, err := c.ListPods(labels.Set(controllerRequest.DesiredState.ReplicaSelector).AsSelector())
	if err != nil {
		glog.Fatalf("FAILED: unable to get pods to list: %v", err)
	}
	if err := wait.Poll(time.Second, time.Second*10, podsOnMinions(c, pods)); err != nil {
		glog.Fatalf("FAILED: pods never started running %v", err)
	}

	glog.Infof("Pods created")
}

func runAtomicPutTest(c *client.Client) {
	var svc api.Service
	err := c.Post().Path("services").Body(
		api.Service{
			JSONBase: api.JSONBase{ID: "atomicservice", APIVersion: "v1beta1"},
			Port:     12345,
			Labels: map[string]string{
				"name": "atomicService",
			},
			// This is here because validation requires it.
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed creating atomicService: %v", err)
	}
	glog.Info("Created atomicService")
	testLabels := labels.Set{
		"foo": "bar",
	}
	for i := 0; i < 5; i++ {
		// a: z, b: y, etc...
		testLabels[string([]byte{byte('a' + i)})] = string([]byte{byte('z' - i)})
	}
	var wg sync.WaitGroup
	wg.Add(len(testLabels))
	for label, value := range testLabels {
		go func(l, v string) {
			for {
				glog.Infof("Starting to update (%s, %s)", l, v)
				var tmpSvc api.Service
				err := c.Get().
					Path("services").
					Path(svc.ID).
					PollPeriod(100 * time.Millisecond).
					Do().
					Into(&tmpSvc)
				if err != nil {
					glog.Errorf("Error getting atomicService: %v", err)
					continue
				}
				if tmpSvc.Selector == nil {
					tmpSvc.Selector = map[string]string{l: v}
				} else {
					tmpSvc.Selector[l] = v
				}
				glog.Infof("Posting update (%s, %s)", l, v)
				err = c.Put().Path("services").Path(svc.ID).Body(&tmpSvc).Do().Error()
				if err != nil {
					if se, ok := err.(*client.StatusErr); ok {
						if se.Status.Code == http.StatusConflict {
							glog.Infof("Conflict: (%s, %s)", l, v)
							// This is what we expect.
							continue
						}
					}
					glog.Errorf("Unexpected error putting atomicService: %v", err)
					continue
				}
				break
			}
			glog.Infof("Done update (%s, %s)", l, v)
			wg.Done()
		}(label, value)
	}
	wg.Wait()
	if err := c.Get().Path("services").Path(svc.ID).Do().Into(&svc); err != nil {
		glog.Fatalf("Failed getting atomicService after writers are complete: %v", err)
	}
	if !reflect.DeepEqual(testLabels, labels.Set(svc.Selector)) {
		glog.Fatalf("Selector PUTs were not atomic: wanted %v, got %v", testLabels, svc.Selector)
	}
	glog.Info("Atomic PUTs work.")
}

type testFunc func(*client.Client)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(3 * time.Minute)
		glog.Fatalf("This test has timed out.")
	}()

	manifestURL := ServeCachedManifestFile()

	apiServerURL := startComponents(manifestURL)

	// Ok. we're good to go.
	glog.Infof("API Server started on %s", apiServerURL)
	// Wait for the synchronization threads to come up.
	time.Sleep(time.Second * 10)

	kubeClient := client.NewOrDie(apiServerURL, nil)

	// Run tests in parallel
	testFuncs := []testFunc{
		runReplicationControllerTest,
		runAtomicPutTest,
	}
	var wg sync.WaitGroup
	wg.Add(len(testFuncs))
	for i := range testFuncs {
		f := testFuncs[i]
		go func() {
			f(kubeClient)
			wg.Done()
		}()
	}
	wg.Wait()

	// Check that kubelet tried to make the pods.
	// Using a set to list unique creation attempts. Our fake is
	// really stupid, so kubelet tries to create these multiple times.
	createdPods := util.StringSet{}
	for _, p := range fakeDocker1.Created {
		// The last 8 characters are random, so slice them off.
		if n := len(p); n > 8 {
			createdPods.Insert(p[:n-8])
		}
	}
	for _, p := range fakeDocker2.Created {
		// The last 8 characters are random, so slice them off.
		if n := len(p); n > 8 {
			createdPods.Insert(p[:n-8])
		}
	}
	// We expect 5: 2 net containers + 2 pods from the replication controller +
	//              1 net container + 2 pods from the URL.
	if len(createdPods) != 7 {
		glog.Fatalf("Unexpected list of created pods:\n\n%#v\n\n%#v\n\n%#v\n\n", createdPods.List(), fakeDocker1.Created, fakeDocker2.Created)
	}
	glog.Infof("OK - found created pods: %#v", createdPods.List())
}

// ServeCachedManifestFile serves a file for kubelet to read.
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
	testManifestFile = `version: v1beta2
id: container-vm-guestbook
containers:
  - name: redis
    image: dockerfile/redis
    volumeMounts:
      - name: redis-data
        mountPath: /data

  - name: guestbook
    image: google/guestbook-python-redis
    ports:
      - name: www
        hostPort: 80
        containerPort: 80

volumes:
  - name: redis-data`
)
