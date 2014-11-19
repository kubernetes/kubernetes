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
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	minionControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	replicationControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/standalone"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

const testRootDir = "/tmp/kubelet"
const testRootDir2 = "/tmp/kubelet2"

var (
	fakeDocker1, fakeDocker2 dockertools.FakeDockerClient
)

type fakeKubeletClient struct{}

func (fakeKubeletClient) GetPodInfo(host, podNamespace, podID string) (api.PodInfo, error) {
	// This is a horrible hack to get around the fact that we can't provide
	// different port numbers per kubelet...
	var c client.PodInfoGetter
	switch host {
	case "localhost":
		c = &client.HTTPKubeletClient{
			Client: http.DefaultClient,
			Port:   10250,
		}
	case "machine":
		c = &client.HTTPKubeletClient{
			Client: http.DefaultClient,
			Port:   10251,
		}
	default:
		glog.Fatalf("Can't get info for: '%v', '%v - %v'", host, podNamespace, podID)
	}
	return c.GetPodInfo("localhost", podNamespace, podID)
}

func (fakeKubeletClient) HealthCheck(host string) (health.Status, error) {
	return health.Healthy, nil
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
	sleep := 4 * time.Second
	ok := false
	for i := 0; i < 3; i++ {
		keys, err := etcdClient.Get("/", false, false)
		if err != nil {
			glog.Warningf("Unable to list root etcd keys: %v", err)
			if i < 2 {
				time.Sleep(sleep)
				sleep = sleep * sleep
			}
			continue
		}
		for _, node := range keys.Node.Nodes {
			if _, err := etcdClient.Delete(node.Key, true); err != nil {
				glog.Fatalf("Unable delete key: %v", err)
			}
		}
		ok = true
		break
	}
	if !ok {
		glog.Fatalf("Failed to connect to etcd")
	}

	cl := client.NewOrDie(&client.Config{Host: apiServer.URL, Version: testapi.Version()})
	cl.PollPeriod = time.Millisecond * 100
	cl.Sync = true

	helper, err := master.NewEtcdHelper(etcdClient, "")
	if err != nil {
		glog.Fatalf("Unable to get etcd helper: %v", err)
	}

	// Master
	host, port, err := net.SplitHostPort(strings.TrimLeft(apiServer.URL, "http://"))
	if err != nil {
		glog.Fatalf("Unable to parse URL '%v': %v", apiServer.URL, err)
	}
	portNumber, err := strconv.Atoi(port)
	if err != nil {
		glog.Fatalf("Nonnumeric port? %v", err)
	}

	// Create a master and install handlers into mux.
	m := master.New(&master.Config{
		Client:            cl,
		EtcdHelper:        helper,
		KubeletClient:     fakeKubeletClient{},
		EnableLogsSupport: false,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysAllowAuthorizer(),

		ReadWritePort: portNumber,
		ReadOnlyPort:  portNumber,
		PublicAddress: host,
	})
	handler.delegate = m.Handler

	// Scheduler
	schedulerConfigFactory := factory.NewConfigFactory(cl)
	schedulerConfig, err := schedulerConfigFactory.Create(nil, nil)
	if err != nil {
		glog.Fatal("Couldn't create scheduler config: %v", err)
	}
	scheduler.New(schedulerConfig).Run()

	endpoints := service.NewEndpointController(cl)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	controllerManager := replicationControllerPkg.NewReplicationManager(cl)

	// Prove that controllerManager's watch works by making it not sync until after this
	// test is over. (Hopefully we don't take 10 minutes!)
	controllerManager.Run(10 * time.Minute)

	nodeResources := &api.NodeResources{}
	minionController := minionControllerPkg.NewMinionController(nil, "", machineList, nodeResources, cl)
	minionController.Run(10 * time.Second)

	// Kubelet (localhost)
	standalone.SimpleRunKubelet(etcdClient, &fakeDocker1, machineList[0], testRootDir, manifestURL, "127.0.0.1", 10250)
	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	standalone.SimpleRunKubelet(etcdClient, &fakeDocker2, machineList[1], testRootDir2, "", "127.0.0.1", 10251)

	return apiServer.URL
}

// podsOnMinions returns true when all of the selected pods exist on a minion.
func podsOnMinions(c *client.Client, pods api.PodList) wait.ConditionFunc {
	podInfo := fakeKubeletClient{}
	return func() (bool, error) {
		for i := range pods.Items {
			host, id, namespace := pods.Items[i].Status.Host, pods.Items[i].Name, pods.Items[i].Namespace
			if len(host) == 0 {
				return false, nil
			}
			if _, err := podInfo.GetPodInfo(host, namespace, id); err != nil {
				return false, nil
			}
		}
		return true, nil
	}
}

func endpointsSet(c *client.Client, serviceNamespace, serviceID string, endpointCount int) wait.ConditionFunc {
	return func() (bool, error) {
		endpoints, err := c.Endpoints(serviceNamespace).Get(serviceID)
		if err != nil {
			return false, nil
		}
		return len(endpoints.Endpoints) == endpointCount, nil
	}
}

func podExists(c *client.Client, podNamespace string, podID string) wait.ConditionFunc {
	return func() (bool, error) {
		_, err := c.Pods(podNamespace).Get(podID)
		return err == nil, nil
	}
}

func runReplicationControllerTest(c *client.Client) {
	data, err := ioutil.ReadFile("api/examples/controller.json")
	if err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}
	var controller api.ReplicationController
	if err := api.Scheme.DecodeInto(data, &controller); err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}

	glog.Infof("Creating replication controllers")
	if _, err := c.ReplicationControllers(api.NamespaceDefault).Create(&controller); err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}
	glog.Infof("Done creating replication controllers")

	// Give the controllers some time to actually create the pods
	if err := wait.Poll(time.Second, time.Second*30, c.ControllerHasDesiredReplicas(controller)); err != nil {
		glog.Fatalf("FAILED: pods never created %v", err)
	}

	// wait for minions to indicate they have info about the desired pods
	pods, err := c.Pods(api.NamespaceDefault).List(labels.Set(controller.Spec.Selector).AsSelector())
	if err != nil {
		glog.Fatalf("FAILED: unable to get pods to list: %v", err)
	}
	if err := wait.Poll(time.Second, time.Second*30, podsOnMinions(c, *pods)); err != nil {
		glog.Fatalf("FAILED: pods never started running %v", err)
	}

	glog.Infof("Pods created")
}

func runAPIVersionsTest(c *client.Client) {
	v, err := c.ServerAPIVersions()
	if err != nil {
		glog.Fatalf("failed to get api versions: %v", err)
	}
	if e, a := []string{"v1beta1", "v1beta2"}, v.Versions; !reflect.DeepEqual(e, a) {
		glog.Fatalf("Expected version list '%v', got '%v'", e, a)
	}
	glog.Infof("Version test passed")
}

func runSelfLinkTest(c *client.Client) {
	var svc api.Service
	err := c.Post().Path("services").Body(
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: "selflinktest",
				Labels: map[string]string{
					"name": "selflinktest",
				},
			},
			Spec: api.ServiceSpec{
				Port: 12345,
				// This is here because validation requires it.
				Selector: map[string]string{
					"foo": "bar",
				},
			},
		},
	).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed creating selflinktest service: %v", err)
	}
	err = c.Get().AbsPath(svc.SelfLink).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed listing service with supplied self link '%v': %v", svc.SelfLink, err)
	}

	var svcList api.ServiceList
	err = c.Get().Path("services").Do().Into(&svcList)
	if err != nil {
		glog.Fatalf("Failed listing services: %v", err)
	}

	err = c.Get().AbsPath(svcList.SelfLink).Do().Into(&svcList)
	if err != nil {
		glog.Fatalf("Failed listing services with supplied self link '%v': %v", svcList.SelfLink, err)
	}

	found := false
	for i := range svcList.Items {
		item := &svcList.Items[i]
		if item.Name != "selflinktest" {
			continue
		}
		found = true
		err = c.Get().AbsPath(item.SelfLink).Do().Into(&svc)
		if err != nil {
			glog.Fatalf("Failed listing service with supplied self link '%v': %v", item.SelfLink, err)
		}
		break
	}
	if !found {
		glog.Fatalf("never found selflinktest service")
	}
	glog.Infof("Self link test passed")

	// TODO: Should test PUT at some point, too.
}

func runAtomicPutTest(c *client.Client) {
	var svc api.Service
	err := c.Post().Path("services").Body(
		&api.Service{
			TypeMeta: api.TypeMeta{
				APIVersion: latest.Version,
			},
			ObjectMeta: api.ObjectMeta{
				Name: "atomicservice",
				Labels: map[string]string{
					"name": "atomicService",
				},
			},
			Spec: api.ServiceSpec{
				Port: 12345,
				// This is here because validation requires it.
				Selector: map[string]string{
					"foo": "bar",
				},
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
					Path(svc.Name).
					Do().
					Into(&tmpSvc)
				if err != nil {
					glog.Errorf("Error getting atomicService: %v", err)
					continue
				}
				if tmpSvc.Spec.Selector == nil {
					tmpSvc.Spec.Selector = map[string]string{l: v}
				} else {
					tmpSvc.Spec.Selector[l] = v
				}
				glog.Infof("Posting update (%s, %s)", l, v)
				err = c.Put().Path("services").Path(svc.Name).Body(&tmpSvc).Do().Error()
				if err != nil {
					if errors.IsConflict(err) {
						glog.Infof("Conflict: (%s, %s)", l, v)
						// This is what we expect.
						continue
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
	if err := c.Get().Path("services").Path(svc.Name).Do().Into(&svc); err != nil {
		glog.Fatalf("Failed getting atomicService after writers are complete: %v", err)
	}
	if !reflect.DeepEqual(testLabels, labels.Set(svc.Spec.Selector)) {
		glog.Fatalf("Selector PUTs were not atomic: wanted %v, got %v", testLabels, svc.Spec.Selector)
	}
	glog.Info("Atomic PUTs work.")
}

func runMasterServiceTest(client *client.Client) {
	time.Sleep(12 * time.Second)
	var svcList api.ServiceList
	err := client.Get().
		Namespace("default").
		Path("services").
		Do().
		Into(&svcList)
	if err != nil {
		glog.Fatalf("unexpected error listing services: %v", err)
	}
	var foundRW, foundRO bool
	found := util.StringSet{}
	for i := range svcList.Items {
		found.Insert(svcList.Items[i].Name)
		if svcList.Items[i].Name == "kubernetes" {
			foundRW = true
		}
		if svcList.Items[i].Name == "kubernetes-ro" {
			foundRO = true
		}
	}
	if foundRW {
		var ep api.Endpoints
		err := client.Get().
			Namespace("default").
			Path("endpoints").
			Path("kubernetes").
			Do().
			Into(&ep)
		if err != nil {
			glog.Fatalf("unexpected error listing endpoints for kubernetes service: %v", err)
		}
		if len(ep.Endpoints) == 0 {
			glog.Fatalf("no endpoints for kubernetes service: %v", ep)
		}
	} else {
		glog.Errorf("no RW service found: %v", found)
	}
	if foundRO {
		var ep api.Endpoints
		err := client.Get().
			Namespace("default").
			Path("endpoints").
			Path("kubernetes-ro").
			Do().
			Into(&ep)
		if err != nil {
			glog.Fatalf("unexpected error listing endpoints for kubernetes service: %v", err)
		}
		if len(ep.Endpoints) == 0 {
			glog.Fatalf("no endpoints for kubernetes service: %v", ep)
		}
	} else {
		glog.Errorf("no RO service found: %v", found)
	}
	if !foundRW || !foundRO {
		glog.Fatalf("Kubernetes service test failed: %v", found)
	}
	glog.Infof("Master service test passed.")
}

func runServiceTest(client *client.Client) {
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "thisisalonglabel",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c1",
					Image: "foo",
					Ports: []api.Port{
						{ContainerPort: 1234},
					},
				},
			},
		},
		Status: api.PodStatus{
			PodIP: "1.2.3.4",
		},
	}
	_, err := client.Pods(api.NamespaceDefault).Create(&pod)
	if err != nil {
		glog.Fatalf("Failed to create pod: %v, %v", pod, err)
	}
	if err := wait.Poll(time.Second, time.Second*20, podExists(client, pod.Namespace, pod.Name)); err != nil {
		glog.Fatalf("FAILED: pod never started running %v", err)
	}
	svc1 := api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service1"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "thisisalonglabel",
			},
			Port: 8080,
		},
	}
	_, err = client.Services(api.NamespaceDefault).Create(&svc1)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc1, err)
	}
	if err := wait.Poll(time.Second, time.Second*20, endpointsSet(client, svc1.Namespace, svc1.Name, 1)); err != nil {
		glog.Fatalf("FAILED: unexpected endpoints: %v", err)
	}
	// A second service with the same port.
	svc2 := api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service2"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "thisisalonglabel",
			},
			Port: 8080,
		},
	}
	_, err = client.Services(api.NamespaceDefault).Create(&svc2)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc2, err)
	}
	if err := wait.Poll(time.Second, time.Second*20, endpointsSet(client, svc2.Namespace, svc2.Name, 1)); err != nil {
		glog.Fatalf("FAILED: unexpected endpoints: %v", err)
	}
	glog.Info("Service test passed.")
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

	kubeClient := client.NewOrDie(&client.Config{Host: apiServerURL, Version: testapi.Version()})

	// Run tests in parallel
	testFuncs := []testFunc{
		runReplicationControllerTest,
		runAtomicPutTest,
		runServiceTest,
		runAPIVersionsTest,
		runMasterServiceTest,
		runSelfLinkTest,
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
	//              1 net container + 2 pods from the URL +
	//              1 net container + 1 pod from the service test.
	if len(createdPods) != 9 {
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
	// Note that kubelet complains about these containers not having a self link.
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
