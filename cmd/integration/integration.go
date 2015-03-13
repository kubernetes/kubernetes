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
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	kubeletapp "github.com/GoogleCloudPlatform/kubernetes/cmd/kubelet/app"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	nodeControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	replicationControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/empty_dir"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

var (
	fakeDocker1, fakeDocker2 dockertools.FakeDockerClient
)

type fakeKubeletClient struct{}

func (fakeKubeletClient) GetPodStatus(host, podNamespace, podID string) (api.PodStatusResult, error) {
	glog.V(3).Infof("Trying to get container info for %v/%v/%v", host, podNamespace, podID)
	// This is a horrible hack to get around the fact that we can't provide
	// different port numbers per kubelet...
	var c client.PodInfoGetter
	switch host {
	case "localhost":
		c = &client.HTTPKubeletClient{
			Client: http.DefaultClient,
			Port:   10250,
		}
	case "127.0.0.1":
		c = &client.HTTPKubeletClient{
			Client: http.DefaultClient,
			Port:   10251,
		}
	default:
		glog.Fatalf("Can't get info for: '%v', '%v - %v'", host, podNamespace, podID)
	}
	r, err := c.GetPodStatus("127.0.0.1", podNamespace, podID)
	if err != nil {
		return r, err
	}
	r.Status.PodIP = "1.2.3.4"
	m := make(api.PodInfo)
	for k, v := range r.Status.Info {
		v.Ready = true
		m[k] = v
	}
	r.Status.Info = m
	return r, nil
}

func (fakeKubeletClient) GetNodeInfo(host string) (api.NodeInfo, error) {
	return api.NodeInfo{}, nil
}

func (fakeKubeletClient) HealthCheck(host string) (probe.Result, error) {
	return probe.Success, nil
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
	servers := []string{}
	glog.Infof("Creating etcd client pointing to %v", servers)
	machineList := []string{"localhost", "127.0.0.1"}

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

	publicAddress := net.ParseIP(host)
	if publicAddress == nil {
		glog.Fatalf("no public address for %s", host)
	}

	// Create a master and install handlers into mux.
	m := master.New(&master.Config{
		Client:            cl,
		EtcdHelper:        helper,
		KubeletClient:     fakeKubeletClient{},
		EnableLogsSupport: false,
		EnableProfiling:   true,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:  admit.NewAlwaysAdmit(),
		ReadWritePort:     portNumber,
		ReadOnlyPort:      portNumber,
		PublicAddress:     publicAddress,
		CacheTimeout:      2 * time.Second,
		SyncPodStatus:     true,
	})
	handler.delegate = m.Handler

	// Scheduler
	schedulerConfigFactory := factory.NewConfigFactory(cl)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		glog.Fatalf("Couldn't create scheduler config: %v", err)
	}
	scheduler.New(schedulerConfig).Run()

	endpoints := service.NewEndpointController(cl)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	controllerManager := replicationControllerPkg.NewReplicationManager(cl)

	// TODO: Write an integration test for the replication controllers watch.
	controllerManager.Run(1 * time.Second)

	nodeResources := &api.NodeResources{}

	nodeController := nodeControllerPkg.NewNodeController(nil, "", machineList, nodeResources, cl, fakeKubeletClient{}, 10, 5*time.Minute)
	nodeController.Run(5*time.Second, true, false)

	// Kubelet (localhost)
	testRootDir := makeTempDirOrDie("kubelet_integ_1.")
	glog.Infof("Using %s as root dir for kubelet #1", testRootDir)
	kubeletapp.SimpleRunKubelet(cl, &fakeDocker1, machineList[0], testRootDir, manifestURL, "127.0.0.1", 10250, api.NamespaceDefault, empty_dir.ProbeVolumePlugins(), nil)
	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	testRootDir = makeTempDirOrDie("kubelet_integ_2.")
	glog.Infof("Using %s as root dir for kubelet #2", testRootDir)
	kubeletapp.SimpleRunKubelet(cl, &fakeDocker2, machineList[1], testRootDir, "", "127.0.0.1", 10251, api.NamespaceDefault, empty_dir.ProbeVolumePlugins(), nil)

	return apiServer.URL
}

func makeTempDirOrDie(prefix string) string {
	tempDir, err := ioutil.TempDir("/tmp", prefix)
	if err != nil {
		glog.Fatalf("Can't make a temp rootdir: %v", err)
	}
	if err = os.MkdirAll(tempDir, 0750); err != nil {
		glog.Fatalf("Can't mkdir(%q): %v", tempDir, err)
	}
	return tempDir
}

// podsOnMinions returns true when all of the selected pods exist on a minion.
func podsOnMinions(c *client.Client, pods api.PodList) wait.ConditionFunc {
	podInfo := fakeKubeletClient{}
	return func() (bool, error) {
		for i := range pods.Items {
			host, id, namespace := pods.Items[i].Status.Host, pods.Items[i].Name, pods.Items[i].Namespace
			glog.Infof("Check whether pod %s.%s exists on node %q", id, namespace, host)
			if len(host) == 0 {
				glog.Infof("Pod %s.%s is not bound to a host yet", id, namespace)
				return false, nil
			}
			if _, err := podInfo.GetPodStatus(host, namespace, id); err != nil {
				glog.Infof("GetPodStatus error: %v", err)
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
			glog.Infof("Error on creating endpoints: %v", err)
			return false, nil
		}
		glog.Infof("endpoints: %v", endpoints.Endpoints)
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
	data, err := ioutil.ReadFile("cmd/integration/controller.json")
	if err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}
	var controller api.ReplicationController
	if err := api.Scheme.DecodeInto(data, &controller); err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}

	glog.Infof("Creating replication controllers")
	updated, err := c.ReplicationControllers("test").Create(&controller)
	if err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}
	glog.Infof("Done creating replication controllers")

	// Give the controllers some time to actually create the pods
	if err := wait.Poll(time.Second, time.Second*30, client.ControllerHasDesiredReplicas(c, updated)); err != nil {
		glog.Fatalf("FAILED: pods never created %v", err)
	}

	// wait for minions to indicate they have info about the desired pods
	pods, err := c.Pods("test").List(labels.Set(updated.Spec.Selector).AsSelector())
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

func runSelfLinkTestOnNamespace(c *client.Client, namespace string) {
	var svc api.Service
	err := c.Post().
		NamespaceIfScoped(namespace, len(namespace) > 0).
		Resource("services").Body(
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      "selflinktest",
				Namespace: namespace,
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
				Protocol:        "TCP",
				SessionAffinity: "None",
			},
		},
	).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed creating selflinktest service: %v", err)
	}
	// TODO: this is not namespace aware
	err = c.Get().RequestURI(svc.SelfLink).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed listing service with supplied self link '%v': %v", svc.SelfLink, err)
	}

	var svcList api.ServiceList
	err = c.Get().NamespaceIfScoped(namespace, len(namespace) > 0).Resource("services").Do().Into(&svcList)
	if err != nil {
		glog.Fatalf("Failed listing services: %v", err)
	}

	err = c.Get().RequestURI(svcList.SelfLink).Do().Into(&svcList)
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
		err = c.Get().RequestURI(item.SelfLink).Do().Into(&svc)
		if err != nil {
			glog.Fatalf("Failed listing service with supplied self link '%v': %v", item.SelfLink, err)
		}
		break
	}
	if !found {
		glog.Fatalf("never found selflinktest service in namespace %s", namespace)
	}
	glog.Infof("Self link test passed in namespace %s", namespace)

	// TODO: Should test PUT at some point, too.
}

func runAtomicPutTest(c *client.Client) {
	var svc api.Service
	err := c.Post().Resource("services").Body(
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
				Protocol:        "TCP",
				SessionAffinity: "None",
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
					Resource("services").
					Name(svc.Name).
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
				err = c.Put().Resource("services").Name(svc.Name).Body(&tmpSvc).Do().Error()
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
	if err := c.Get().Resource("services").Name(svc.Name).Do().Into(&svc); err != nil {
		glog.Fatalf("Failed getting atomicService after writers are complete: %v", err)
	}
	if !reflect.DeepEqual(testLabels, labels.Set(svc.Spec.Selector)) {
		glog.Fatalf("Selector PUTs were not atomic: wanted %v, got %v", testLabels, svc.Spec.Selector)
	}
	glog.Info("Atomic PUTs work.")
}

func runPatchTest(c *client.Client) {
	name := "patchservice"
	resource := "services"
	var svc api.Service
	err := c.Post().Resource(resource).Body(
		&api.Service{
			TypeMeta: api.TypeMeta{
				APIVersion: latest.Version,
			},
			ObjectMeta: api.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"name": name,
				},
			},
			Spec: api.ServiceSpec{
				Port: 12345,
				// This is here because validation requires it.
				Selector: map[string]string{
					"foo": "bar",
				},
				Protocol:        "TCP",
				SessionAffinity: "None",
			},
		},
	).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed creating patchservice: %v", err)
	}
	if len(svc.Labels) != 1 {
		glog.Fatalf("Original length does not equal one")
	}

	// add label
	_, err = c.Patch().Resource(resource).Name(name).Body([]byte("{\"labels\":{\"foo\":\"bar\"}}")).Do().Get()
	if err != nil {
		glog.Fatalf("Failed updating patchservice: %v", err)
	}
	err = c.Get().Resource(resource).Name(name).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed getting patchservice: %v", err)
	}
	if len(svc.Labels) != 2 || svc.Labels["foo"] != "bar" {
		glog.Fatalf("Failed updating patchservice, labels are: %v", svc.Labels)
	}

	// remove one label
	_, err = c.Patch().Resource(resource).Name(name).Body([]byte("{\"labels\":{\"name\":null}}")).Do().Get()
	if err != nil {
		glog.Fatalf("Failed updating patchservice: %v", err)
	}
	err = c.Get().Resource(resource).Name(name).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed getting patchservice: %v", err)
	}
	if len(svc.Labels) != 1 || svc.Labels["foo"] != "bar" {
		glog.Fatalf("Failed updating patchservice, labels are: %v", svc.Labels)
	}

	// remove all labels
	_, err = c.Patch().Resource(resource).Name(name).Body([]byte("{\"labels\":null}")).Do().Get()
	if err != nil {
		glog.Fatalf("Failed updating patchservice: %v", err)
	}
	err = c.Get().Resource(resource).Name(name).Do().Into(&svc)
	if err != nil {
		glog.Fatalf("Failed getting patchservice: %v", err)
	}
	if svc.Labels != nil {
		glog.Fatalf("Failed remove all labels from patchservice: %v", svc.Labels)
	}

	glog.Info("PATCHs work.")
}

func runMasterServiceTest(client *client.Client) {
	time.Sleep(12 * time.Second)
	var svcList api.ServiceList
	err := client.Get().
		Namespace("default").
		Resource("services").
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
			Resource("endpoints").
			Name("kubernetes").
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
			Resource("endpoints").
			Name("kubernetes-ro").
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
	pod := &api.Pod{
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
					Ports: []api.ContainerPort{
						{ContainerPort: 1234},
					},
					ImagePullPolicy: "PullIfNotPresent",
				},
			},
			RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
			DNSPolicy:     api.DNSClusterFirst,
		},
		Status: api.PodStatus{
			PodIP: "1.2.3.4",
		},
	}
	pod, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		glog.Fatalf("Failed to create pod: %v, %v", pod, err)
	}
	if err := wait.Poll(time.Second, time.Second*20, podExists(client, pod.Namespace, pod.Name)); err != nil {
		glog.Fatalf("FAILED: pod never started running %v", err)
	}
	svc1 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service1"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "thisisalonglabel",
			},
			Port:            8080,
			Protocol:        "TCP",
			SessionAffinity: "None",
		},
	}
	svc1, err = client.Services(api.NamespaceDefault).Create(svc1)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc1, err)
	}

	// create an identical service in the default namespace
	svc3 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service1"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "thisisalonglabel",
			},
			Port:            8080,
			Protocol:        "TCP",
			SessionAffinity: "None",
		},
	}
	svc3, err = client.Services("other").Create(svc3)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc3, err)
	}

	if err := wait.Poll(time.Second, time.Second*20, endpointsSet(client, svc1.Namespace, svc1.Name, 1)); err != nil {
		glog.Fatalf("FAILED: unexpected endpoints: %v", err)
	}
	// A second service with the same port.
	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service2"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "thisisalonglabel",
			},
			Port:            8080,
			Protocol:        "TCP",
			SessionAffinity: "None",
		},
	}
	svc2, err = client.Services(api.NamespaceDefault).Create(svc2)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc2, err)
	}
	if err := wait.Poll(time.Second, time.Second*20, endpointsSet(client, svc2.Namespace, svc2.Name, 1)); err != nil {
		glog.Fatalf("FAILED: unexpected endpoints: %v", err)
	}

	if ok, err := endpointsSet(client, svc3.Namespace, svc3.Name, 0)(); !ok || err != nil {
		glog.Fatalf("FAILED: service in other namespace should have no endpoints: %v %v", ok, err)
	}

	svcList, err := client.Services(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Fatalf("Failed to list services across namespaces: %v", err)
	}
	names := util.NewStringSet()
	for _, svc := range svcList.Items {
		names.Insert(fmt.Sprintf("%s/%s", svc.Namespace, svc.Name))
	}
	if !names.HasAll("default/kubernetes", "default/kubernetes-ro", "default/service1", "default/service2", "other/service1") {
		glog.Fatalf("Unexpected service list: %#v", names)
	}

	glog.Info("Service test passed.")
}

type testFunc func(*client.Client)

func main() {
	util.InitFlags()
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
		runPatchTest,
		runServiceTest,
		runAPIVersionsTest,
		runMasterServiceTest,
		func(c *client.Client) {
			runSelfLinkTestOnNamespace(c, "")
			runSelfLinkTestOnNamespace(c, "other")
		},
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

	// Check that kubelet tried to make the containers.
	// Using a set to list unique creation attempts. Our fake is
	// really stupid, so kubelet tries to create these multiple times.
	createdConts := util.StringSet{}
	for _, p := range fakeDocker1.Created {
		// The last 8 characters are random, so slice them off.
		if n := len(p); n > 8 {
			createdConts.Insert(p[:n-8])
		}
	}
	for _, p := range fakeDocker2.Created {
		// The last 8 characters are random, so slice them off.
		if n := len(p); n > 8 {
			createdConts.Insert(p[:n-8])
		}
	}
	// We expect 9: 2 infra containers + 2 containers from the replication controller +
	//              1 infra container + 2 containers from the URL +
	//              1 infra container + 1 container from the service test.
	if len(createdConts) != 9 {
		glog.Fatalf("Expected 9 containers; got %v\n\nlist of created containers:\n\n%#v\n\nDocker 1 Created:\n\n%#v\n\nDocker 2 Created:\n\n%#v\n\n", len(createdConts), createdConts.List(), fakeDocker1.Created, fakeDocker2.Created)
	}
	glog.Infof("OK - found created containers: %#v", createdConts.List())
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
