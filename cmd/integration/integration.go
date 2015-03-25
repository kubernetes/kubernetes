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
	"errors"
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
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	nodeControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	replicationControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/empty_dir"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

var (
	fakeDocker1, fakeDocker2 dockertools.FakeDockerClient
	// API version that should be used by the client to talk to the server.
	apiVersion string
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

func (fakeKubeletClient) GetConnectionInfo(host string) (string, uint, http.RoundTripper, error) {
	return "", 0, nil, errors.New("Not Implemented")
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

func startComponents(firstManifestURL, secondManifestURL, apiVersion string) (string, string) {
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

	cl := client.NewOrDie(&client.Config{Host: apiServer.URL, Version: apiVersion})

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
		EnableV1Beta3:     true,
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
	// ensure the service endpoints are sync'd several times within the window that the integration tests wait
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*4)

	controllerManager := replicationControllerPkg.NewReplicationManager(cl)

	// TODO: Write an integration test for the replication controllers watch.
	controllerManager.Run(1 * time.Second)

	nodeResources := &api.NodeResources{}

	nodeController := nodeControllerPkg.NewNodeController(nil, "", machineList, nodeResources, cl, fakeKubeletClient{}, 10, 5*time.Minute)
	nodeController.Run(5*time.Second, true, false)
	cadvisorInterface := new(cadvisor.Fake)

	// Kubelet (localhost)
	testRootDir := makeTempDirOrDie("kubelet_integ_1.", "")
	configFilePath := makeTempDirOrDie("config", testRootDir)
	glog.Infof("Using %s as root dir for kubelet #1", testRootDir)
	kcfg := kubeletapp.SimpleKubelet(cl, &fakeDocker1, machineList[0], testRootDir, firstManifestURL, "127.0.0.1", 10250, api.NamespaceDefault, empty_dir.ProbeVolumePlugins(), nil, cadvisorInterface, configFilePath)
	kubeletapp.RunKubelet(kcfg)
	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	testRootDir = makeTempDirOrDie("kubelet_integ_2.", "")
	glog.Infof("Using %s as root dir for kubelet #2", testRootDir)
	kcfg = kubeletapp.SimpleKubelet(cl, &fakeDocker2, machineList[1], testRootDir, secondManifestURL, "127.0.0.1", 10251, api.NamespaceDefault, empty_dir.ProbeVolumePlugins(), nil, cadvisorInterface, "")
	kubeletapp.RunKubelet(kcfg)
	return apiServer.URL, configFilePath
}

func makeTempDirOrDie(prefix string, baseDir string) string {
	if baseDir == "" {
		baseDir = "/tmp"
	}
	tempDir, err := ioutil.TempDir(baseDir, prefix)
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
		for _, e := range endpoints.Endpoints {
			glog.Infof("%s/%s endpoint: %s:%d %#v", serviceNamespace, serviceID, e.IP, e.Port, e.TargetRef)
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

func podNotFound(c *client.Client, podNamespace string, podID string) wait.ConditionFunc {
	return func() (bool, error) {
		_, err := c.Pods(podNamespace).Get(podID)
		return apierrors.IsNotFound(err), nil
	}
}

func podRunning(c *client.Client, podNamespace string, podID string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.Pods(podNamespace).Get(podID)
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry, but log the error.
			glog.Errorf("Error when reading pod %q: %v", podID, err)
			return false, nil
		}
		if pod.Status.Phase != api.PodRunning {
			return false, nil
		}
		return true, nil
	}
}

func runStaticPodTest(c *client.Client, configFilePath string) {
	var testCases = []struct {
		desc         string
		fileContents string
	}{
		{
			desc: "static-pod-from-manifest",
			fileContents: `version: v1beta2
id: static-pod-from-manifest
containers:
  - name: static-container
    image: kubernetes/pause`,
		},
		{
			desc: "static-pod-from-spec",
			fileContents: `{
				"kind": "Pod",
				"apiVersion": "v1beta3",
				"metadata": {
					"name": "static-pod-from-spec"
				},
				"spec": {
					"containers": [{
						"name": "static-container",
						"image": "kubernetes/pause"
					}]
				}
			}`,
		},
	}

	for _, testCase := range testCases {
		func() {
			desc := testCase.desc
			manifestFile, err := ioutil.TempFile(configFilePath, "")
			defer os.Remove(manifestFile.Name())
			ioutil.WriteFile(manifestFile.Name(), []byte(testCase.fileContents), 0600)

			// Wait for the mirror pod to be created.
			podName := fmt.Sprintf("%s-localhost", desc)
			namespace := kubelet.NamespaceDefault
			if err := wait.Poll(time.Second, time.Minute*2,
				podRunning(c, namespace, podName)); err != nil {
				if pods, err := c.Pods(namespace).List(labels.Everything()); err == nil {
					for _, pod := range pods.Items {
						glog.Infof("pod found: %s/%s", namespace, pod.Name)
					}
				}
				glog.Fatalf("%s FAILED: mirror pod has not been created or is not running: %v", desc, err)
			}
			// Delete the mirror pod, and wait for it to be recreated.
			c.Pods(namespace).Delete(podName)
			if err = wait.Poll(time.Second, time.Second*30,
				podRunning(c, namespace, podName)); err != nil {
				glog.Fatalf("%s FAILED: mirror pod has not been re-created or is not running: %v", desc, err)
			}
			// Remove the manifest file, and wait for the mirror pod to be deleted.
			os.Remove(manifestFile.Name())
			if err = wait.Poll(time.Second, time.Second*30,
				podNotFound(c, namespace, podName)); err != nil {
				glog.Fatalf("%s FAILED: mirror pod has not been deleted: %v", desc, err)
			}
		}()
	}
}

func runReplicationControllerTest(c *client.Client) {
	clientAPIVersion := c.APIVersion()
	data, err := ioutil.ReadFile("cmd/integration/" + clientAPIVersion + "-controller.json")
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
	clientVersion := c.APIVersion()
	if err != nil {
		glog.Fatalf("failed to get api versions: %v", err)
	}
	// Verify that the server supports the API version used by the client.
	for _, version := range v.Versions {
		if version == clientVersion {
			glog.Infof("Version test passed")
			return
		}
	}
	glog.Fatalf("Server does not support APIVersion used by client. Server supported APIVersions: '%v', client APIVersion: '%v'", v.Versions, clientVersion)
}

func runSelfLinkTestOnNamespace(c *client.Client, namespace string) {
	svcBody := api.Service{
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
	}
	services := c.Services(namespace)
	svc, err := services.Create(&svcBody)
	if err != nil {
		glog.Fatalf("Failed creating selflinktest service: %v", err)
	}
	err = c.Get().RequestURI(svc.SelfLink).Do().Into(svc)
	if err != nil {
		glog.Fatalf("Failed listing service with supplied self link '%v': %v", svc.SelfLink, err)
	}

	svcList, err := services.List(labels.Everything())
	if err != nil {
		glog.Fatalf("Failed listing services: %v", err)
	}

	err = c.Get().RequestURI(svcList.SelfLink).Do().Into(svcList)
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
		err = c.Get().RequestURI(item.SelfLink).Do().Into(svc)
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
	svcBody := api.Service{
		TypeMeta: api.TypeMeta{
			APIVersion: c.APIVersion(),
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
	}
	services := c.Services(api.NamespaceDefault)
	svc, err := services.Create(&svcBody)
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
				tmpSvc, err := services.Get(svc.Name)
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
				tmpSvc, err = services.Update(tmpSvc)
				if err != nil {
					if apierrors.IsConflict(err) {
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
	svc, err = services.Get(svc.Name)
	if err != nil {
		glog.Fatalf("Failed getting atomicService after writers are complete: %v", err)
	}
	if !reflect.DeepEqual(testLabels, labels.Set(svc.Spec.Selector)) {
		glog.Fatalf("Selector PUTs were not atomic: wanted %v, got %v", testLabels, svc.Spec.Selector)
	}
	glog.Info("Atomic PUTs work.")
}

func runPatchTest(c *client.Client) {
	name := "patchservice"
	svcBody := api.Service{
		TypeMeta: api.TypeMeta{
			APIVersion: c.APIVersion(),
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
	}
	services := c.Services(api.NamespaceDefault)
	svc, err := services.Create(&svcBody)
	if err != nil {
		glog.Fatalf("Failed creating patchservice: %v", err)
	}
	if len(svc.Labels) != 1 {
		glog.Fatalf("Original length does not equal one")
	}

	// add label
	svc.Labels["foo"] = "bar"
	if _, err = services.Update(svc); err != nil {
		glog.Fatalf("Failed updating patchservice: %v", err)
	}
	if svc, err = services.Get(name); err != nil {
		glog.Fatalf("Failed getting patchservice: %v", err)
	}
	if len(svc.Labels) != 2 || svc.Labels["foo"] != "bar" {
		glog.Fatalf("Failed updating patchservice, labels are: %v", svc.Labels)
	}

	// remove one label
	delete(svc.Labels, "name")
	if _, err = services.Update(svc); err != nil {
		glog.Fatalf("Failed updating patchservice: %v", err)
	}
	if svc, err = services.Get(name); err != nil {
		glog.Fatalf("Failed getting patchservice: %v", err)
	}
	if len(svc.Labels) != 1 || svc.Labels["foo"] != "bar" {
		glog.Fatalf("Failed updating patchservice, labels are: %v", svc.Labels)
	}

	// remove all labels
	svc.Labels = nil
	if _, err = services.Update(svc); err != nil {
		glog.Fatalf("Failed updating patchservice: %v", err)
	}
	if svc, err = services.Get(name); err != nil {
		glog.Fatalf("Failed getting patchservice: %v", err)
	}
	if svc.Labels != nil {
		glog.Fatalf("Failed remove all labels from patchservice: %v", svc.Labels)
	}

	glog.Info("PATCHs work.")
}

func runMasterServiceTest(client *client.Client) {
	time.Sleep(12 * time.Second)
	svcList, err := client.Services(api.NamespaceDefault).List(labels.Everything())
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
		ep, err := client.Endpoints(api.NamespaceDefault).Get("kubernetes")
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
		ep, err := client.Endpoints(api.NamespaceDefault).Get("kubernetes-ro")
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
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
			RestartPolicy: api.RestartPolicyAlways,
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

	// create an identical service in the non-default namespace
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

	if err := wait.Poll(time.Second, time.Second*30, endpointsSet(client, svc1.Namespace, svc1.Name, 1)); err != nil {
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
	if err := wait.Poll(time.Second, time.Second*30, endpointsSet(client, svc2.Namespace, svc2.Name, 1)); err != nil {
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

func addFlags(fs *pflag.FlagSet) {
	fs.StringVar(&apiVersion, "apiVersion", latest.Version, "API version that should be used by the client for communicating with the server")
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	addFlags(pflag.CommandLine)

	util.InitFlags()
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(3 * time.Minute)
		glog.Fatalf("This test has timed out.")
	}()

	glog.Infof("Running tests for APIVersion: %s", apiVersion)

	firstManifestURL := ServeCachedManifestFile(testPodSpecFile)
	secondManifestURL := ServeCachedManifestFile(testManifestFile)
	apiServerURL, configFilePath := startComponents(firstManifestURL, secondManifestURL, apiVersion)

	// Ok. we're good to go.
	glog.Infof("API Server started on %s", apiServerURL)
	// Wait for the synchronization threads to come up.
	time.Sleep(time.Second * 10)

	kubeClient := client.NewOrDie(&client.Config{Host: apiServerURL, Version: apiVersion})

	// Run tests in parallel
	testFuncs := []testFunc{
		runReplicationControllerTest,
		runAtomicPutTest,
		runPatchTest,
		runServiceTest,
		runAPIVersionsTest,
		runMasterServiceTest,
		func(c *client.Client) {
			runSelfLinkTestOnNamespace(c, api.NamespaceDefault)
			runSelfLinkTestOnNamespace(c, "other")
		},
		func(c *client.Client) {
			runStaticPodTest(c, configFilePath)
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
	// We expect 9: 2 pod infra containers + 2 containers from the replication controller +
	//              1 pod infra container + 2 containers from the URL on first Kubelet +
	//              1 pod infra container + 2 containers from the URL on second Kubelet +
	//              1 pod infra container + 1 container from the service test.
	// In addition, runStaticPodTest creates 2 pod infra containers +
	//                                       2 pod containers from the file (1 for manifest and 1 for spec)
	// The total number of container created is 13

	if len(createdConts) != 16 {
		glog.Fatalf("Expected 16 containers; got %v\n\nlist of created containers:\n\n%#v\n\nDocker 1 Created:\n\n%#v\n\nDocker 2 Created:\n\n%#v\n\n", len(createdConts), createdConts.List(), fakeDocker1.Created, fakeDocker2.Created)
	}
	glog.Infof("OK - found created containers: %#v", createdConts.List())
}

// ServeCachedManifestFile serves a file for kubelet to read.
func ServeCachedManifestFile(contents string) (servingAddress string) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manifest" {
			w.Write([]byte(contents))
			return
		}
		glog.Fatalf("Got request: %#v\n", r)
		http.NotFound(w, r)
	}))
	return server.URL + "/manifest"
}

const (
	testPodSpecFile = `{
		"kind": "Pod",
		"apiVersion": "v1beta3",
		"metadata": {
			"name": "container-vm-guestbook-pod-spec"
		},
		"spec": {
			"containers": [
				{
					"name": "redis",
					"image": "dockerfile/redis",
					"volumeMounts": [{
						"name": "redis-data",
						"mountPath": "/data"
					}]
				},
				{
					"name": "guestbook",
					"image": "google/guestbook-python-redis",
					"ports": [{
						"name": "www",
						"hostPort": 80,
						"containerPort": 80
					}]
				}],
			"volumes": [{	"name": "redis-data" }]
		}
	}`
)

const (
	// This is copied from, and should be kept in sync with:
	// https://raw.githubusercontent.com/GoogleCloudPlatform/container-vm-guestbook-redis-python/master/manifest.yaml
	// Note that kubelet complains about these containers not having a self link.
	testManifestFile = `version: v1beta2
id: container-vm-guestbook-manifest
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
