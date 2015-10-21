/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	docker "github.com/fsouza/go-dockerclient"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/node"
	replicationControllerPkg "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/e2e"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

var (
	fakeDocker1, fakeDocker2 dockertools.FakeDockerClient
	// Limit the number of concurrent tests.
	maxConcurrency int
)

type fakeKubeletClient struct{}

func (fakeKubeletClient) GetConnectionInfo(host string) (string, uint, http.RoundTripper, error) {
	return "", 0, nil, errors.New("Not Implemented")
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

func startComponents(firstManifestURL, secondManifestURL string) (string, string) {
	// Setup
	servers := []string{}
	glog.Infof("Creating etcd client pointing to %v", servers)

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

	cl := client.NewOrDie(&client.Config{Host: apiServer.URL, Version: testapi.Default.GroupAndVersion()})

	// TODO: caesarxuchao: hacky way to specify version of Experimental client.
	// We will fix this by supporting multiple group versions in Config
	cl.ExtensionsClient = client.NewExtensionsOrDie(&client.Config{Host: apiServer.URL, Version: testapi.Extensions.GroupAndVersion()})

	storageVersions := make(map[string]string)
	etcdStorage, err := master.NewEtcdStorage(etcdClient, latest.GroupOrDie("").InterfacesFor, testapi.Default.GroupAndVersion(), etcdtest.PathPrefix())
	storageVersions[""] = testapi.Default.GroupAndVersion()
	if err != nil {
		glog.Fatalf("Unable to get etcd storage: %v", err)
	}
	expEtcdStorage, err := master.NewEtcdStorage(etcdClient, latest.GroupOrDie("extensions").InterfacesFor, testapi.Extensions.GroupAndVersion(), etcdtest.PathPrefix())
	storageVersions["extensions"] = testapi.Extensions.GroupAndVersion()
	if err != nil {
		glog.Fatalf("Unable to get etcd storage for experimental: %v", err)
	}
	storageDestinations := master.NewStorageDestinations()
	storageDestinations.AddAPIGroup("", etcdStorage)
	storageDestinations.AddAPIGroup("extensions", expEtcdStorage)

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
		StorageDestinations:   storageDestinations,
		KubeletClient:         fakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableProfiling:       true,
		APIPrefix:             "/api",
		APIGroupPrefix:        "/apis",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
		ReadWritePort:         portNumber,
		PublicAddress:         publicAddress,
		CacheTimeout:          2 * time.Second,
		StorageVersions:       storageVersions,
	})
	handler.delegate = m.Handler

	// Scheduler
	schedulerConfigFactory := factory.NewConfigFactory(cl, nil)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		glog.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"})
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(cl.Events(""))
	scheduler.New(schedulerConfig).Run()

	// ensure the service endpoints are sync'd several times within the window that the integration tests wait
	go endpointcontroller.NewEndpointController(cl, controller.NoResyncPeriodFunc).
		Run(3, util.NeverStop)

	// TODO: Write an integration test for the replication controllers watch.
	go replicationControllerPkg.NewReplicationManager(cl, controller.NoResyncPeriodFunc, replicationControllerPkg.BurstReplicas).
		Run(3, util.NeverStop)

	nodeController := nodecontroller.NewNodeController(nil, cl, 5*time.Minute, util.NewFakeRateLimiter(),
		40*time.Second, 60*time.Second, 5*time.Second, nil, false)
	nodeController.Run(5 * time.Second)
	cadvisorInterface := new(cadvisor.Fake)

	// Kubelet (localhost)
	testRootDir := makeTempDirOrDie("kubelet_integ_1.", "")
	configFilePath := makeTempDirOrDie("config", testRootDir)
	glog.Infof("Using %s as root dir for kubelet #1", testRootDir)
	fakeDocker1.VersionInfo = docker.Env{"ApiVersion=1.20"}

	kcfg := kubeletapp.SimpleKubelet(
		cl,
		&fakeDocker1,
		"localhost",
		testRootDir,
		firstManifestURL,
		"127.0.0.1",
		10250, /* KubeletPort */
		0,     /* ReadOnlyPort */
		api.NamespaceDefault,
		empty_dir.ProbeVolumePlugins(),
		nil,
		cadvisorInterface,
		configFilePath,
		nil,
		kubecontainer.FakeOS{},
		1*time.Second,  /* FileCheckFrequency */
		1*time.Second,  /* HTTPCheckFrequency */
		10*time.Second, /* MinimumGCAge */
		3*time.Second,  /* NodeStatusUpdateFrequency */
		10*time.Second /* SyncFrequency */)

	kubeletapp.RunKubelet(kcfg, nil)
	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	testRootDir = makeTempDirOrDie("kubelet_integ_2.", "")
	glog.Infof("Using %s as root dir for kubelet #2", testRootDir)
	fakeDocker2.VersionInfo = docker.Env{"ApiVersion=1.20"}

	kcfg = kubeletapp.SimpleKubelet(
		cl,
		&fakeDocker2,
		"127.0.0.1",
		testRootDir,
		secondManifestURL,
		"127.0.0.1",
		10251, /* KubeletPort */
		0,     /* ReadOnlyPort */
		api.NamespaceDefault,
		empty_dir.ProbeVolumePlugins(),
		nil,
		cadvisorInterface,
		"",
		nil,
		kubecontainer.FakeOS{},
		1*time.Second,  /* FileCheckFrequency */
		1*time.Second,  /* HTTPCheckFrequency */
		10*time.Second, /* MinimumGCAge */
		3*time.Second,  /* NodeStatusUpdateFrequency */
		10*time.Second /* SyncFrequency */)

	kubeletapp.RunKubelet(kcfg, nil)
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
func podsOnMinions(c *client.Client, podNamespace string, labelSelector labels.Selector) wait.ConditionFunc {
	// Wait until all pods are running on the node.
	return func() (bool, error) {
		pods, err := c.Pods(podNamespace).List(labelSelector, fields.Everything())
		if err != nil {
			glog.Infof("Unable to get pods to list: %v", err)
			return false, nil
		}
		for i := range pods.Items {
			pod := pods.Items[i]
			podString := fmt.Sprintf("%q/%q", pod.Namespace, pod.Name)
			glog.Infof("Check whether pod %q exists on node %q", podString, pod.Spec.NodeName)
			if len(pod.Spec.NodeName) == 0 {
				glog.Infof("Pod %q is not bound to a host yet", podString)
				return false, nil
			}
			if pod.Status.Phase != api.PodRunning {
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
			glog.Infof("Error getting endpoints: %v", err)
			return false, nil
		}
		count := 0
		for _, ss := range endpoints.Subsets {
			for _, addr := range ss.Addresses {
				for _, port := range ss.Ports {
					count++
					glog.Infof("%s/%s endpoint: %s:%d %#v", serviceNamespace, serviceID, addr.IP, port.Port, addr.TargetRef)
				}
			}
		}
		return count == endpointCount, nil
	}
}

func countEndpoints(eps *api.Endpoints) int {
	count := 0
	for i := range eps.Subsets {
		count += len(eps.Subsets[i].Addresses) * len(eps.Subsets[i].Ports)
	}
	return count
}

func podExists(c *client.Client, podNamespace string, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		_, err := c.Pods(podNamespace).Get(podName)
		return err == nil, nil
	}
}

func podNotFound(c *client.Client, podNamespace string, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		_, err := c.Pods(podNamespace).Get(podName)
		return apierrors.IsNotFound(err), nil
	}
}

func podRunning(c *client.Client, podNamespace string, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.Pods(podNamespace).Get(podName)
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry, but log the error.
			glog.Errorf("Error when reading pod %q: %v", podName, err)
			return false, nil
		}
		if pod.Status.Phase != api.PodRunning {
			return false, nil
		}
		return true, nil
	}
}

// runStaticPodTest is disabled until #6651 is resolved.
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
				"apiVersion": "v1",
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
				if pods, err := c.Pods(namespace).List(labels.Everything(), fields.Everything()); err == nil {
					for _, pod := range pods.Items {
						glog.Infof("pod found: %s/%s", namespace, pod.Name)
					}
				}
				glog.Fatalf("%s FAILED: mirror pod has not been created or is not running: %v", desc, err)
			}
			// Delete the mirror pod, and wait for it to be recreated.
			c.Pods(namespace).Delete(podName, nil)
			if err = wait.Poll(time.Second, time.Minute*1,
				podRunning(c, namespace, podName)); err != nil {
				glog.Fatalf("%s FAILED: mirror pod has not been re-created or is not running: %v", desc, err)
			}
			// Remove the manifest file, and wait for the mirror pod to be deleted.
			os.Remove(manifestFile.Name())
			if err = wait.Poll(time.Second, time.Minute*1,
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

	// In practice the controller doesn't need 60s to create a handful of pods, but network latencies on CI
	// systems have been observed to vary unpredictably, so give the controller enough time to create pods.
	// Our e2e scalability tests will catch controllers that are *actually* slow.
	if err := wait.Poll(time.Second, time.Second*60, client.ControllerHasDesiredReplicas(c, updated)); err != nil {
		glog.Fatalf("FAILED: pods never created %v", err)
	}

	// Poll till we can retrieve the status of all pods matching the given label selector from their minions.
	// This involves 3 operations:
	//	- The scheduler must assign all pods to a minion
	//	- The assignment must reflect in a `List` operation against the apiserver, for labels matching the selector
	//  - We need to be able to query the kubelet on that minion for information about the pod
	if err := wait.Poll(
		time.Second, time.Second*30, podsOnMinions(c, "test", labels.Set(updated.Spec.Selector).AsSelector())); err != nil {
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
			// This is here because validation requires it.
			Selector: map[string]string{
				"foo": "bar",
			},
			Ports: []api.ServicePort{{
				Port:     12345,
				Protocol: "TCP",
			}},
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
		TypeMeta: unversioned.TypeMeta{
			APIVersion: c.APIVersion(),
		},
		ObjectMeta: api.ObjectMeta{
			Name: "atomicservice",
			Labels: map[string]string{
				"name": "atomicService",
			},
		},
		Spec: api.ServiceSpec{
			// This is here because validation requires it.
			Selector: map[string]string{
				"foo": "bar",
			},
			Ports: []api.ServicePort{{
				Port:     12345,
				Protocol: "TCP",
			}},
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
	resource := "services"
	svcBody := api.Service{
		TypeMeta: unversioned.TypeMeta{
			APIVersion: c.APIVersion(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: map[string]string{},
		},
		Spec: api.ServiceSpec{
			// This is here because validation requires it.
			Selector: map[string]string{
				"foo": "bar",
			},
			Ports: []api.ServicePort{{
				Port:     12345,
				Protocol: "TCP",
			}},
			SessionAffinity: "None",
		},
	}
	services := c.Services(api.NamespaceDefault)
	svc, err := services.Create(&svcBody)
	if err != nil {
		glog.Fatalf("Failed creating patchservice: %v", err)
	}

	patchBodies := map[string]map[api.PatchType]struct {
		AddLabelBody        []byte
		RemoveLabelBody     []byte
		RemoveAllLabelsBody []byte
	}{
		"v1": {
			api.JSONPatchType: {
				[]byte(`[{"op":"add","path":"/metadata/labels","value":{"foo":"bar","baz":"qux"}}]`),
				[]byte(`[{"op":"remove","path":"/metadata/labels/foo"}]`),
				[]byte(`[{"op":"remove","path":"/metadata/labels"}]`),
			},
			api.MergePatchType: {
				[]byte(`{"metadata":{"labels":{"foo":"bar","baz":"qux"}}}`),
				[]byte(`{"metadata":{"labels":{"foo":null}}}`),
				[]byte(`{"metadata":{"labels":null}}`),
			},
			api.StrategicMergePatchType: {
				[]byte(`{"metadata":{"labels":{"foo":"bar","baz":"qux"}}}`),
				[]byte(`{"metadata":{"labels":{"foo":null}}}`),
				[]byte(`{"metadata":{"labels":{"$patch":"replace"}}}`),
			},
		},
	}

	pb := patchBodies[c.APIVersion()]

	execPatch := func(pt api.PatchType, body []byte) error {
		return c.Patch(pt).
			Resource(resource).
			Namespace(api.NamespaceDefault).
			Name(name).
			Body(body).
			Do().
			Error()
	}
	for k, v := range pb {
		// add label
		err := execPatch(k, v.AddLabelBody)
		if err != nil {
			glog.Fatalf("Failed updating patchservice with patch type %s: %v", k, err)
		}
		svc, err = services.Get(name)
		if err != nil {
			glog.Fatalf("Failed getting patchservice: %v", err)
		}
		if len(svc.Labels) != 2 || svc.Labels["foo"] != "bar" || svc.Labels["baz"] != "qux" {
			glog.Fatalf("Failed updating patchservice with patch type %s: labels are: %v", k, svc.Labels)
		}

		// remove one label
		err = execPatch(k, v.RemoveLabelBody)
		if err != nil {
			glog.Fatalf("Failed updating patchservice with patch type %s: %v", k, err)
		}
		svc, err = services.Get(name)
		if err != nil {
			glog.Fatalf("Failed getting patchservice: %v", err)
		}
		if len(svc.Labels) != 1 || svc.Labels["baz"] != "qux" {
			glog.Fatalf("Failed updating patchservice with patch type %s: labels are: %v", k, svc.Labels)
		}

		// remove all labels
		err = execPatch(k, v.RemoveAllLabelsBody)
		if err != nil {
			glog.Fatalf("Failed updating patchservice with patch type %s: %v", k, err)
		}
		svc, err = services.Get(name)
		if err != nil {
			glog.Fatalf("Failed getting patchservice: %v", err)
		}
		if svc.Labels != nil {
			glog.Fatalf("Failed remove all labels from patchservice with patch type %s: %v", k, svc.Labels)
		}
	}

	glog.Info("PATCHs work.")
}

func runMasterServiceTest(client *client.Client) {
	time.Sleep(12 * time.Second)
	svcList, err := client.Services(api.NamespaceDefault).List(labels.Everything())
	if err != nil {
		glog.Fatalf("unexpected error listing services: %v", err)
	}
	var foundRW bool
	found := sets.String{}
	for i := range svcList.Items {
		found.Insert(svcList.Items[i].Name)
		if svcList.Items[i].Name == "kubernetes" {
			foundRW = true
		}
	}
	if foundRW {
		ep, err := client.Endpoints(api.NamespaceDefault).Get("kubernetes")
		if err != nil {
			glog.Fatalf("unexpected error listing endpoints for kubernetes service: %v", err)
		}
		if countEndpoints(ep) == 0 {
			glog.Fatalf("no endpoints for kubernetes service: %v", ep)
		}
	} else {
		glog.Errorf("no RW service found: %v", found)
		glog.Fatal("Kubernetes service test failed")
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
			Ports: []api.ServicePort{{
				Port:     8080,
				Protocol: "TCP",
			}},
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
			Ports: []api.ServicePort{{
				Port:     8080,
				Protocol: "TCP",
			}},
			SessionAffinity: "None",
		},
	}
	svc3, err = client.Services("other").Create(svc3)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc3, err)
	}

	// TODO Reduce the timeouts in this test when endpoints controller is sped up. See #6045.
	if err := wait.Poll(time.Second, time.Second*60, endpointsSet(client, svc1.Namespace, svc1.Name, 1)); err != nil {
		glog.Fatalf("FAILED: unexpected endpoints: %v", err)
	}
	// A second service with the same port.
	svc2 := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service2"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{
				"name": "thisisalonglabel",
			},
			Ports: []api.ServicePort{{
				Port:     8080,
				Protocol: "TCP",
			}},
			SessionAffinity: "None",
		},
	}
	svc2, err = client.Services(api.NamespaceDefault).Create(svc2)
	if err != nil {
		glog.Fatalf("Failed to create service: %v, %v", svc2, err)
	}
	if err := wait.Poll(time.Second, time.Second*60, endpointsSet(client, svc2.Namespace, svc2.Name, 1)); err != nil {
		glog.Fatalf("FAILED: unexpected endpoints: %v", err)
	}

	if err := wait.Poll(time.Second, time.Second*60, endpointsSet(client, svc3.Namespace, svc3.Name, 0)); err != nil {
		glog.Fatalf("FAILED: service in other namespace should have no endpoints: %v", err)
	}

	svcList, err := client.Services(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Fatalf("Failed to list services across namespaces: %v", err)
	}
	names := sets.NewString()
	for _, svc := range svcList.Items {
		names.Insert(fmt.Sprintf("%s/%s", svc.Namespace, svc.Name))
	}
	if !names.HasAll("default/kubernetes", "default/service1", "default/service2", "other/service1") {
		glog.Fatalf("Unexpected service list: %#v", names)
	}

	glog.Info("Service test passed.")
}

func runSchedulerNoPhantomPodsTest(client *client.Client) {
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c1",
					Image: "kubernetes/pause",
					Ports: []api.ContainerPort{
						{ContainerPort: 1234, HostPort: 9999},
					},
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
		},
	}

	// Assuming we only have two kublets, the third pod here won't schedule
	// if the scheduler doesn't correctly handle the delete for the second
	// pod.
	pod.ObjectMeta.Name = "phantom.foo"
	foo, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		glog.Fatalf("Failed to create pod: %v, %v", pod, err)
	}
	if err := wait.Poll(time.Second, time.Second*30, podRunning(client, foo.Namespace, foo.Name)); err != nil {
		glog.Fatalf("FAILED: pod never started running %v", err)
	}

	pod.ObjectMeta.Name = "phantom.bar"
	bar, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		glog.Fatalf("Failed to create pod: %v, %v", pod, err)
	}
	if err := wait.Poll(time.Second, time.Second*30, podRunning(client, bar.Namespace, bar.Name)); err != nil {
		glog.Fatalf("FAILED: pod never started running %v", err)
	}

	// Delete a pod to free up room.
	glog.Infof("Deleting pod %v", bar.Name)
	err = client.Pods(api.NamespaceDefault).Delete(bar.Name, api.NewDeleteOptions(0))
	if err != nil {
		glog.Fatalf("FAILED: couldn't delete pod %q: %v", bar.Name, err)
	}

	pod.ObjectMeta.Name = "phantom.baz"
	baz, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		glog.Fatalf("Failed to create pod: %v, %v", pod, err)
	}
	if err := wait.Poll(time.Second, time.Second*60, podRunning(client, baz.Namespace, baz.Name)); err != nil {
		if pod, perr := client.Pods(api.NamespaceDefault).Get("phantom.bar"); perr == nil {
			glog.Fatalf("FAILED: 'phantom.bar' was never deleted: %#v", pod)
		} else {
			glog.Fatalf("FAILED: (Scheduler probably didn't process deletion of 'phantom.bar') Pod never started running: %v", err)
		}
	}

	glog.Info("Scheduler doesn't make phantom pods: test passed.")
}

type testFunc func(*client.Client)

func addFlags(fs *pflag.FlagSet) {
	fs.IntVar(
		&maxConcurrency, "max-concurrency", -1, "Maximum number of tests to be run simultaneously. Unlimited if set to negative.")
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

	glog.Infof("Running tests for APIVersion: %s", os.Getenv("KUBE_TEST_API"))

	firstManifestURL := ServeCachedManifestFile(testPodSpecFile)
	secondManifestURL := ServeCachedManifestFile(testPodSpecFile)
	apiServerURL, _ := startComponents(firstManifestURL, secondManifestURL)

	// Ok. we're good to go.
	glog.Infof("API Server started on %s", apiServerURL)
	// Wait for the synchronization threads to come up.
	time.Sleep(time.Second * 10)

	kubeClient := client.NewOrDie(&client.Config{Host: apiServerURL, Version: testapi.Default.GroupAndVersion()})
	// TODO: caesarxuchao: hacky way to specify version of Experimental client.
	// We will fix this by supporting multiple group versions in Config
	kubeClient.ExtensionsClient = client.NewExtensionsOrDie(&client.Config{Host: apiServerURL, Version: testapi.Extensions.GroupAndVersion()})

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
	}

	// Only run at most maxConcurrency tests in parallel.
	if maxConcurrency <= 0 {
		maxConcurrency = len(testFuncs)
	}
	glog.Infof("Running %d tests in parallel.", maxConcurrency)
	ch := make(chan struct{}, maxConcurrency)

	var wg sync.WaitGroup
	wg.Add(len(testFuncs))
	for i := range testFuncs {
		f := testFuncs[i]
		go func() {
			ch <- struct{}{}
			f(kubeClient)
			<-ch
			wg.Done()
		}()
	}
	wg.Wait()
	close(ch)

	// Check that kubelet tried to make the containers.
	// Using a set to list unique creation attempts. Our fake is
	// really stupid, so kubelet tries to create these multiple times.
	createdConts := sets.String{}
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
	// The total number of container created is 9

	if len(createdConts) != 12 {
		glog.Fatalf("Expected 12 containers; got %v\n\nlist of created containers:\n\n%#v\n\nDocker 1 Created:\n\n%#v\n\nDocker 2 Created:\n\n%#v\n\n", len(createdConts), createdConts.List(), fakeDocker1.Created, fakeDocker2.Created)
	}
	glog.Infof("OK - found created containers: %#v", createdConts.List())

	// This test doesn't run with the others because it can't run in
	// parallel and also it schedules extra pods which would change the
	// above pod counting logic.
	runSchedulerNoPhantomPodsTest(kubeClient)

	glog.Infof("\n\nLogging high latency metrics from the 10250 kubelet")
	e2e.HighLatencyKubeletOperations(nil, 1*time.Second, "localhost:10250")
	glog.Infof("\n\nLogging high latency metrics from the 10251 kubelet")
	e2e.HighLatencyKubeletOperations(nil, 1*time.Second, "localhost:10251")
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
		"apiVersion": "v1",
		"metadata": {
			"name": "container-vm-guestbook-pod-spec"
		},
		"spec": {
			"containers": [
				{
					"name": "redis",
					"image": "redis",
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
