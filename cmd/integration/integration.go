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
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	gruntime "runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	endpointcontroller "k8s.io/kubernetes/pkg/controller/endpoint"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	nodecontroller "k8s.io/kubernetes/pkg/controller/node"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	e2e "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"golang.org/x/net/context"
)

var (
	fakeDocker1 = dockertools.NewFakeDockerClient()
	fakeDocker2 = dockertools.NewFakeDockerClient()
	// Limit the number of concurrent tests.
	maxConcurrency int
	watchCache     bool

	longTestTimeout = time.Second * 500

	maxTestTimeout = time.Minute * 15
)

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
	handler := delegateHandler{}
	apiServer := httptest.NewServer(&handler)

	cfg := etcd.Config{
		Endpoints: []string{"http://127.0.0.1:4001"},
	}
	etcdClient, err := etcd.New(cfg)
	if err != nil {
		glog.Fatalf("Error creating etcd client: %v", err)
	}
	glog.Infof("Creating etcd client pointing to %v", cfg.Endpoints)

	keysAPI := etcd.NewKeysAPI(etcdClient)
	sleep := 4 * time.Second
	ok := false
	for i := 0; i < 3; i++ {
		keys, err := keysAPI.Get(context.TODO(), "/", nil)
		if err != nil {
			glog.Warningf("Unable to list root etcd keys: %v", err)
			if i < 2 {
				time.Sleep(sleep)
				sleep = sleep * sleep
			}
			continue
		}
		for _, node := range keys.Node.Nodes {
			if _, err := keysAPI.Delete(context.TODO(), node.Key, &etcd.DeleteOptions{Recursive: true}); err != nil {
				glog.Fatalf("Unable delete key: %v", err)
			}
		}
		ok = true
		break
	}
	if !ok {
		glog.Fatalf("Failed to connect to etcd")
	}

	cl := client.NewOrDie(&restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	// TODO: caesarxuchao: hacky way to specify version of Experimental client.
	// We will fix this by supporting multiple group versions in Config
	cl.ExtensionsClient = client.NewExtensionsOrDie(&restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()}})

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
		glog.Fatalf("No public address for %s", host)
	}

	// The caller of master.New should guarantee pulicAddress is properly set
	hostIP, err := utilnet.ChooseBindAddress(publicAddress)
	if err != nil {
		glog.Fatalf("Unable to find suitable network address.error='%v' . "+
			"Fail to get a valid public address for master.", err)
	}

	masterConfig := framework.NewMasterConfig()
	masterConfig.EnableCoreControllers = true
	masterConfig.EnableProfiling = true
	masterConfig.ReadWritePort = portNumber
	masterConfig.PublicAddress = hostIP
	masterConfig.CacheTimeout = 2 * time.Second
	masterConfig.EnableWatchCache = watchCache

	// Create a master and install handlers into mux.
	m, err := master.New(masterConfig)
	if err != nil {
		glog.Fatalf("Error in bringing up the master: %v", err)
	}
	handler.delegate = m.Handler

	// Scheduler
	schedulerConfigFactory := factory.NewConfigFactory(cl, api.DefaultSchedulerName, api.DefaultHardPodAffinitySymmetricWeight, api.DefaultFailureDomains)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		glog.Fatalf("Couldn't create scheduler config: %v", err)
	}
	eventBroadcaster := record.NewBroadcaster()
	schedulerConfig.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: api.DefaultSchedulerName})
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(cl.Events(""))
	scheduler.New(schedulerConfig).Run()

	podInformer := informers.CreateSharedIndexPodInformer(clientset, controller.NoResyncPeriodFunc(), cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})

	// ensure the service endpoints are sync'd several times within the window that the integration tests wait
	go endpointcontroller.NewEndpointController(podInformer, clientset).
		Run(3, wait.NeverStop)

	// TODO: Write an integration test for the replication controllers watch.
	go replicationcontroller.NewReplicationManager(podInformer, clientset, controller.NoResyncPeriodFunc, replicationcontroller.BurstReplicas, 4096).
		Run(3, wait.NeverStop)

	go podInformer.Run(wait.NeverStop)

	nodeController := nodecontroller.NewNodeController(nil, clientset, 5*time.Minute, flowcontrol.NewFakeAlwaysRateLimiter(), flowcontrol.NewFakeAlwaysRateLimiter(),
		40*time.Second, 60*time.Second, 5*time.Second, nil, false)
	nodeController.Run(5 * time.Second)
	cadvisorInterface := new(cadvisortest.Fake)

	// Kubelet (localhost)
	testRootDir := integration.MakeTempDirOrDie("kubelet_integ_1.", "")
	configFilePath := integration.MakeTempDirOrDie("config", testRootDir)
	glog.Infof("Using %s as root dir for kubelet #1", testRootDir)
	cm := cm.NewStubContainerManager()
	kcfg := kubeletapp.SimpleKubelet(
		clientset,
		fakeDocker1,
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
		containertest.FakeOS{},
		1*time.Second,  /* FileCheckFrequency */
		1*time.Second,  /* HTTPCheckFrequency */
		10*time.Second, /* MinimumGCAge */
		3*time.Second,  /* NodeStatusUpdateFrequency */
		10*time.Second, /* SyncFrequency */
		10*time.Second, /* OutOfDiskTransitionFrequency */
		40,             /* MaxPods */
		cm, net.ParseIP("127.0.0.1"))

	kubeletapp.RunKubelet(kcfg)
	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	testRootDir = integration.MakeTempDirOrDie("kubelet_integ_2.", "")
	glog.Infof("Using %s as root dir for kubelet #2", testRootDir)

	kcfg = kubeletapp.SimpleKubelet(
		clientset,
		fakeDocker2,
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
		containertest.FakeOS{},
		1*time.Second,  /* FileCheckFrequency */
		1*time.Second,  /* HTTPCheckFrequency */
		10*time.Second, /* MinimumGCAge */
		3*time.Second,  /* NodeStatusUpdateFrequency */
		10*time.Second, /* SyncFrequency */
		10*time.Second, /* OutOfDiskTransitionFrequency */

		40, /* MaxPods */
		cm,
		net.ParseIP("127.0.0.1"))

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

// podsOnNodes returns true when all of the selected pods exist on a node.
func podsOnNodes(c *client.Client, podNamespace string, labelSelector labels.Selector) wait.ConditionFunc {
	// Wait until all pods are running on the node.
	return func() (bool, error) {
		options := api.ListOptions{LabelSelector: labelSelector}
		pods, err := c.Pods(podNamespace).List(options)
		if err != nil {
			glog.Infof("Unable to get pods to list: %v", err)
			return false, nil
		}
		for i := range pods.Items {
			pod := pods.Items[i]
			podString := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)
			glog.Infof("Check whether pod %q exists on node %q", podString, pod.Spec.NodeName)
			if len(pod.Spec.NodeName) == 0 {
				glog.Infof("Pod %q is not bound to a host yet", podString)
				return false, nil
			}
			if pod.Status.Phase != api.PodRunning {
				glog.Infof("Pod %q is not running, status: %v", podString, pod.Status.Phase)
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
			glog.V(2).Infof("Pod %s/%s was not found", podNamespace, podName)
			return false, nil
		}
		if err != nil {
			// This could be a connection error so we want to retry, but log the error.
			glog.Errorf("Error when reading pod %q: %v", podName, err)
			return false, nil
		}
		if pod.Status.Phase != api.PodRunning {
			glog.V(2).Infof("Pod %s/%s is not running. In phase %q", podNamespace, podName, pod.Status.Phase)
			return false, nil
		}
		return true, nil
	}
}

func runAPIVersionsTest(c *client.Client) {
	g, err := c.ServerGroups()
	clientVersion := c.APIVersion().String()
	if err != nil {
		glog.Fatalf("Failed to get api versions: %v", err)
	}
	versions := unversioned.ExtractGroupVersions(g)

	// Verify that the server supports the API version used by the client.
	for _, version := range versions {
		if version == clientVersion {
			glog.Infof("Version test passed")
			return
		}
	}
	glog.Fatalf("Server does not support APIVersion used by client. Server supported APIVersions: '%v', client APIVersion: '%v'", versions, clientVersion)
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

	svcList, err := services.List(api.ListOptions{})
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
			APIVersion: c.APIVersion().String(),
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
			APIVersion: c.APIVersion().String(),
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

	patchBodies := map[unversioned.GroupVersion]map[api.PatchType]struct {
		AddLabelBody        []byte
		RemoveLabelBody     []byte
		RemoveAllLabelsBody []byte
	}{
		v1.SchemeGroupVersion: {
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
	svcList, err := client.Services(api.NamespaceDefault).List(api.ListOptions{})
	if err != nil {
		glog.Fatalf("Unexpected error listing services: %v", err)
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
			glog.Fatalf("Unexpected error listing endpoints for kubernetes service: %v", err)
		}
		if countEndpoints(ep) == 0 {
			glog.Fatalf("No endpoints for kubernetes service: %v", ep)
		}
	} else {
		glog.Errorf("No RW service found: %v", found)
		glog.Fatal("Kubernetes service test failed")
	}
	glog.Infof("Master service test passed.")
}

func runSchedulerNoPhantomPodsTest(client *client.Client) {
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c1",
					Image: "gcr.io/google_containers/pause-amd64:3.0",
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
	if err := wait.Poll(time.Second, longTestTimeout, podRunning(client, foo.Namespace, foo.Name)); err != nil {
		glog.Fatalf("FAILED: pod never started running %v", err)
	}

	pod.ObjectMeta.Name = "phantom.bar"
	bar, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		glog.Fatalf("Failed to create pod: %v, %v", pod, err)
	}
	if err := wait.Poll(time.Second, longTestTimeout, podRunning(client, bar.Namespace, bar.Name)); err != nil {
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
	if err := wait.Poll(time.Second, longTestTimeout, podRunning(client, baz.Namespace, baz.Name)); err != nil {
		if pod, perr := client.Pods(api.NamespaceDefault).Get("phantom.bar"); perr == nil {
			glog.Fatalf("FAILED: 'phantom.bar' was never deleted: %#v, err: %v", pod, err)
		} else {
			glog.Fatalf("FAILED: (Scheduler probably didn't process deletion of 'phantom.bar') Pod never started running: err: %v, perr: %v", err, perr)
		}
	}

	glog.Info("Scheduler doesn't make phantom pods: test passed.")
}

type testFunc func(*client.Client)

func addFlags(fs *pflag.FlagSet) {
	fs.IntVar(
		&maxConcurrency, "max-concurrency", -1, "Maximum number of tests to be run simultaneously. Unlimited if set to negative.")
	fs.BoolVar(
		&watchCache, "watch-cache", false, "Turn on watch cache on API server.")
}

func main() {
	gruntime.GOMAXPROCS(gruntime.NumCPU())
	addFlags(pflag.CommandLine)

	flag.InitFlags()
	utilruntime.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(maxTestTimeout)
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

	kubeClient := client.NewOrDie(
		&restclient.Config{
			Host:          apiServerURL,
			ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()},
			QPS:           20,
			Burst:         50,
		})
	// TODO: caesarxuchao: hacky way to specify version of Experimental client.
	// We will fix this by supporting multiple group versions in Config
	kubeClient.ExtensionsClient = client.NewExtensionsOrDie(
		&restclient.Config{
			Host:          apiServerURL,
			ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()},
			QPS:           20,
			Burst:         50,
		})

	// Run tests in parallel
	testFuncs := []testFunc{
		runAtomicPutTest,
		runPatchTest,
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
	// We expect 6 containers:
	//              1 pod infra container + 2 containers from the URL on first Kubelet +
	//              1 pod infra container + 2 containers from the URL on second Kubelet +
	// The total number of container created is 6

	if len(createdConts) != 6 {
		glog.Fatalf("Expected 6 containers; got %v\n\nlist of created containers:\n\n%#v\n\nDocker 1 Created:\n\n%#v\n\nDocker 2 Created:\n\n%#v\n\n", len(createdConts), createdConts.List(), fakeDocker1.Created, fakeDocker2.Created)
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
					"image": "gcr.io/google_containers/redis:e2e",
					"volumeMounts": [{
						"name": "redis-data",
						"mountPath": "/data"
					}]
				},
				{
					"name": "guestbook",
					"image": "gcr.io/google_samples/gb-frontend:v3",
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
