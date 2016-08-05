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
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	gruntime "runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
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
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"

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

	podInformer := informers.CreateSharedPodIndexInformer(clientset, controller.NoResyncPeriodFunc())

	// ensure the service endpoints are sync'd several times within the window that the integration tests wait
	go endpointcontroller.NewEndpointController(podInformer, clientset).
		Run(3, wait.NeverStop)

	// TODO: Write an integration test for the replication controllers watch.
	go replicationcontroller.NewReplicationManager(podInformer, clientset, controller.NoResyncPeriodFunc, replicationcontroller.BurstReplicas, 4096).
		Run(3, wait.NeverStop)

	go podInformer.Run(wait.NeverStop)

	nodeController, err := nodecontroller.NewNodeController(nil, clientset, 5*time.Minute, flowcontrol.NewFakeAlwaysRateLimiter(), flowcontrol.NewFakeAlwaysRateLimiter(),
		40*time.Second, 60*time.Second, 5*time.Second, nil, nil, 0, false)
	if err != nil {
		glog.Fatalf("Failed to initialize nodecontroller: %v", err)
	}
	nodeController.Run(5 * time.Second)
	cadvisorInterface := new(cadvisortest.Fake)

	// Kubelet (localhost)
	testRootDir := testutils.MakeTempDirOrDie("kubelet_integ_1.", "")
	configFilePath := testutils.MakeTempDirOrDie("config", testRootDir)
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
		&containertest.FakeOS{},
		1*time.Second,  /* FileCheckFrequency */
		1*time.Second,  /* HTTPCheckFrequency */
		10*time.Second, /* MinimumGCAge */
		3*time.Second,  /* NodeStatusUpdateFrequency */
		10*time.Second, /* SyncFrequency */
		10*time.Second, /* OutOfDiskTransitionFrequency */
		10*time.Second, /* EvictionPressureTransitionPeriod */
		40,             /* MaxPods */
		0,              /* PodsPerCore*/
		cm, net.ParseIP("127.0.0.1"))

	kubeletapp.RunKubelet(kcfg)
	// Kubelet (machine)
	// Create a second kubelet so that the guestbook example's two redis slaves both
	// have a place they can schedule.
	testRootDir = testutils.MakeTempDirOrDie("kubelet_integ_2.", "")
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
		&containertest.FakeOS{},
		1*time.Second,  /* FileCheckFrequency */
		1*time.Second,  /* HTTPCheckFrequency */
		10*time.Second, /* MinimumGCAge */
		3*time.Second,  /* NodeStatusUpdateFrequency */
		10*time.Second, /* SyncFrequency */
		10*time.Second, /* OutOfDiskTransitionFrequency */
		10*time.Second, /* EvictionPressureTransitionPeriod */
		40,             /* MaxPods */
		0,              /* PodsPerCore*/
		cm,
		net.ParseIP("127.0.0.1"))

	kubeletapp.RunKubelet(kcfg)
	return apiServer.URL, configFilePath
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

func runSchedulerNoPhantomPodsTest(client *client.Client) {
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c1",
					Image: e2e.GetPauseImageName(client),
					Ports: []api.ContainerPort{
						{ContainerPort: 1234, HostPort: 9999},
					},
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
		},
	}

	// Assuming we only have two kubelets, the third pod here won't schedule
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
	testFuncs := []testFunc{}

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
