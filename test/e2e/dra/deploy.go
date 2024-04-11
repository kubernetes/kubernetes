/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
	"google.golang.org/grpc"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	resourceapiinformer "k8s.io/client-go/informers/resource/v1alpha2"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/drivers/proxy"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	NodePrepareResourcesMethod   = "/v1alpha3.Node/NodePrepareResources"
	NodeUnprepareResourcesMethod = "/v1alpha3.Node/NodeUnprepareResources"
)

type Nodes struct {
	NodeNames []string
}

// NewNodes selects nodes to run the test on.
func NewNodes(f *framework.Framework, minNodes, maxNodes int) *Nodes {
	nodes := &Nodes{}
	ginkgo.BeforeEach(func(ctx context.Context) {
		ginkgo.By("selecting nodes")
		// The kubelet plugin is harder. We deploy the builtin manifest
		// after patching in the driver name and all nodes on which we
		// want the plugin to run.
		//
		// Only a subset of the nodes are picked to avoid causing
		// unnecessary load on a big cluster.
		nodeList, err := e2enode.GetBoundedReadySchedulableNodes(ctx, f.ClientSet, maxNodes)
		framework.ExpectNoError(err, "get nodes")
		numNodes := int32(len(nodeList.Items))
		if int(numNodes) < minNodes {
			e2eskipper.Skipf("%d ready nodes required, only have %d", minNodes, numNodes)
		}
		nodes.NodeNames = nil
		for _, node := range nodeList.Items {
			nodes.NodeNames = append(nodes.NodeNames, node.Name)
		}
		framework.Logf("testing on nodes %v", nodes.NodeNames)

		// Watch claims in the namespace. This is useful for monitoring a test
		// and enables additional sanity checks.
		claimInformer := resourceapiinformer.NewResourceClaimInformer(f.ClientSet, f.Namespace.Name, 100*time.Hour /* resync */, nil)
		cancelCtx, cancel := context.WithCancelCause(context.Background())
		var wg sync.WaitGroup
		ginkgo.DeferCleanup(func() {
			cancel(errors.New("test has completed"))
			wg.Wait()
		})
		_, err = claimInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj any) {
				defer ginkgo.GinkgoRecover()
				claim := obj.(*resourcev1alpha2.ResourceClaim)
				framework.Logf("New claim:\n%s", format.Object(claim, 1))
				validateClaim(claim)
			},
			UpdateFunc: func(oldObj, newObj any) {
				defer ginkgo.GinkgoRecover()
				oldClaim := oldObj.(*resourcev1alpha2.ResourceClaim)
				newClaim := newObj.(*resourcev1alpha2.ResourceClaim)
				framework.Logf("Updated claim:\n%s\nDiff:\n%s", format.Object(newClaim, 1), cmp.Diff(oldClaim, newClaim))
				validateClaim(newClaim)
			},
			DeleteFunc: func(obj any) {
				defer ginkgo.GinkgoRecover()
				claim := obj.(*resourcev1alpha2.ResourceClaim)
				framework.Logf("Deleted claim:\n%s", format.Object(claim, 1))
			},
		})
		framework.ExpectNoError(err, "AddEventHandler")
		wg.Add(1)
		go func() {
			defer wg.Done()
			claimInformer.Run(cancelCtx.Done())
		}()
	})
	return nodes
}

func validateClaim(claim *resourcev1alpha2.ResourceClaim) {
	// The apiserver doesn't enforce that a claim always has a finalizer
	// while being allocated. This is a convention that whoever allocates a
	// claim has to follow to prevent using a claim that is at risk of
	// being deleted.
	if claim.Status.Allocation != nil && len(claim.Finalizers) == 0 {
		framework.Failf("Invalid claim: allocated without any finalizer:\n%s", format.Object(claim, 1))
	}
}

// NewDriver sets up controller (as client of the cluster) and
// kubelet plugin (via proxy) before the test runs. It cleans
// up after the test.
func NewDriver(f *framework.Framework, nodes *Nodes, configureResources func() app.Resources) *Driver {
	d := &Driver{
		f:            f,
		fail:         map[MethodInstance]bool{},
		callCounts:   map[MethodInstance]int64{},
		NodeV1alpha3: true,
	}

	ginkgo.BeforeEach(func() {
		resources := configureResources()
		if len(resources.Nodes) == 0 {
			// This always has to be set because the driver might
			// not run on all nodes.
			resources.Nodes = nodes.NodeNames
		}
		ginkgo.DeferCleanup(d.IsGone) // Register first so it gets called last.
		d.SetUp(nodes, resources)
		ginkgo.DeferCleanup(d.TearDown)
	})
	return d
}

type MethodInstance struct {
	Nodename   string
	FullMethod string
}

type Driver struct {
	f       *framework.Framework
	ctx     context.Context
	cleanup []func() // executed first-in-first-out
	wg      sync.WaitGroup

	NameSuffix string
	Controller *app.ExampleController
	Name       string
	Nodes      map[string]*app.ExamplePlugin

	parameterMode         parameterMode
	parameterAPIGroup     string
	parameterAPIVersion   string
	claimParameterAPIKind string
	classParameterAPIKind string

	NodeV1alpha3 bool

	mutex      sync.Mutex
	fail       map[MethodInstance]bool
	callCounts map[MethodInstance]int64
}

type parameterMode string

const (
	parameterModeConfigMap  parameterMode = "configmap"  // ConfigMap parameters, control plane controller.
	parameterModeStructured parameterMode = "structured" // No ConfigMaps, directly create and reference in-tree parameter objects.
	parameterModeTranslated parameterMode = "translated" // Reference ConfigMaps in claim and class, generate in-tree parameter objects.
)

func (d *Driver) SetUp(nodes *Nodes, resources app.Resources) {
	ginkgo.By(fmt.Sprintf("deploying driver on nodes %v", nodes.NodeNames))
	d.Nodes = map[string]*app.ExamplePlugin{}
	d.Name = d.f.UniqueName + d.NameSuffix + ".k8s.io"
	resources.DriverName = d.Name

	ctx, cancel := context.WithCancel(context.Background())
	if d.NameSuffix != "" {
		logger := klog.FromContext(ctx)
		logger = klog.LoggerWithName(logger, "instance"+d.NameSuffix)
		ctx = klog.NewContext(ctx, logger)
	}
	d.ctx = ctx
	d.cleanup = append(d.cleanup, cancel)

	switch d.parameterMode {
	case "", parameterModeConfigMap:
		// The controller is easy: we simply connect to the API server.
		d.Controller = app.NewController(d.f.ClientSet, resources)
		d.wg.Add(1)
		go func() {
			defer d.wg.Done()
			d.Controller.Run(d.ctx, 5 /* workers */)
		}()
	}

	manifests := []string{
		// The code below matches the content of this manifest (ports,
		// container names, etc.).
		"test/e2e/testing-manifests/dra/dra-test-driver-proxy.yaml",
	}
	if d.parameterMode == "" {
		d.parameterMode = parameterModeConfigMap
	}
	var numResourceInstances = -1 // disabled
	if d.parameterMode != parameterModeConfigMap {
		numResourceInstances = resources.MaxAllocations
	}
	switch d.parameterMode {
	case parameterModeConfigMap, parameterModeTranslated:
		d.parameterAPIGroup = ""
		d.parameterAPIVersion = "v1"
		d.claimParameterAPIKind = "ConfigMap"
		d.classParameterAPIKind = "ConfigMap"
	case parameterModeStructured:
		d.parameterAPIGroup = "resource.k8s.io"
		d.parameterAPIVersion = "v1alpha2"
		d.claimParameterAPIKind = "ResourceClaimParameters"
		d.classParameterAPIKind = "ResourceClassParameters"
	default:
		framework.Failf("unknown test driver parameter mode: %s", d.parameterMode)
	}

	instanceKey := "app.kubernetes.io/instance"
	rsName := ""
	draAddr := path.Join(framework.TestContext.KubeletRootDir, "plugins", d.Name+".sock")
	numNodes := int32(len(nodes.NodeNames))
	err := utils.CreateFromManifests(ctx, d.f, d.f.Namespace, func(item interface{}) error {
		switch item := item.(type) {
		case *appsv1.ReplicaSet:
			item.Name += d.NameSuffix
			rsName = item.Name
			item.Spec.Replicas = &numNodes
			item.Spec.Selector.MatchLabels[instanceKey] = d.Name
			item.Spec.Template.Labels[instanceKey] = d.Name
			item.Spec.Template.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].LabelSelector.MatchLabels[instanceKey] = d.Name
			item.Spec.Template.Spec.Affinity.NodeAffinity = &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "kubernetes.io/hostname",
									Operator: v1.NodeSelectorOpIn,
									Values:   nodes.NodeNames,
								},
							},
						},
					},
				},
			}
			item.Spec.Template.Spec.Volumes[0].HostPath.Path = path.Join(framework.TestContext.KubeletRootDir, "plugins")
			item.Spec.Template.Spec.Volumes[2].HostPath.Path = path.Join(framework.TestContext.KubeletRootDir, "plugins_registry")
			item.Spec.Template.Spec.Containers[0].Args = append(item.Spec.Template.Spec.Containers[0].Args, "--endpoint=/plugins_registry/"+d.Name+"-reg.sock")
			item.Spec.Template.Spec.Containers[1].Args = append(item.Spec.Template.Spec.Containers[1].Args, "--endpoint=/dra/"+d.Name+".sock")
		case *apiextensionsv1.CustomResourceDefinition:
			item.Name = strings.ReplaceAll(item.Name, "dra.e2e.example.com", d.parameterAPIGroup)
			item.Spec.Group = d.parameterAPIGroup

		}
		return nil
	}, manifests...)
	framework.ExpectNoError(err, "deploy kubelet plugin replicaset")

	rs, err := d.f.ClientSet.AppsV1().ReplicaSets(d.f.Namespace.Name).Get(ctx, rsName, metav1.GetOptions{})
	framework.ExpectNoError(err, "get replicaset")

	// Wait for all pods to be running.
	if err := e2ereplicaset.WaitForReplicaSetTargetAvailableReplicas(ctx, d.f.ClientSet, rs, numNodes); err != nil {
		framework.ExpectNoError(err, "all kubelet plugin proxies running")
	}
	requirement, err := labels.NewRequirement(instanceKey, selection.Equals, []string{d.Name})
	framework.ExpectNoError(err, "create label selector requirement")
	selector := labels.NewSelector().Add(*requirement)
	pods, err := d.f.ClientSet.CoreV1().Pods(d.f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: selector.String()})
	framework.ExpectNoError(err, "list proxy pods")
	gomega.Expect(numNodes).To(gomega.Equal(int32(len(pods.Items))), "number of proxy pods")

	// Run registar and plugin for each of the pods.
	for _, pod := range pods.Items {
		// Need a local variable, not the loop variable, for the anonymous
		// callback functions below.
		pod := pod
		nodename := pod.Spec.NodeName
		logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "kubelet plugin"), "node", pod.Spec.NodeName, "pod", klog.KObj(&pod))
		loggerCtx := klog.NewContext(ctx, logger)
		plugin, err := app.StartPlugin(loggerCtx, "/cdi", d.Name, d.f.ClientSet, nodename,
			app.FileOperations{
				Create: func(name string, content []byte) error {
					klog.Background().Info("creating CDI file", "node", nodename, "filename", name, "content", string(content))
					return d.createFile(&pod, name, content)
				},
				Remove: func(name string) error {
					klog.Background().Info("deleting CDI file", "node", nodename, "filename", name)
					return d.removeFile(&pod, name)
				},
				NumResourceInstances: numResourceInstances,
			},
			kubeletplugin.GRPCVerbosity(0),
			kubeletplugin.GRPCInterceptor(func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
				return d.interceptor(nodename, ctx, req, info, handler)
			}),
			kubeletplugin.GRPCStreamInterceptor(func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) (err error) {
				return d.streamInterceptor(nodename, srv, ss, info, handler)
			}),
			kubeletplugin.PluginListener(listen(ctx, d.f, pod.Name, "plugin", 9001)),
			kubeletplugin.RegistrarListener(listen(ctx, d.f, pod.Name, "registrar", 9000)),
			kubeletplugin.KubeletPluginSocketPath(draAddr),
			kubeletplugin.NodeV1alpha3(d.NodeV1alpha3),
		)
		framework.ExpectNoError(err, "start kubelet plugin for node %s", pod.Spec.NodeName)
		d.cleanup = append(d.cleanup, func() {
			// Depends on cancel being called first.
			plugin.Stop()
		})
		d.Nodes[nodename] = plugin
	}

	// Wait for registration.
	ginkgo.By("wait for plugin registration")
	gomega.Eventually(func() map[string][]app.GRPCCall {
		notRegistered := make(map[string][]app.GRPCCall)
		for nodename, plugin := range d.Nodes {
			calls := plugin.GetGRPCCalls()
			if contains, err := app.BeRegistered.Match(calls); err != nil || !contains {
				notRegistered[nodename] = calls
			}
		}
		return notRegistered
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "hosts where the plugin has not been registered yet")
}

func (d *Driver) createFile(pod *v1.Pod, name string, content []byte) error {
	buffer := bytes.NewBuffer(content)
	// Writing the content can be slow. Better create a temporary file and
	// move it to the final destination once it is complete.
	tmpName := name + ".tmp"
	if err := d.podIO(pod).CreateFile(tmpName, buffer); err != nil {
		_ = d.podIO(pod).RemoveAll(tmpName)
		return err
	}
	return d.podIO(pod).Rename(tmpName, name)
}

func (d *Driver) removeFile(pod *v1.Pod, name string) error {
	return d.podIO(pod).RemoveAll(name)
}

func (d *Driver) podIO(pod *v1.Pod) proxy.PodDirIO {
	logger := klog.Background()
	return proxy.PodDirIO{
		F:             d.f,
		Namespace:     pod.Namespace,
		PodName:       pod.Name,
		ContainerName: "plugin",
		Logger:        &logger,
	}
}

func listen(ctx context.Context, f *framework.Framework, podName, containerName string, port int) net.Listener {
	addr := proxy.Addr{
		Namespace:     f.Namespace.Name,
		PodName:       podName,
		ContainerName: containerName,
		Port:          port,
	}
	listener, err := proxy.Listen(ctx, f.ClientSet, f.ClientConfig(), addr)
	framework.ExpectNoError(err, "listen for connections from %+v", addr)
	return listener
}

func (d *Driver) TearDown() {
	for _, c := range d.cleanup {
		c()
	}
	d.cleanup = nil
	d.wg.Wait()
}

func (d *Driver) IsGone(ctx context.Context) {
	gomega.Eventually(ctx, func(ctx context.Context) ([]resourcev1alpha2.ResourceSlice, error) {
		slices, err := d.f.ClientSet.ResourceV1alpha2().ResourceSlices().List(ctx, metav1.ListOptions{FieldSelector: "driverName=" + d.Name})
		if err != nil {
			return nil, err
		}
		return slices.Items, err
	}).Should(gomega.BeEmpty())
}

func (d *Driver) interceptor(nodename string, ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	m := MethodInstance{nodename, info.FullMethod}
	d.callCounts[m]++
	if d.fail[m] {
		return nil, errors.New("injected error")
	}

	return handler(ctx, req)
}

func (d *Driver) streamInterceptor(nodename string, srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	// Stream calls block for a long time. We must not hold the lock while
	// they are running.
	d.mutex.Lock()
	m := MethodInstance{nodename, info.FullMethod}
	d.callCounts[m]++
	fail := d.fail[m]
	d.mutex.Unlock()

	if fail {
		return errors.New("injected error")
	}

	return handler(srv, stream)
}

func (d *Driver) Fail(m MethodInstance, injectError bool) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.fail[m] = injectError
}

func (d *Driver) CallCount(m MethodInstance) int64 {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	return d.callCounts[m]
}

func (d *Driver) Nodenames() (nodenames []string) {
	for nodename := range d.Nodes {
		nodenames = append(nodenames, nodename)
	}
	sort.Strings(nodenames)
	return
}
