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

package utils

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
	"google.golang.org/grpc"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/selection"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	cgoresource "k8s.io/client-go/kubernetes/typed/resource/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/cache"
	draclient "k8s.io/dynamic-resource-allocation/client"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/cmd/exec"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/deploy/example"
	testdrivergomega "k8s.io/kubernetes/test/e2e/dra/test-driver/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/drivers/proxy"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

type Nodes struct {
	// NodeNames has the main set of node names.
	NodeNames []string
	tempDir   string
	// NumReservedNodes specifies the desired number of
	// extra nodes that get set aside. That many node names
	// will be stored in ExtraNodeNames.
	//
	// Must be <= the minimum number of requested nodes.
	NumReservedNodes int
	// ExtraNodeNames has exactly as many node names as
	// requested via NumReservedNodes. Those nodes are
	// different than the nodes listed in NodeNames.
	ExtraNodeNames []string
}

// NewNodes selects nodes to run the test on.
//
// Call this outside of ginkgo.It, then use the instance inside ginkgo.It.
func NewNodes(f *framework.Framework, minNodes, maxNodes int) *Nodes {
	nodes := &Nodes{}
	ginkgo.BeforeEach(func(ctx context.Context) {
		nodes.init(ctx, f, minNodes, maxNodes)
	})
	return nodes
}

// NewNodesNow is a variant of NewNodes which can be used inside a ginkgo.It.
func NewNodesNow(ctx context.Context, f *framework.Framework, minNodes, maxNodes int) *Nodes {
	nodes := &Nodes{}
	nodes.init(ctx, f, minNodes, maxNodes)
	return nodes
}

func (nodes *Nodes) init(ctx context.Context, f *framework.Framework, minNodes, maxNodes int) {
	nodes.tempDir = ginkgo.GinkgoT().TempDir()

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
		e2eskipper.Skipf("%d ready nodes required, only have %d", minNodes+nodes.NumReservedNodes, numNodes)
	}
	nodes.NodeNames = nil
	for i, node := range nodeList.Items {
		if i < nodes.NumReservedNodes {
			nodes.ExtraNodeNames = append(nodes.ExtraNodeNames, node.Name)
			continue
		}
		nodes.NodeNames = append(nodes.NodeNames, node.Name)
	}
	sort.Strings(nodes.NodeNames)
	framework.Logf("testing on nodes %v", nodes.NodeNames)

	// Watch claims in the namespace. This is useful for monitoring a test
	// and enables additional sanity checks.
	resourceClaimLogger := klog.LoggerWithName(klog.FromContext(ctx), "ResourceClaimListWatch")
	var resourceClaimWatchCounter atomic.Int32
	resourceClient := draclient.New(f.ClientSet)
	claimInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
				slices, err := resourceClient.ResourceClaims("").List(ctx, options)
				if err == nil {
					resourceClaimLogger.Info("Listed ResourceClaims", "resourceAPI", resourceClient.CurrentAPI(), "numClaims", len(slices.Items), "listMeta", slices.ListMeta)
				} else {
					resourceClaimLogger.Info("Listing ResourceClaims failed", "resourceAPI", resourceClient.CurrentAPI(), "err", err)
				}
				return slices, err
			},
			WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
				w, err := resourceClient.ResourceClaims("").Watch(ctx, options)
				if err == nil {
					resourceClaimLogger.Info("Started watching ResourceClaims", "resourceAPI", resourceClient.CurrentAPI())
					wrapper := newWatchWrapper(klog.LoggerWithName(resourceClaimLogger, fmt.Sprintf("%d", resourceClaimWatchCounter.Load())), w)
					resourceClaimWatchCounter.Add(1)
					go wrapper.run()
					w = wrapper
				} else {
					resourceClaimLogger.Info("Watching ResourceClaims failed", "resourceAPI", resourceClient.CurrentAPI(), "err", err)
				}
				return w, err
			},
		},
		&resourceapi.ResourceClaim{},
		// No resync because all it would do is periodically trigger syncing pools
		// again by reporting all slices as updated with the object as old/new.
		0,
		nil,
	)
	cancelCtx, cancel := context.WithCancelCause(context.Background())
	var wg sync.WaitGroup
	ginkgo.DeferCleanup(func() {
		cancel(errors.New("test has completed"))
		wg.Wait()
	})
	_, err = claimInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			defer ginkgo.GinkgoRecover()
			claim := obj.(*resourceapi.ResourceClaim)
			resourceClaimLogger.Info("New claim", "claim", format.Object(claim, 0))
			validateClaim(claim)
		},
		UpdateFunc: func(oldObj, newObj any) {
			defer ginkgo.GinkgoRecover()
			oldClaim := oldObj.(*resourceapi.ResourceClaim)
			newClaim := newObj.(*resourceapi.ResourceClaim)
			resourceClaimLogger.Info("Updated claim", "newClaim", format.Object(newClaim, 0), "diff", cmp.Diff(oldClaim, newClaim))
			validateClaim(newClaim)
		},
		DeleteFunc: func(obj any) {
			defer ginkgo.GinkgoRecover()
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			claim := obj.(*resourceapi.ResourceClaim)
			resourceClaimLogger.Info("Deleted claim", "claim", format.Object(claim, 0))
		},
	})
	framework.ExpectNoError(err, "AddEventHandler")
	wg.Add(1)
	go func() {
		defer wg.Done()
		claimInformer.Run(cancelCtx.Done())
	}()
}

type watchWrapper struct {
	logger     klog.Logger
	delegate   watch.Interface
	resultChan chan watch.Event
}

func newWatchWrapper(logger klog.Logger, delegate watch.Interface) *watchWrapper {
	return &watchWrapper{
		logger:     logger,
		delegate:   delegate,
		resultChan: make(chan watch.Event, 100),
	}
}

func (w *watchWrapper) run() {
	defer utilruntime.HandleCrashWithLogger(w.logger)
	defer close(w.resultChan)
	inputChan := w.delegate.ResultChan()
	for {
		event, ok := <-inputChan
		if !ok {
			w.logger.Info("Wrapped result channel was closed, stopping event forwarding")
			return
		}
		w.logger.Info("Received event", "event", event.Type, "content", fmt.Sprintf("%T", event.Object))
		w.resultChan <- event
	}
}

func (w *watchWrapper) Stop() {
	w.delegate.Stop()
}

func (w *watchWrapper) ResultChan() <-chan watch.Event {
	return w.resultChan
}

func validateClaim(claim *resourceapi.ResourceClaim) {
	// The apiserver doesn't enforce that a claim always has a finalizer
	// while being allocated. This is a convention that whoever allocates a
	// claim has to follow to prevent using a claim that is at risk of
	// being deleted.
	if claim.Status.Allocation != nil && len(claim.Finalizers) == 0 {
		framework.Failf("Invalid claim: allocated without any finalizer:\n%s", format.Object(claim, 1))
	}
}

const (
	// multiHostDriverResources identifies DriverResources that are associated with multiple devices, i.e.
	// is not managed by a driver. So any Pools and ResourceSlices will be published directly
	// to the cluster, rather than through the driver.
	multiHostDriverResources = "multi-host"
)

// driverResourcesGenFunc defines the callback that will be invoked by the driver to generate the
// DriverResources that will be used to construct the ResourceSlices.
type driverResourcesGenFunc func(nodes *Nodes) map[string]resourceslice.DriverResources

// driverResourcesMutatorFunc defines the function signature for mutators that will
// update the DriverResources after they have been generated.
type driverResourcesMutatorFunc func(map[string]resourceslice.DriverResources)

// NewDriver sets up controller (as client of the cluster) and
// kubelet plugin (via proxy) before the test runs. It cleans
// up after the test.
//
// Call this outside of ginkgo.It, then use the instance inside ginkgo.It.
func NewDriver(f *framework.Framework, nodes *Nodes, driverResourcesGenerator driverResourcesGenFunc, driverResourcesMutators ...driverResourcesMutatorFunc) *Driver {
	d := NewDriverInstance(f)

	ginkgo.BeforeEach(func() {
		driverResources := driverResourcesGenerator(nodes)
		for _, mutator := range driverResourcesMutators {
			mutator(driverResources)
		}
		d.Run(nodes, driverResources)
	})
	return d
}

// NewDriverInstance is a variant of NewDriver where the driver is inactive and must
// be started explicitly with Run. May be used inside ginkgo.It.
func NewDriverInstance(f *framework.Framework) *Driver {
	d := &Driver{
		f:          f,
		fail:       map[MethodInstance]bool{},
		callCounts: map[MethodInstance]int64{},
		// By default, test with all gRPC APIs.
		// TODO: should setting this be optional to test the actual helper defaults?
		NodeV1:      true,
		NodeV1beta1: true,
		// By default, assume that the kubelet supports DRA and that
		// the driver's removal causes ResourceSlice cleanup.
		WithKubelet:                true,
		ExpectResourceSliceRemoval: true,
	}
	d.initName()
	return d
}

// ClientV1 returns a wrapper for client-go which provides the V1 API on top of whatever is enabled in the cluster.
func (d *Driver) ClientV1() cgoresource.ResourceV1Interface {
	return draclient.New(d.f.ClientSet)
}

func (d *Driver) Run(nodes *Nodes, driverResources map[string]resourceslice.DriverResources) {
	d.SetUp(nodes, driverResources)
	ginkgo.DeferCleanup(d.TearDown)
}

// NewGetSlices generates a function for gomega.Eventually/Consistently which
// returns the ResourceSliceList.
func (d *Driver) NewGetSlices() framework.GetFunc[*resourceapi.ResourceSliceList] {
	return framework.ListObjects(d.ClientV1().ResourceSlices().List, metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + d.Name})
}

type MethodInstance struct {
	NodeName   string
	FullMethod string
}

type Driver struct {
	f                  *framework.Framework
	ctx                context.Context
	cleanup            []func(context.Context) // executed first-in-first-out
	wg                 sync.WaitGroup
	serviceAccountName string

	// NameSuffix can be set while registering a test to deploy different
	// drivers in the same test namespace.
	NameSuffix string

	// InstanceSuffix can be set while registering a test to deploy two different
	// instances of the same driver. Used to generate unique objects in the API server.
	// The socket path is still the same.
	InstanceSuffix string

	// RollingUpdate can be set to true to enable using different socket names
	// for different pods and thus seamless upgrades. Must be supported by the kubelet!
	RollingUpdate bool

	// Normally, tearing down the driver should cause ResourceSlices to get removed eventually.
	// The exception is when the driver is part of a rolling update and is torn down first.
	ExpectResourceSliceRemoval bool

	// Name gets derived automatically from the current test namespace and
	// (if set) the NameSuffix while setting up the driver for a test.
	Name string

	// Nodes contains entries for each node selected for a test when the test runs.
	// In addition, there is one entry for a fictional node.
	Nodes map[string]KubeletPlugin

	// IsLocal can be set to true when using local-up-cluster.sh *and* ensuring
	// that /var/lib/kubelet/plugins, /var/lib/kubelet/plugins_registry and
	// /var/run/cdi are writable by the current user.
	IsLocal bool

	NodeV1      bool
	NodeV1beta1 bool

	// Register the DRA test driver with the kubelet and expect DRA to work (= feature.DynamicResourceAllocation).
	WithKubelet bool

	mutex      sync.Mutex
	fail       map[MethodInstance]bool
	callCounts map[MethodInstance]int64
}

type KubeletPlugin struct {
	*app.ExamplePlugin
	ClientSet kubernetes.Interface
}

func (d *Driver) initName() {
	d.Name = d.f.UniqueName + d.NameSuffix + ".k8s.io"
}

func (d *Driver) SetUp(nodes *Nodes, driverResources map[string]resourceslice.DriverResources) {
	d.initName()
	ginkgo.By(fmt.Sprintf("deploying driver %s on nodes %v", d.Name, nodes.NodeNames))
	d.Nodes = make(map[string]KubeletPlugin)

	ctx, cancel := context.WithCancel(context.Background())
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "driverName", d.Name)
	if d.InstanceSuffix != "" {
		instance, _ := strings.CutPrefix(d.InstanceSuffix, "-")
		logger = klog.LoggerWithValues(logger, "instance", instance)
	}
	ctx = klog.NewContext(ctx, logger)
	d.ctx = ctx
	d.cleanup = append(d.cleanup, func(context.Context) { cancel() })

	// After shutdown, check that all ResourceSlices were removed, either by the kubelet
	// or our own test code. This runs last because it gets registered first.
	if d.ExpectResourceSliceRemoval {
		ginkgo.DeferCleanup(d.IsGone)
	}

	driverResource, useMultiHostDriverResources := driverResources[multiHostDriverResources]
	if useMultiHostDriverResources || !d.WithKubelet {
		// We have to remove ResourceSlices ourselves.
		// Otherwise the kubelet does it after unregistering the driver.
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := d.f.ClientSet.ResourceV1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + d.Name})
			framework.ExpectNoError(err, "delete ResourceSlices of the driver")
		})
	}

	// If found, we create ResourceSlices that are associated with multiple nodes
	// through the node selector. Thus, the ResourceSlices are published here
	// rather than through the driver on a specific node.
	if useMultiHostDriverResources {
		for poolName, pool := range driverResource.Pools {
			for i, slice := range pool.Slices {
				resourceSlice := &resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("%s-%d", d.Name, i), // globally unique
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: d.Name,
						Pool: resourceapi.ResourcePool{
							Name:               poolName,
							Generation:         pool.Generation,
							ResourceSliceCount: int64(len(pool.Slices)),
						},
						NodeSelector: pool.NodeSelector,
						Devices:      slice.Devices,
					},
				}
				_, err := d.f.ClientSet.ResourceV1().ResourceSlices().Create(ctx, resourceSlice, metav1.CreateOptions{})
				framework.ExpectNoError(err)
			}
		}
	}

	manifests := []string{
		// The code below matches the content of this manifest (ports,
		// container names, etc.).
		"test/e2e/testing-manifests/dra/dra-test-driver-proxy.yaml",
	}

	// Create service account and corresponding RBAC rules.
	d.serviceAccountName = "dra-kubelet-plugin-" + d.Name + d.InstanceSuffix + "-service-account"
	content := example.PluginPermissions
	content = strings.ReplaceAll(content, "dra-kubelet-plugin-namespace", d.f.Namespace.Name)
	content = strings.ReplaceAll(content, "dra-kubelet-plugin", "dra-kubelet-plugin-"+d.Name+d.InstanceSuffix)
	d.createFromYAML(ctx, []byte(content), d.f.Namespace.Name)

	// Using a ReplicaSet instead of a DaemonSet has the advantage that we can control
	// the lifecycle explicitly, in particular run two pods per node long enough to
	// run checks.
	instanceKey := "app.kubernetes.io/instance"
	rsName := ""
	numNodes := int32(len(nodes.NodeNames))
	pluginDataDirectoryPath := path.Join(framework.TestContext.KubeletRootDir, "plugins", d.Name)
	registrarDirectoryPath := path.Join(framework.TestContext.KubeletRootDir, "plugins_registry")
	instanceName := d.Name + d.InstanceSuffix
	err := utils.CreateFromManifests(ctx, d.f, d.f.Namespace, func(item interface{}) error {
		switch item := item.(type) {
		case *appsv1.ReplicaSet:
			item.Name += d.NameSuffix + d.InstanceSuffix
			rsName = item.Name
			item.Spec.Replicas = &numNodes
			item.Spec.Selector.MatchLabels[instanceKey] = instanceName
			item.Spec.Template.Labels[instanceKey] = instanceName
			item.Spec.Template.Spec.ServiceAccountName = d.serviceAccountName
			item.Spec.Template.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0].LabelSelector.MatchLabels[instanceKey] = instanceName
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
			if d.IsLocal {
				// Drop mounting of directories. All operations run locally.
				item.Spec.Template.Spec.Volumes = nil
				item.Spec.Template.Spec.Containers[0].VolumeMounts = nil
				// No privileges required either.
				item.Spec.Template.Spec.SecurityContext = nil
				item.Spec.Template.Spec.Containers[0].SecurityContext = nil
			} else {
				item.Spec.Template.Spec.Volumes[0].HostPath.Path = pluginDataDirectoryPath
				item.Spec.Template.Spec.Volumes[1].HostPath.Path = registrarDirectoryPath
				item.Spec.Template.Spec.Containers[0].VolumeMounts[0].MountPath = pluginDataDirectoryPath
				item.Spec.Template.Spec.Containers[0].VolumeMounts[1].MountPath = registrarDirectoryPath
			}
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
	requirement, err := labels.NewRequirement(instanceKey, selection.Equals, []string{instanceName})
	framework.ExpectNoError(err, "create label selector requirement")
	selector := labels.NewSelector().Add(*requirement)
	pods, err := d.f.ClientSet.CoreV1().Pods(d.f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: selector.String()})
	framework.ExpectNoError(err, "list proxy pods")
	gomega.Expect(numNodes).To(gomega.Equal(int32(len(pods.Items))), "number of proxy pods")
	sort.Slice(pods.Items, func(i, j int) bool {
		return pods.Items[i].Spec.NodeName < pods.Items[j].Spec.NodeName
	})

	// Run registrar and plugin for each of the pods.
	for _, pod := range pods.Items {
		nodename := pod.Spec.NodeName

		// Authenticate the plugin so that it has the exact same
		// permissions as the daemonset pod. This includes RBAC and a
		// validating admission policy which limits writes to per-node
		// ResourceSlices.
		//
		// We could retrieve
		// /var/run/secrets/kubernetes.io/serviceaccount/token from
		// each pod and use it. That would check that
		// ServiceAccountTokenNodeBindingValidation works. But that's
		// better covered by a test owned by SIG Auth (like the one in
		// https://github.com/kubernetes/kubernetes/pull/124711).
		//
		// Here we merely use impersonation, which is faster.
		driverClient := d.ImpersonateKubeletPlugin(&pod)

		logger := klog.LoggerWithValues(klog.LoggerWithName(logger, "kubelet-plugin"), "node", pod.Spec.NodeName, "pod", klog.KObj(&pod))
		loggerCtx := klog.NewContext(ctx, logger)
		fileOps := app.FileOperations{
			Create: func(name string, content []byte) error {
				klog.Background().Info("creating CDI file", "node", nodename, "filename", name, "content", string(content))
				if d.IsLocal {
					// Name starts with /cdi, which is how it is mapped in the container.
					// Here we need it under /var/run.
					// Try to create /var/run/cdi, it might not exist yet.
					name = path.Join("/var/run", name)
					if err := os.MkdirAll(path.Dir(name), 0700); err != nil {
						return fmt.Errorf("create CDI directory: %w", err)
					}
					if err := os.WriteFile(name, content, 0644); err != nil {
						return fmt.Errorf("write CDI file: %w", err)
					}
					return nil
				}
				return d.createFile(&pod, name, content)
			},
			Remove: func(name string) error {
				klog.Background().Info("deleting CDI file", "node", nodename, "filename", name)
				if d.IsLocal {
					name = path.Join("/var/run", name)
					return os.Remove(name)
				}
				return d.removeFile(&pod, name)
			},
			HandleError: func(ctx context.Context, err error, msg string) {
				// Record a failure, but don't kill the background goroutine.
				defer ginkgo.GinkgoRecover()
				// During tests when canceling the context it is possible to get all kinds of
				// follow-up errors for that, like:
				//   processing ResourceSlice objects: retrieve node "127.0.0.1": client rate limiter Wait returned an error: context canceled
				//
				// The "context canceled" error was not wrapped, so `errors.Is` doesn't work.
				// Instead of trying to detect errors which can be ignored, let's only
				// treat errors as failures which definitely shouldn't occur:
				var droppedFields *resourceslice.DroppedFieldsError
				if errors.As(err, &droppedFields) {
					framework.Failf("driver %s: %v", d.Name, err)
				}
			},
		}

		if dr, ok := driverResources[nodename]; ok {
			fileOps.DriverResources = &dr
		}
		// All listeners running in this pod use a new unique local port number
		// by atomically incrementing this variable.
		listenerPort := int32(9000)
		rollingUpdateUID := pod.UID
		serialize := true
		if !d.RollingUpdate {
			rollingUpdateUID = ""
			// A test might have to execute two gRPC calls in parallel, so only
			// serialize when we explicitly want to test a rolling update.
			serialize = false
		}

		plugin, err := app.StartPlugin(loggerCtx, "/cdi", d.Name, driverClient, nodename, fileOps,
			kubeletplugin.GRPCVerbosity(0),
			kubeletplugin.GRPCInterceptor(func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
				return d.interceptor(nodename, ctx, req, info, handler)
			}),
			kubeletplugin.GRPCStreamInterceptor(func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) (err error) {
				return d.streamInterceptor(nodename, srv, ss, info, handler)
			}),
			kubeletplugin.NodeV1(d.NodeV1),
			kubeletplugin.NodeV1beta1(d.NodeV1beta1),

			kubeletplugin.RollingUpdate(rollingUpdateUID),
			kubeletplugin.Serialize(serialize),
			kubeletplugin.FlockDirectoryPath(nodes.tempDir),

			kubeletplugin.PluginDataDirectoryPath(pluginDataDirectoryPath),
			kubeletplugin.PluginListener(d.listen(&pod, &listenerPort)),

			kubeletplugin.RegistrarDirectoryPath(registrarDirectoryPath),
			kubeletplugin.RegistrarListener(d.listen(&pod, &listenerPort)),
		)
		framework.ExpectNoError(err, "start kubelet plugin for node %s", pod.Spec.NodeName)
		d.cleanup = append(d.cleanup, func(ctx context.Context) {
			// Depends on cancel being called first.
			plugin.Stop()

			// Also explicitly stop all pods.
			ginkgo.By("scaling down driver proxy pods for " + d.Name)
			rs, err := d.f.ClientSet.AppsV1().ReplicaSets(d.f.Namespace.Name).Get(ctx, rsName, metav1.GetOptions{})
			framework.ExpectNoError(err, "get ReplicaSet for driver "+d.Name)
			rs.Spec.Replicas = ptr.To(int32(0))
			rs, err = d.f.ClientSet.AppsV1().ReplicaSets(d.f.Namespace.Name).Update(ctx, rs, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "scale down ReplicaSet for driver "+d.Name)
			if err := e2ereplicaset.WaitForReplicaSetTargetAvailableReplicas(ctx, d.f.ClientSet, rs, 0); err != nil {
				framework.ExpectNoError(err, "all kubelet plugin proxies stopped")
			}
		})
		d.Nodes[nodename] = KubeletPlugin{ExamplePlugin: plugin, ClientSet: driverClient}
	}

	if !d.WithKubelet {
		return
	}

	// Wait for registration.
	ginkgo.By("wait for plugin registration")
	gomega.Eventually(func() map[string][]app.GRPCCall {
		notRegistered := make(map[string][]app.GRPCCall)
		for nodename, plugin := range d.Nodes {
			calls := plugin.GetGRPCCalls()
			if contains, err := testdrivergomega.BeRegistered.Match(calls); err != nil || !contains {
				notRegistered[nodename] = calls
			}
		}
		return notRegistered
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "hosts where the plugin has not been registered yet")
}

func (d *Driver) ImpersonateKubeletPlugin(pod *v1.Pod) kubernetes.Interface {
	ginkgo.GinkgoHelper()
	driverUserInfo := (&serviceaccount.ServiceAccountInfo{
		Name:      d.serviceAccountName,
		Namespace: pod.Namespace,
		NodeName:  pod.Spec.NodeName,
		PodName:   pod.Name,
		PodUID:    string(pod.UID),
	}).UserInfo()
	driverClientConfig := d.f.ClientConfig()
	driverClientConfig.Impersonate = rest.ImpersonationConfig{
		UserName: driverUserInfo.GetName(),
		Groups:   driverUserInfo.GetGroups(),
		Extra:    driverUserInfo.GetExtra(),
	}
	driverClient, err := kubernetes.NewForConfig(driverClientConfig)
	framework.ExpectNoError(err, "create client for driver")
	return driverClient
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

func (d *Driver) createFromYAML(ctx context.Context, content []byte, namespace string) {
	// Not caching the discovery result isn't very efficient, but good enough.
	discoveryCache := memory.NewMemCacheClient(d.f.ClientSet.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryCache)

	for _, content := range bytes.Split(content, []byte("---\n")) {
		if len(content) == 0 {
			continue
		}

		var obj *unstructured.Unstructured
		framework.ExpectNoError(yaml.UnmarshalStrict(content, &obj), fmt.Sprintf("Full YAML:\n%s\n", string(content)))

		gv, err := schema.ParseGroupVersion(obj.GetAPIVersion())
		framework.ExpectNoError(err, fmt.Sprintf("extract group+version from object %q", klog.KObj(obj)))
		gk := schema.GroupKind{Group: gv.Group, Kind: obj.GetKind()}

		mapping, err := restMapper.RESTMapping(gk, gv.Version)
		framework.ExpectNoError(err, fmt.Sprintf("map %q to resource", gk))

		resourceClient := d.f.DynamicClient.Resource(mapping.Resource)
		options := metav1.CreateOptions{
			// If the YAML input is invalid, then we want the
			// apiserver to tell us via an error. This can
			// happen because decoding into an unstructured object
			// doesn't validate.
			FieldValidation: "Strict",
		}
		switch mapping.Scope.Name() {
		case meta.RESTScopeNameRoot:
			_, err = resourceClient.Create(ctx, obj, options)
		case meta.RESTScopeNameNamespace:
			if namespace == "" {
				framework.Failf("need namespace for object type %s", gk)
			}
			_, err = resourceClient.Namespace(namespace).Create(ctx, obj, options)
		}
		framework.ExpectNoError(err, "create object")
		ginkgo.DeferCleanup(func(ctx context.Context) {
			del := resourceClient.Delete
			if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
				del = resourceClient.Namespace(namespace).Delete
			}
			err := del(ctx, obj.GetName(), metav1.DeleteOptions{})
			if !apierrors.IsNotFound(err) {
				framework.ExpectNoError(err, fmt.Sprintf("deleting %s.%s %s", obj.GetKind(), obj.GetAPIVersion(), klog.KObj(obj)))
			}
		})
	}
}

func (d *Driver) podIO(pod *v1.Pod) proxy.PodDirIO {
	logger := klog.Background()
	return proxy.PodDirIO{
		F:             d.f,
		Namespace:     pod.Namespace,
		PodName:       pod.Name,
		ContainerName: pod.Spec.Containers[0].Name,
		Logger:        &logger,
	}
}

// errListenerDone is the special error that we use to shut down.
// It doesn't need to be logged.
var errListenerDone = errors.New("listener is shutting down")

// listen returns the function which the kubeletplugin helper needs to open a listening socket.
// For that it spins up hostpathplugin in the pod for the desired node
// and connects to hostpathplugin via port forwarding.
func (d *Driver) listen(pod *v1.Pod, port *int32) func(ctx context.Context, endpoint string) (net.Listener, error) {
	return func(ctx context.Context, endpoint string) (l net.Listener, e error) {
		// No need create sockets, the kubelet is not expected to use them.
		if !d.WithKubelet {
			return newNullListener(), nil
		}

		// Try opening the socket directly on the local host. Falls back to pod if that fails.
		// Closing the listener will unlink the socket.
		if d.IsLocal {
			dir := path.Dir(endpoint)
			if err := os.MkdirAll(dir, 0755); err != nil {
				return nil, err
			}
			return net.ListenUnix("unix", &net.UnixAddr{Name: endpoint, Net: "unix"})
		}

		// "Allocate" a new port by by bumping the per-pod counter by one.
		port := atomic.AddInt32(port, 1)

		logger := klog.FromContext(ctx)
		logger = klog.LoggerWithName(logger, "socket-listener")
		logger = klog.LoggerWithValues(logger, "endpoint", endpoint, "port", port)
		ctx = klog.NewContext(ctx, logger)

		// Start hostpathplugin in proxy mode and keep it running until the listener gets closed.
		req := d.f.ClientSet.CoreV1().RESTClient().Post().
			Resource("pods").
			Namespace(d.f.Namespace.Name).
			Name(pod.Name).
			SubResource("exec").
			VersionedParams(&v1.PodExecOptions{
				Container: pod.Spec.Containers[0].Name,
				Command: []string{
					"/hostpathplugin",
					"--v=5",
					"--endpoint=" + endpoint,
					fmt.Sprintf("--proxy-endpoint=tcp://:%d", port),
				},
				Stdout: true,
				Stderr: true,
			}, scheme.ParameterCodec)
		var wg sync.WaitGroup
		wg.Add(1)
		cmdCtx, cmdCancel := context.WithCancelCause(ctx)
		go func() {
			defer wg.Done()
			cmdLogger := klog.LoggerWithName(logger, "hostpathplugin")
			cmdCtx := klog.NewContext(cmdCtx, cmdLogger)
			logger.V(1).Info("Starting...")
			defer logger.V(1).Info("Stopped")

			// This may fail temporarily, which is recoverable by executing again.
			delayFn := wait.Backoff{
				Duration: time.Second,
				Cap:      30 * time.Second,
				Steps:    30,
				Factor:   2.0,
				Jitter:   1.0,
			}.DelayWithReset(clock.RealClock{}, 5*time.Minute)
			runHostpathPlugin := func(ctx context.Context) (bool, error) {
				// errors.Is(err, listenerDoneErr) would be nicer, but we don't get
				// that error from remotecommand. Instead forgo logging when we already shut down.
				if err := execute(ctx, req.URL(), d.f.ClientConfig(), 5); err != nil && ctx.Err() == nil {
					klog.FromContext(ctx).V(5).Info("execution failed, will retry", "err", err)
				}
				// There is no reason to stop except for context cancellation =>
				// condition always false, no fatal errors.
				return false, nil
			}
			_ = delayFn.Until(cmdCtx, true /* immediate */, true /* sliding */, runHostpathPlugin)

			// Killing hostpathplugin does not remove the socket. Need to do that manually.
			req := d.f.ClientSet.CoreV1().RESTClient().Post().
				Resource("pods").
				Namespace(d.f.Namespace.Name).
				Name(pod.Name).
				SubResource("exec").
				VersionedParams(&v1.PodExecOptions{
					Container: pod.Spec.Containers[0].Name,
					Command: []string{
						"rm",
						"-f",
						endpoint,
					},
					Stdout: true,
					Stderr: true,
				}, scheme.ParameterCodec)
			cleanupLogger := klog.LoggerWithName(logger, "cleanup")
			cleanupCtx := klog.NewContext(ctx, cleanupLogger)
			if err := execute(cleanupCtx, req.URL(), d.f.ClientConfig(), 0); err != nil {
				cleanupLogger.Error(err, "Socket removal failed")
			}
		}()
		defer func() {
			// If we don't return a functional listener, then clean up.
			if e != nil {
				cmdCancel(e)
			}
		}()
		stopHostpathplugin := func() {
			cmdCancel(errListenerDone)
			wg.Wait()
		}

		addr := proxy.Addr{
			Namespace:     d.f.Namespace.Name,
			PodName:       pod.Name,
			ContainerName: pod.Spec.Containers[0].Name,
			Port:          int(port),
		}
		listener, err := proxy.Listen(ctx, d.f.ClientSet, d.f.ClientConfig(), addr)
		if err != nil {
			return nil, fmt.Errorf("listen for connections from %+v: %w", addr, err)
		}
		return &listenerWithClose{Listener: listener, close: stopHostpathplugin}, nil
	}
}

// listenerWithClose wraps Close so that it also shuts down hostpathplugin.
type listenerWithClose struct {
	net.Listener
	close func()
}

func (l *listenerWithClose) Close() error {
	// First close connections, then shut down the remote command.
	// Otherwise the connection code is unhappy and logs errors.
	err := l.Listener.Close()
	l.close()
	return err
}

func newNullListener() net.Listener {
	ctx, cancel := context.WithCancelCause(context.Background())
	return &nullListener{ctx: ctx, cancel: cancel}
}

// nullListener blocks all Accept calls until the listener is closed.
type nullListener struct {
	ctx    context.Context
	cancel func(err error)
}

func (l *nullListener) Accept() (net.Conn, error) {
	<-l.ctx.Done()
	return nil, context.Cause(l.ctx)
}

func (l *nullListener) Close() error {
	l.cancel(errors.New("listener was closed"))
	return nil
}

func (l *nullListener) Addr() net.Addr {
	return &net.UnixAddr{}
}

// execute runs a remote command with stdout/stderr redirected to log messages at the chosen verbosity level.
func execute(ctx context.Context, url *url.URL, config *rest.Config, verbosity int) error {
	// Stream output as long as we run, i.e. ignore cancellation.
	stdout := pipe(context.WithoutCancel(ctx), "STDOUT", verbosity)
	stderr := pipe(context.WithoutCancel(ctx), "STDERR", verbosity)
	defer func() { _ = stdout.Close() }()
	defer func() { _ = stderr.Close() }()

	executor := exec.DefaultRemoteExecutor{}
	return executor.ExecuteWithContext(ctx, url, config, nil, stdout, stderr, false, nil)
}

// pipe creates an in-memory pipe and starts logging whatever is sent through that pipe in the background.
func pipe(ctx context.Context, msg string, verbosity int) *io.PipeWriter {
	logger := klog.FromContext(ctx)

	reader, writer := io.Pipe()
	go func() {
		buffer := make([]byte, 10*1024)
		for {
			n, err := reader.Read(buffer)
			if n > 0 {
				logger.V(verbosity).Info(msg, "msg", string(buffer[0:n]))
			}
			if err != nil {
				if !errors.Is(err, io.EOF) {
					logger.Error(err, msg)
				}
				reader.CloseWithError(err)
				return
			}
			if ctx.Err() != nil {
				reader.CloseWithError(context.Cause(ctx))
				return
			}
		}
	}()
	return writer
}

func (d *Driver) TearDown(ctx context.Context) {
	for _, c := range d.cleanup {
		c(ctx)
	}
	d.cleanup = nil
	d.wg.Wait()
}

// IsGone checks that the kubelet is done with the driver.
// This is done by waiting for the kubelet to remove the
// driver's ResourceSlices, which takes at least 30 seconds
// because of the delay in the kubelet.
//
// Only use this in tests where kubelet support for DRA is guaranteed.
func (d *Driver) IsGone(ctx context.Context) {
	ginkgo.By(fmt.Sprintf("Waiting for ResourceSlices of driver %s to be removed...", d.Name))
	gomega.Eventually(ctx, d.NewGetSlices()).WithTimeout(2 * time.Minute).Should(gomega.HaveField("Items", gomega.BeEmpty()))
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
