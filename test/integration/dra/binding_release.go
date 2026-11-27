/*
Copyright The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
)

const bindingReleasePluginName = "BindingReleaseTest"

// Intercepting Unreserve keeps the device occupied until the test has observed
// the waiter's rejection. Other events cannot rescue a missing deletion hint.
type bindingReleasePlugin struct {
	*dynamicresources.DynamicResources
	resolver              fwk.DeviceClassResolver
	claims                fwk.ResourceClaimTracker
	isolate               atomic.Bool
	holder, waiter, plain string
	atUnreserve           chan struct{}
	resume                chan struct{}
	entered               sync.Once
	bindingTimedOut       atomic.Bool
	released              atomic.Bool
	attempts              atomic.Int32
	lateHints             atomic.Int32
	plainSkips            atomic.Int32
}

func (p *bindingReleasePlugin) Name() string { return bindingReleasePluginName }

func (p *bindingReleasePlugin) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if pod.Name == p.waiter {
		p.attempts.Add(1)
	}
	return p.DynamicResources.PreFilter(ctx, state, pod, nodes)
}

func (p *bindingReleasePlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, node string) *fwk.Status {
	status := p.DynamicResources.PreBind(ctx, state, pod, node)
	if pod.Name == p.holder {
		p.bindingTimedOut.Store(errors.Is(status.AsError(), dynamicresources.ErrDeviceBindingTimeout))
	}
	return status
}

func (p *bindingReleasePlugin) Unreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, node string) {
	if pod.Name == p.holder {
		p.entered.Do(func() { close(p.atUnreserve) })
		select {
		case <-p.resume:
		case <-ctx.Done():
		}
	}
	p.DynamicResources.Unreserve(ctx, state, pod, node)
	if pod.Name == p.holder {
		p.released.Store(true)
	}
}

func (p *bindingReleasePlugin) EventsToRegister(ctx context.Context) ([]fwk.ClusterEventWithHint, error) {
	events, err := p.DynamicResources.EventsToRegister(ctx)
	if err != nil {
		return nil, err
	}
	for i := range events {
		original := events[i].QueueingHintFn
		event := events[i].Event
		events[i].QueueingHintFn = func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
			if pod.Name == p.waiter {
				if event.Resource != fwk.AssignedPod || event.ActionType != fwk.Delete {
					if p.isolate.Load() {
						return fwk.QueueSkip, nil
					}
				} else {
					deleted, ok := oldObj.(*v1.Pod)
					if !ok {
						return fwk.QueueSkip, fmt.Errorf("expected deleted Pod, got %T", oldObj)
					}
					if deleted.Name == p.holder && !p.released.Load() {
						return fwk.QueueSkip, nil
					}
				}
			}
			hint := fwk.Queue
			var err error
			if original != nil {
				hint, err = original(logger, pod, oldObj, newObj)
			}
			if pod.Name == p.waiter && event.Resource == fwk.AssignedPod {
				if deleted, ok := oldObj.(*v1.Pod); ok {
					if deleted.Name == p.plain && hint == fwk.QueueSkip && err == nil {
						p.plainSkips.Add(1)
					}
					if deleted.Name == p.holder && p.released.Load() && hint == fwk.Queue && err == nil {
						p.lateHints.Add(1)
					}
				}
			}
			return hint, err
		}
	}
	return events, nil
}

func testBindingRelease(tCtx ktesting.TContext, kind string, createAfterDeletion bool) {
	// The scheduler below runs with the ambient feature gates. In
	// configurations that enable GenericWorkload its informers need the
	// PodGroup API, which a bare test API server does not serve.
	flags := append(framework.DefaultTestServerFlags(), "--runtime-config=scheduling.k8s.io/v1beta1=true")
	server := kubeapiservertesting.StartTestServerOrDie(tCtx,
		kubeapiservertesting.NewDefaultTestServerOptions(), flags, framework.SharedEtcd())
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) { server.TearDownFn() })
	tCtx = tCtx.WithRESTConfig(server.ClientConfig)
	createNodes(tCtx)
	tCtx = prepareClaimController(tCtx)
	namespace := createTestNamespace(tCtx, nil)
	resourceName := "example.com/" + namespace
	class, driver := createTestClassWithSpec(tCtx, namespace, &resourceapi.DeviceClassSpec{ExtendedResourceName: &resourceName})
	slice := st.MakeResourceSlice("worker-0", driver).Obj()
	slice.Spec.Devices = []resourceapi.Device{{Name: "dev-0", BindingConditions: []string{bindingCondition}, BindingFailureConditions: []string{failureCondition}}}
	createSlice(tCtx, slice)

	probe := &bindingReleasePlugin{holder: "holder", waiter: "waiter", plain: "plain",
		atUnreserve: make(chan struct{}), resume: make(chan struct{})}
	var resumeOnce sync.Once
	resume := func() { resumeOnce.Do(func() { close(probe.resume) }) }
	defer resume()
	cfg := newSchedulerComponentConfig(tCtx, "")
	// The config is already defaulted, so replace the enabled plugin rather
	// than adding a disabled entry that would only affect default merging.
	replaced := false
	for i := range cfg.Profiles[0].Plugins.MultiPoint.Enabled {
		if cfg.Profiles[0].Plugins.MultiPoint.Enabled[i].Name == dynamicresources.Name {
			cfg.Profiles[0].Plugins.MultiPoint.Enabled[i].Name = bindingReleasePluginName
			replaced = true
		}
	}
	if !replaced {
		tCtx.Fatal("default scheduler profile does not enable DynamicResources")
	}
	registry := frameworkruntime.Registry{bindingReleasePluginName: func(ctx context.Context, _ runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
		// A failed fast-exit assertion must finish before the fallback binding timeout.
		bindingTimeout := time.Minute
		if kind == "standalone-timeout" {
			bindingTimeout = 8 * time.Second
		}
		plugin, err := dynamicresources.New(ctx, &config.DynamicResourcesArgs{BindingTimeout: &metav1.Duration{Duration: bindingTimeout}},
			handle, feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate))
		if err != nil {
			return nil, err
		}
		probe.DynamicResources = plugin.(*dynamicresources.DynamicResources)
		probe.resolver = handle.SharedDRAManager().DeviceClassResolver()
		probe.claims = handle.SharedDRAManager().ResourceClaims()
		return probe, nil
	}}
	startController := sync.OnceFunc(func() { startClaimController(tCtx) })
	if kind == "standalone-timeout" {
		// Keep the allocation intact until the timer fires; deallocation must not be the cause of exit.
		defer startController()
	} else {
		startController()
	}
	schedulerCtx := tCtx.WithCancel()
	sched, _, schedulerDone := util.StartSchedulerWithDone(schedulerCtx, cfg, registry)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		schedulerCtx.Cancel("test is done")
		select {
		case <-schedulerDone:
		case <-tCtx.Done():
			tCtx.Error("scheduler did not stop before cleanup context expired")
		}
	})

	plain := st.MakePod().Name(probe.plain).Container("test-container").Obj()
	plain.Namespace = namespace
	_, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, plain, metav1.CreateOptions{})
	tCtx.ExpectNoError(err)
	waitForPodScheduled(tCtx, namespace, plain.Name)

	explicit := createClaim(tCtx, namespace, "-holder", class, claim)
	var template *resourceapi.ResourceClaimTemplate
	if kind == "template" {
		template = &resourceapi.ResourceClaimTemplate{ObjectMeta: metav1.ObjectMeta{Name: "template", Namespace: namespace},
			Spec: resourceapi.ResourceClaimTemplateSpec{Spec: explicit.Spec}}
		template, err = tCtx.Client().ResourceV1().ResourceClaimTemplates(namespace).Create(tCtx, template, metav1.CreateOptions{})
		tCtx.ExpectNoError(err)
	}
	makePod := func(name string, useExtended bool, c *resourceapi.ResourceClaim) *v1.Pod {
		p := st.MakePod().Name(name).Namespace(namespace).Container("test-container").Obj()
		if useExtended {
			return createPodWithExtendedResource(tCtx, namespace, resourceName, "1", p)
		}
		if template != nil {
			p.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "device", ResourceClaimTemplateName: &template.Name}}
			created, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, p, metav1.CreateOptions{})
			tCtx.ExpectNoError(err)
			return created
		}
		return createPod(tCtx, namespace, "", p, c)
	}
	extended := kind == "extended" || kind == "extended-renamed" || kind == "extended-removed"
	holder := makePod(probe.holder, extended, explicit)
	allocatedFor := func(pod *v1.Pod) *resourceapi.ResourceClaim {
		claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
		tCtx.ExpectNoError(err)
		for i := range claims.Items {
			c := &claims.Items[i]
			if c.Status.Allocation == nil {
				continue
			}
			for _, consumer := range c.Status.ReservedFor {
				if consumer.UID == pod.UID {
					return c
				}
			}
		}
		return nil
	}
	tCtx.Eventually(func() *resourceapi.ResourceClaim { return allocatedFor(holder) }).WithTimeout(schedulingTimeout).ShouldNot(gomega.BeNil())
	holderClaim := allocatedFor(holder)
	tCtx.Eventually(func(tCtx ktesting.TContext) ([]v1.Event, error) {
		events, err := tCtx.Client().CoreV1().Events(namespace).List(tCtx, metav1.ListOptions{FieldSelector: "involvedObject.name=" + holder.Name})
		if err != nil {
			return nil, err
		}
		return events.Items, nil
	}).WithTimeout(schedulingTimeout).Should(gomega.ContainElement(gomega.HaveField("Reason", "BindingConditionsPending")))

	waiterClaim := explicit
	if kind != "shared" {
		waiterClaim = createClaim(tCtx, namespace, "-waiter", class, claim)
	}
	var waiter *v1.Pod
	createWaiter := func() {
		waiter = makePod(probe.waiter, kind == "extended", waiterClaim)
		if kind == "shared" {
			if len(waiter.Spec.ResourceClaims) != 1 || *waiter.Spec.ResourceClaims[0].ResourceClaimName != explicit.Name {
				tCtx.Fatal("holder and waiter must reference the same standalone claim")
			}
		}
		tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
			return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, waiter.Name, metav1.GetOptions{})
		}).WithTimeout(schedulingTimeout).Should(gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{"Type": gomega.Equal(v1.PodScheduled), "Status": gomega.Equal(v1.ConditionFalse), "Message": gomega.MatchRegexp("cannot allocate|in the process of being allocated")}))))
		probe.isolate.Store(true)
		tCtx.Eventually(sched.SchedulingQueue.UnschedulablePods).WithTimeout(schedulingTimeout).Should(
			gomega.ContainElement(gomega.HaveField("UID", waiter.UID)), "waiter must finish its failed cycle before the release barrier opens")
	}
	if !createAfterDeletion {
		createWaiter()
	}
	if waiter != nil {
		attempts := probe.attempts.Load()
		deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, plain.Name)
		tCtx.Eventually(probe.plainSkips.Load).WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">", 0), "plain pod deletion must reach the DRA hint and return QueueSkip")
		if probe.attempts.Load() != attempts {
			tCtx.Fatal("plain pod deletion triggered another waiter scheduling attempt")
		}
	}
	if kind == "extended-renamed" || kind == "extended-removed" {
		class = class.DeepCopy()
		class.Spec.ExtendedResourceName = nil
		if kind == "extended-renamed" {
			// Short name: the resource name part is limited to 63 bytes and the
			// namespace-derived default name is close to that limit already.
			renamed := "example.com/renamed"
			class.Spec.ExtendedResourceName = &renamed
		}
		_, err := tCtx.Client().ResourceV1().DeviceClasses().Update(tCtx, class, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err)
		tCtx.Eventually(func() *resourceapi.DeviceClass { return probe.resolver.GetDeviceClass(v1.ResourceName(resourceName)) }).WithTimeout(schedulingTimeout).Should(gomega.BeNil(), "scheduler must observe the old mapping disappearing before the holder is deleted")
	}
	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, holder.Name)
	if createAfterDeletion {
		// Sharing is allowed while an allocation still exists. Force the problematic
		// interleaving: the allocation is gone, but its in-flight marker remains.
		waitForStandaloneDeallocation(tCtx, namespace, explicit.Name)
		tCtx.Eventually(func() (*resourceapi.ResourceClaim, error) {
			return probe.claims.Get(namespace, explicit.Name)
		}).WithTimeout(20 * time.Second).Should(gomega.HaveField("Status.Allocation", gomega.BeNil()))
		createWaiter()
	}
	if kind != "standalone-deallocated" && kind != "standalone-timeout" && kind != "shared" {
		// Explicitly delete standalone claims. For pod-owned claims, this simulates
		// garbage collection, which is not running in these tests.
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete,
			tCtx.Client().ResourceV1().ResourceClaims(namespace).Get, holderClaim.Name)
	} else if kind != "standalone-timeout" {
		waitForStandaloneDeallocation(tCtx, namespace, explicit.Name)
	}
	if kind == "standalone-timeout" {
		tCtx.Eventually(probe.atUnreserve).WithTimeout(20*time.Second).Should(gomega.BeClosed(), "binding timeout must eventually enter Unreserve")
		if !probe.bindingTimedOut.Load() {
			tCtx.Fatal("holder must leave PreBind because of bindingTimeout")
		}
		startController()
		waitForStandaloneDeallocation(tCtx, namespace, explicit.Name)
	} else {
		tCtx.Eventually(probe.atUnreserve).WithTimeout(20*time.Second).Should(gomega.BeClosed(), "PreBind must exit before the one-minute fallback timeout")
	}
	attempts := probe.attempts.Load()
	if allocatedFor(waiter) != nil {
		tCtx.Fatal("waiter obtained the device before the holder's Unreserve")
	}
	if probe.released.Load() {
		tCtx.Fatal("holder's Unreserve passed the test barrier prematurely")
	}
	resume()
	tCtx.Eventually(func() *resourceapi.ResourceClaim { return allocatedFor(waiter) }).WithTimeout(20*time.Second).ShouldNot(gomega.BeNil(), "waiter must recover without the periodic queue flush")
	tCtx.Eventually(probe.lateHints.Load).WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">", 0), "the post-Unreserve deletion must return Queue for this waiter")
	if probe.attempts.Load() <= attempts {
		tCtx.Fatal("waiter was not scheduled again after the holder's Unreserve")
	}
	result := allocatedFor(waiter)
	if len(result.Status.Allocation.Devices.Results) != 1 || result.Status.Allocation.Devices.Results[0].Device != "dev-0" {
		tCtx.Fatalf("waiter must receive the released device dev-0, got %#v", result.Status.Allocation.Devices.Results)
	}
}

func waitForStandaloneDeallocation(tCtx ktesting.TContext, namespace, name string) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
		return tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, name, metav1.GetOptions{})
	}).WithTimeout(20*time.Second).Should(gomega.HaveField("Status.Allocation", gomega.BeNil()), "the standalone claim must remain present but become deallocated")
}
