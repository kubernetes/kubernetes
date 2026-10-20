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
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// These tests flip process-global feature gates and each owns its
// apiserver and scheduler, so none of them may run in parallel.

const (
	saGateName     = "SharingAffinityGate"
	saGateLabel    = "sharing-affinity-gate"
	saGateWait     = "wait"
	saGateReject   = "reject"
	saGatePreBind  = "prebind-fail-once"
	saNode         = "worker-0"
	saMemory       = resourceapi.QualifiedName("memory")
	saSubnetExpr   = `parameters.?subnetID.orValue("")`
	saConsistently = 5 * time.Second
)

var saSchedulerConfig = `
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
podInitialBackoffSeconds: 1
podMaxBackoffSeconds: 1
profiles:
- schedulerName: default-scheduler
  plugins:
    permit:
      enabled:
      - name: ` + saGateName + `
    preBind:
      enabled:
      - name: ` + saGateName + `
`

// sharingAffinityGate gives tests control over the window after Reserve:
// Permit runs before any PreBind, so Wait holds a tentative lock and Reject
// drives Unreserve; the PreBind hook runs after DynamicResources persisted.
type sharingAffinityGate struct {
	handle fwk.Handle

	mutex      sync.Mutex
	waiting    map[string]bool
	rejected   map[string]int
	preBindHit map[string]int
}

var (
	_ fwk.PermitPlugin  = &sharingAffinityGate{}
	_ fwk.PreBindPlugin = &sharingAffinityGate{}
)

func (g *sharingAffinityGate) Name() string { return saGateName }

func (g *sharingAffinityGate) Permit(_ context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) (*fwk.Status, time.Duration) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	switch pod.Labels[saGateLabel] {
	case saGateWait:
		g.waiting[pod.Name] = true
		return fwk.NewStatus(fwk.Wait), 2 * time.Minute
	case saGateReject:
		g.rejected[pod.Name]++
		return fwk.NewStatus(fwk.Unschedulable, "rejected by sharing affinity test gate"), 0
	}
	return nil, 0
}

func (g *sharingAffinityGate) PreBindPreFlight(_ context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	if pod.Labels[saGateLabel] != saGatePreBind {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	return nil, nil
}

func (g *sharingAffinityGate) PreBind(_ context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) *fwk.Status {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	g.preBindHit[pod.Name]++
	if g.preBindHit[pod.Name] == 1 {
		return fwk.NewStatus(fwk.Error, "injected PreBind failure after DynamicResources")
	}
	return nil
}

func (g *sharingAffinityGate) count(m map[string]int, name string) int {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return m[name]
}

func (g *sharingAffinityGate) isWaiting(name string) bool {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g.waiting[name]
}

type saEnv struct {
	tCtx ktesting.TContext
	gate *sharingAffinityGate
}

// newSAEnv starts an apiserver with sharing affinity stored, plus nodes and
// the claim controller (needed to release claim finalizers on deletion).
func newSAEnv(t *testing.T) *saEnv {
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRASharingAffinity:    true,
	})
	server, err := kubeapiservertesting.StartTestServer(tCtx.WithoutCancel(), kubeapiservertesting.NewDefaultTestServerOptions(), framework.DefaultTestServerFlags(), framework.SharedEtcd())
	tCtx.ExpectNoError(err, "start apiserver")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) { server.TearDownFn() })
	tCtx = tCtx.WithRESTConfig(server.ClientConfig)
	createNodes(tCtx)
	tCtx = prepareClaimController(tCtx)
	startClaimController(tCtx)
	return &saEnv{tCtx: tCtx}
}

// startScheduler reads the current scheduler-side gates, so callers flip
// DRASharingAffinity before calling it.
func (e *saEnv) startScheduler() func() {
	gate := &sharingAffinityGate{waiting: map[string]bool{}, rejected: map[string]int{}, preBindHit: map[string]int{}}
	e.gate = gate
	registry := frameworkruntime.Registry{
		saGateName: func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			gate.handle = h
			return gate, nil
		},
	}
	schedCtx := e.tCtx.WithCancel()
	cfg := newSchedulerComponentConfig(schedCtx, saSchedulerConfig)
	_, informerFactory, done := util.StartSchedulerWithDone(schedCtx, cfg, registry)
	var once sync.Once
	stop := func() {
		once.Do(func() {
			schedCtx.Cancel("stop sharing affinity scheduler")
			select {
			case <-done:
			case <-time.After(30 * time.Second):
				e.tCtx.Fatal("sharing affinity scheduler did not stop within 30s")
			}
			informerFactory.Shutdown()
		})
	}
	e.tCtx.CleanupCtx(func(ktesting.TContext) { stop() })
	return stop
}

// scope is a per-subtest namespace with its own driver and class.
type saScope struct {
	tCtx   ktesting.TContext
	ns     string
	driver string
	class  *resourceapi.DeviceClass
}

func (e *saEnv) run(name string, cb func(s *saScope)) {
	e.tCtx.Run(name, func(tCtx ktesting.TContext) {
		ns := createTestNamespace(tCtx, nil)
		class, driver := createTestClass(tCtx, ns)
		cb(&saScope{tCtx: tCtx, ns: ns, driver: driver, class: class})
	})
}

type saDevice struct {
	name       string
	memory     string
	multiple   bool
	attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
}

func shared(name, memory string) saDevice {
	return saDevice{name: name, memory: memory, multiple: true}
}

// saSlices is every slice a helper call created for one pool.
type saSlices []*resourceapi.ResourceSlice

// slice publishes a pool. Validation keeps sharingAffinity in a dedicated
// metadata slice, so extractors plus devices become two slices and count is
// raised by one; count is otherwise the caller's ResourceSliceCount.
func (s *saScope) slice(name, pool string, generation, count int64, extractors []resourceapi.SharingAffinityExtractor, devices ...saDevice) saSlices {
	if len(extractors) > 0 && len(devices) > 0 {
		count++
		return append(s.slice(name+"-dev", pool, generation, count, nil, devices...),
			s.slice(name+"-meta", pool, generation, count, extractors)...)
	}
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: s.ns + "-" + name},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:          s.driver,
			Pool:            resourceapi.ResourcePool{Name: pool, Generation: generation, ResourceSliceCount: count},
			NodeName:        ptr.To(saNode),
			SharingAffinity: extractors,
		},
	}
	for _, d := range devices {
		device := resourceapi.Device{Name: d.name, Attributes: d.attributes}
		if d.multiple {
			device.AllowMultipleAllocations = ptr.To(true)
		}
		if d.memory != "" {
			device.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{saMemory: {Value: resource.MustParse(d.memory)}}
		}
		slice.Spec.Devices = append(slice.Spec.Devices, device)
	}
	return saSlices{createSlice(s.tCtx, slice)}
}

func subnetExtractor(expr string) []resourceapi.SharingAffinityExtractor {
	return []resourceapi.SharingAffinityExtractor{{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: expr}}
}

func nicParams(fields string) string {
	if fields != "" {
		fields = "," + fields
	}
	return `{"apiVersion":"networking.example.com/v1","kind":"NICConfig"` + fields + `}`
}

func subnet(id string) string { return nicParams(fmt.Sprintf(`"subnetID":%q`, id)) }

type saRequest struct {
	name        string
	memory      string
	subRequests []string
	selectors   []resourceapi.DeviceSelector
}

type saConfig struct {
	requests []string
	params   string
}

func (s *saScope) claim(name string, requests []saRequest, configs ...saConfig) *resourceapi.ResourceClaim {
	claim := &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: s.ns}}
	capacity := func(memory string) *resourceapi.CapacityRequirements {
		if memory == "" {
			return nil
		}
		return &resourceapi.CapacityRequirements{Requests: map[resourceapi.QualifiedName]resource.Quantity{saMemory: resource.MustParse(memory)}}
	}
	for _, r := range requests {
		request := resourceapi.DeviceRequest{Name: r.name}
		if len(r.subRequests) == 0 {
			request.Exactly = &resourceapi.ExactDeviceRequest{
				DeviceClassName: s.class.Name, AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
				Capacity:  capacity(r.memory),
				Selectors: r.selectors,
			}
		}
		for _, sub := range r.subRequests {
			request.FirstAvailable = append(request.FirstAvailable, resourceapi.DeviceSubRequest{
				Name: sub, DeviceClassName: s.class.Name, AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
				Capacity:  capacity(r.memory),
				Selectors: r.selectors,
			})
		}
		claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, request)
	}
	for _, c := range configs {
		claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, resourceapi.DeviceClaimConfiguration{
			Requests: c.requests,
			DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: s.driver, Parameters: runtime.RawExtension{Raw: []byte(c.params)},
			}},
		})
	}
	claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Create(s.tCtx, claim, metav1.CreateOptions{})
	s.tCtx.ExpectNoError(err, "create claim "+name)
	return claim
}

// nicClaim is a single shared-NIC request carrying one config.
func (s *saScope) nicClaim(name, memory, params string) *resourceapi.ResourceClaim {
	var configs []saConfig
	if params != "" {
		configs = append(configs, saConfig{requests: []string{"nic"}, params: params})
	}
	return s.claim(name, []saRequest{{name: "nic", memory: memory}}, configs...)
}

func (s *saScope) pod(name, gate string, claims ...*resourceapi.ResourceClaim) *v1.Pod {
	w := st.MakePod().Name(name).Container("c")
	if gate != "" {
		w = w.Label(saGateLabel, gate)
	}
	return createPod(s.tCtx, s.ns, "", w.Obj(), claims...)
}

func (s *saScope) scheduled(pod *v1.Pod) *v1.Pod {
	s.tCtx.Helper()
	pod = waitForPodScheduled(s.tCtx, s.ns, pod.Name)
	if pod.Spec.NodeName != saNode {
		s.tCtx.Fatalf("pod %s bound to %q, want %q", pod.Name, pod.Spec.NodeName, saNode)
	}
	return pod
}

// notScheduled waits for a PodScheduled=False condition with the given
// reason and message fragments; absent fragments are asserted too.
func (s *saScope) notScheduled(pod *v1.Pod, reason string, contains []string, absent ...string) *v1.Pod {
	s.tCtx.Helper()
	var got *v1.Pod
	s.tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		p, err := tCtx.Client().CoreV1().Pods(s.ns).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		got = p
		for _, c := range p.Status.Conditions {
			if c.Type == v1.PodScheduled && c.Status == v1.ConditionFalse && c.Reason == reason {
				return c.Message, nil
			}
		}
		return "", nil
	}).WithTimeout(schedulingTimeout).Should(gomega.And(append([]gomega.OmegaMatcher{gomega.Not(gomega.BeEmpty())},
		saContainAll(contains)...)...), "pod %s should be %s", pod.Name, reason)
	for _, a := range absent {
		for _, c := range got.Status.Conditions {
			if strings.Contains(c.Message, a) {
				s.tCtx.Fatalf("pod %s condition leaks %q: %s", pod.Name, a, c.Message)
			}
		}
	}
	if got.Spec.NodeName != "" {
		s.tCtx.Fatalf("pod %s unexpectedly bound to %s", pod.Name, got.Spec.NodeName)
	}
	return got
}

func saContainAll(fragments []string) []gomega.OmegaMatcher {
	var m []gomega.OmegaMatcher
	for _, f := range fragments {
		m = append(m, gomega.ContainSubstring(f))
	}
	return m
}

// staysPending asserts a bounded window in which the pod is not bound and
// its claim stays unallocated.
func (s *saScope) staysPending(pod *v1.Pod, claims ...*resourceapi.ResourceClaim) {
	s.tCtx.Helper()
	s.tCtx.Consistently(func(tCtx ktesting.TContext) (string, error) {
		p, err := tCtx.Client().CoreV1().Pods(s.ns).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		for _, c := range claims {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(tCtx, c.Name, metav1.GetOptions{})
			if err != nil {
				return "", err
			}
			if c.Status.Allocation != nil {
				return "claim " + c.Name + " allocated", nil
			}
		}
		return p.Spec.NodeName, nil
	}).WithTimeout(saConsistently).WithPolling(250*time.Millisecond).Should(gomega.BeEmpty(), "pod %s must stay pending", pod.Name)
}

func (s *saScope) allocated(claim *resourceapi.ResourceClaim) []resourceapi.DeviceRequestAllocationResult {
	s.tCtx.Helper()
	c := waitForClaimAllocatedToDevice(s.tCtx, s.ns, claim.Name, schedulingTimeout)
	return c.Status.Allocation.Devices.Results
}

func (s *saScope) unallocated(claim *resourceapi.ResourceClaim) {
	s.tCtx.Helper()
	c, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(s.tCtx, claim.Name, metav1.GetOptions{})
	s.tCtx.ExpectNoError(err, "get claim")
	if c.Status.Allocation != nil {
		s.tCtx.Fatalf("claim %s unexpectedly allocated: %+v", claim.Name, c.Status.Allocation.Devices.Results)
	}
}

func (s *saScope) device(claim *resourceapi.ResourceClaim) string {
	s.tCtx.Helper()
	results := s.allocated(claim)
	if len(results) != 1 {
		s.tCtx.Fatalf("claim %s: want one result, got %+v", claim.Name, results)
	}
	return results[0].Pool + "/" + results[0].Device
}

func (s *saScope) release(pod *v1.Pod, claims ...*resourceapi.ResourceClaim) {
	s.tCtx.Helper()
	deleteAndWait(s.tCtx, s.tCtx.Client().CoreV1().Pods(s.ns).Delete, s.tCtx.Client().CoreV1().Pods(s.ns).Get, pod.Name)
	for _, c := range claims {
		deleteAndWait(s.tCtx, s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Delete, s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get, c.Name)
	}
}

func (s *saScope) updateSlice(slices saSlices, mutate func(*resourceapi.ResourceSlice)) {
	s.tCtx.Helper()
	for _, slice := range slices {
		current, err := s.tCtx.Client().ResourceV1().ResourceSlices().Get(s.tCtx, slice.Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "get slice")
		mutate(current)
		_, err = s.tCtx.Client().ResourceV1().ResourceSlices().Update(s.tCtx, current, metav1.UpdateOptions{})
		s.tCtx.ExpectNoError(err, "update slice")
	}
}

func (s *saScope) deleteSlice(slices saSlices) {
	s.tCtx.Helper()
	for _, slice := range slices {
		deleteAndWait(s.tCtx, s.tCtx.Client().ResourceV1().ResourceSlices().Delete, s.tCtx.Client().ResourceV1().ResourceSlices().Get, slice.Name)
	}
}

func TestSharingAffinitySchedulingCompatible(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("same-device", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		s.scheduled(s.pod("pb", "", b))
		ra, rb := s.allocated(a), s.allocated(b)
		if ra[0].Device != "nic" || rb[0].Device != "nic" {
			s.tCtx.Fatalf("compatible claims not co-allocated: %+v / %+v", ra, rb)
		}
		if ra[0].ShareID == nil || rb[0].ShareID == nil || *ra[0].ShareID == *rb[0].ShareID {
			s.tCtx.Fatalf("co-allocated claims need distinct ShareIDs: %v / %v", ra[0].ShareID, rb[0].ShareID)
		}
	})
}

func TestSharingAffinitySchedulingMismatch(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("one-device", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
	})
}

func TestSharingAffinitySchedulingReleaseAndRollback(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("release", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.release(pa, a)
		s.scheduled(pb)
		if d := s.device(b); d != "pool/nic" {
			s.tCtx.Fatalf("released device not reused: %s", d)
		}
	})
	env.run("permit-reject-unreserve", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", saGateReject, a)
		s.tCtx.Eventually(func() int { return env.gate.count(env.gate.rejected, pa.Name) }).
			WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">=", 1), "Permit should reject pa")
		s.unallocated(a)
		// Deleting the pod cannot run Unreserve again; a leaked tentative
		// lock from the rejected attempt would keep blocking subnet-b.
		deleteAndWait(s.tCtx, s.tCtx.Client().CoreV1().Pods(s.ns).Delete, s.tCtx.Client().CoreV1().Pods(s.ns).Get, pa.Name)
		s.unallocated(a)
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.scheduled(s.pod("pb", "", b))
		if d := s.device(b); d != "pool/nic" {
			s.tCtx.Fatalf("want pool/nic after rollback, got %s", d)
		}
	})
	env.run("prebind-failure-after-persist", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", saGatePreBind, a)
		pa = s.scheduled(pa)
		if hits := env.gate.count(env.gate.preBindHit, pa.Name); hits < 2 {
			s.tCtx.Fatalf("PreBind failure path not exercised: %d calls", hits)
		}
		s.allocated(a)
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.release(pa, a)
		s.scheduled(pb)
	})
}

func TestSharingAffinitySchedulingCapacity(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("exhausted-lock", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "4Gi"), shared("nic-1", "4Gi"))
		a := s.nicClaim("a", "3Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "3Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pb", "", b))
		if da, db := s.device(a), s.device(b); da == db {
			s.tCtx.Fatalf("capacity exceeded: both 3Gi claims on %s", da)
		}
		// 1Gi remains on each device; observed-only accounting must admit it.
		c := s.nicClaim("c", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pc", "", c))
		d := s.nicClaim("d", "2Gi", subnet("subnet-a"))
		s.notScheduled(s.pod("pd", "", d), v1.PodReasonUnschedulable, nil)
		s.unallocated(d)
	})
}

func TestSharingAffinitySchedulingConcurrent(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("overlapping-attempts", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", saGateWait, a)
		var waiting fwk.WaitingPod
		s.tCtx.Eventually(func() bool {
			if !env.gate.isWaiting(pa.Name) {
				return false
			}
			waiting = env.gate.handle.GetWaitingPod(pa.UID)
			return waiting != nil
		}).WithTimeout(schedulingTimeout).Should(gomega.BeTrue(), "pa should hold its Reserve at Permit")
		// The lock is only in-flight: nothing has been written to the API.
		s.unallocated(a)
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
		waiting.Allow(saGateName)
		s.scheduled(pa)
		if d := s.device(a); d != "pool/nic" {
			s.tCtx.Fatalf("pa allocated %s", d)
		}
		s.staysPending(pb, b)
	})
	env.run("same-pod-one-device", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.notScheduled(s.pod("p", "", a, b), v1.PodReasonUnschedulable, nil)
		s.unallocated(a)
		s.unallocated(b)
	})
	env.run("same-pod-two-devices", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "8Gi"), shared("nic-1", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.scheduled(s.pod("p", "", a, b))
		if da, db := s.device(a), s.device(b); da == db {
			s.tCtx.Fatalf("one allocation co-located incompatible claims on %s", da)
		}
	})
}

func TestSharingAffinitySchedulingGateDisabled(t *testing.T) {
	env := newSAEnv(t)
	type pending struct {
		s     *saScope
		check func()
	}
	// Slices with affinity must be stored while the apiserver gate is on;
	// only the scheduler then runs with the gate off.
	var scopes []saScope
	var declaring, gen2 saSlices
	for _, name := range []string{"declaring", "unconstrained", "older-generation"} {
		tCtx := env.tCtx
		ns := createTestNamespace(tCtx.WithStep(name), nil)
		class, driver := createTestClass(tCtx, ns)
		scopes = append(scopes, saScope{tCtx: tCtx, ns: ns, driver: driver, class: class})
	}
	declaring = scopes[0].slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
	scopes[1].slice("s", "pool", 1, 1, nil, shared("nic", "8Gi"))
	scopes[2].slice("gen1", "pool", 1, 1, nil, shared("nic", "8Gi"))
	gen2 = scopes[2].slice("gen2", "pool", 2, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRASharingAffinity, false)
	stored, err := env.tCtx.Client().ResourceV1().ResourceSlices().Get(env.tCtx, declaring[1].Name, metav1.GetOptions{})
	env.tCtx.ExpectNoError(err, "get declaring slice")
	if len(stored.Spec.SharingAffinity) != 1 {
		t.Fatalf("apiserver must retain sharingAffinity, got %+v", stored.Spec.SharingAffinity)
	}
	env.startScheduler()

	env.tCtx.Run("declaring-pool-unallocatable", func(tCtx ktesting.TContext) {
		s := scopes[0]
		s.tCtx = tCtx
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.notScheduled(s.pod("pa", "", a), v1.PodReasonUnschedulable, nil)
		s.unallocated(a)
	})
	env.tCtx.Run("unconstrained-pool-legacy", func(tCtx ktesting.TContext) {
		s := scopes[1]
		s.tCtx = tCtx
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.scheduled(s.pod("pa", "", a))
		s.scheduled(s.pod("pb", "", b))
		if da, db := s.device(a), s.device(b); da != db {
			s.tCtx.Fatalf("unconstrained shared device not shared: %s vs %s", da, db)
		}
	})
	env.tCtx.Run("delete-exposes-older-generation", func(tCtx ktesting.TContext) {
		s := scopes[2]
		s.tCtx = tCtx
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.deleteSlice(gen2)
		s.scheduled(pa)
	})
}

func TestSharingAffinitySchedulingRequestScope(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("request", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		s.slice("plain", "plain", 1, 1, nil, saDevice{name: "dedicated"})
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi"}, {name: "b"}},
			saConfig{requests: []string{"a"}, params: subnet("subnet-a")})
		s.scheduled(s.pod("p", "", c))
		got := map[string]string{}
		for _, r := range s.allocated(c) {
			got[r.Request] = r.Pool + "/" + r.Device
		}
		if got["a"] != "constrained/nic" || got["b"] != "plain/dedicated" {
			s.tCtx.Fatalf("config must scope to request a only, got %v", got)
		}
	})
	env.run("request-without-alternative", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "8Gi"), shared("nic-1", "8Gi"))
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi"}, {name: "b", memory: "1Gi"}},
			saConfig{requests: []string{"a"}, params: subnet("subnet-a")})
		s.notScheduled(s.pod("p", "", c), v1.PodReasonUnschedulable, nil)
		s.unallocated(c)
	})
	env.run("subrequest", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi", subRequests: []string{"x", "y"}}},
			saConfig{requests: []string{"a/y"}, params: subnet("subnet-a")})
		s.scheduled(s.pod("p", "", c))
		results := s.allocated(c)
		if len(results) != 1 || results[0].Request != "a/y" || results[0].Device != "nic" {
			s.tCtx.Fatalf("unconfigured subrequest x must not use the constrained pool: %+v", results)
		}
	})
	env.run("other-driver-config-ignored", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi"}})
		c.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{{
			Requests: []string{"a"},
			DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: "other.example.com", Parameters: runtime.RawExtension{Raw: []byte(subnet("subnet-a"))},
			}},
		}}
		s.tCtx.ExpectNoError(s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Delete(s.tCtx, c.Name, metav1.DeleteOptions{}))
		waitForNotFound(s.tCtx, s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get, c.Name)
		c.ResourceVersion, c.UID = "", ""
		c, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Create(s.tCtx, c, metav1.CreateOptions{})
		s.tCtx.ExpectNoError(err, "create claim with foreign-driver config")
		s.notScheduled(s.pod("p", "", c), v1.PodReasonUnschedulable, nil)
		s.unallocated(c)
	})
}

func TestSharingAffinitySchedulingMetadataChanges(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("completing-slice-single-event", func(s *saScope) {
		s.slice("devices", "pool", 1, 2, nil, shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.slice("metadata", "pool", 1, 2, subnetExtractor(saSubnetExpr))
		// No further writes: one Add event must converge.
		s.scheduled(pa)
	})
	env.run("same-generation-expression-update", func(s *saScope) {
		slice := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.scheduled(s.pod("pa", "", a))
		s.updateSlice(slice, func(sl *resourceapi.ResourceSlice) {
			if len(sl.Spec.SharingAffinity) > 0 {
				sl.Spec.SharingAffinity[0].Expression = `parameters.?vlan.orValue("")`
			}
		})
		// The bound owner was extracted under the old predicate; its device
		// must not be treated as compatible (or clean) under the new one.
		b := s.nicClaim("b", "1Gi", nicParams(`"subnetID":"subnet-a","vlan":"subnet-a"`))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
		s.release(pa, a)
		s.scheduled(pb)
	})
	env.run("same-generation-equal-map-expression-update", func(s *saScope) {
		slices := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
		a := saScheduleOwner(s, "a", subnet("subnet-a"), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", "subnet", "subnet-a"))
		extractors := subnetExtractor(`has(parameters.subnet) ? parameters.subnet : parameters.subnetID`)
		saRepublishAndProbe(env, s, slices, 1, extractors, nicParams(`"subnet":"subnet-a"`),
			saLocked("pool", "nic", "1Gi", "subnet", "subnet-a"))
		saScheduleOwner(s, "b", nicParams(`"subnet":"subnet-a"`), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
		saRetained(s, a)
	})
	env.run("generation-roll", func(s *saScope) {
		slice := s.slice("s", "pool", 1, 1, subnetExtractor(`parameters.?zone.orValue("")`), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.updateSlice(slice, func(sl *resourceapi.ResourceSlice) {
			sl.Spec.Pool.Generation = 2
			if len(sl.Spec.SharingAffinity) > 0 {
				sl.Spec.SharingAffinity[0].Expression = saSubnetExpr
			}
		})
		s.scheduled(pa)
	})
	env.run("stale-generation-deleted", func(s *saScope) {
		s.slice("gen1", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		gen2 := s.slice("gen2", "pool", 2, 2, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.deleteSlice(gen2)
		s.scheduled(pa)
	})
	env.run("metadata-added-to-unconstrained-pool", func(s *saScope) {
		slice := s.slice("s", "pool", 1, 1, nil, shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", "")
		pa := s.scheduled(s.pod("pa", "", a))
		s.updateSlice(slice, func(sl *resourceapi.ResourceSlice) {
			sl.Spec.Pool.Generation = 2
			sl.Spec.Pool.ResourceSliceCount = 2
		})
		s.slice("meta", "pool", 2, 2, subnetExtractor(saSubnetExpr))
		b := s.nicClaim("b", "1Gi", subnet("subnet-a"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
		s.release(pa, a)
		s.scheduled(pb)
	})
}

func TestSharingAffinitySchedulingExtractionFailures(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("claim-self-inconsistent", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.claim("a", []saRequest{{name: "nic", memory: "1Gi"}},
			saConfig{requests: []string{"nic"}, params: subnet("subnet-a")},
			saConfig{requests: []string{"nic"}, params: subnet("subnet-b")})
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, []string{s.ns + "/a", "self-inconsistent"}, "subnet-b")
		s.staysPending(pa, a)
		s.release(pa, a)
		fixed := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa2", "", fixed))
	})
	env.run("invalid-pool-duplicate-extractor", func(s *saScope) {
		s.slice("devices", "pool", 1, 3, nil, shared("nic", "8Gi"))
		s.slice("meta-1", "pool", 1, 3, subnetExtractor(saSubnetExpr))
		s.slice("meta-2", "pool", 1, 3, subnetExtractor(`parameters.?other.orValue("")`))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.notScheduled(s.pod("pa", "", a), v1.PodReasonUnschedulable, []string{"invalid"})
		s.unallocated(a)
	})
}

// Same as delete-exposes-older-generation, but the scheduler also runs with
// DRAConsumableCapacity disabled.
func TestSharingAffinitySchedulingGateDisabledNoConsumableCapacity(t *testing.T) {
	env := newSAEnv(t)
	tCtx := env.tCtx
	ns := createTestNamespace(tCtx.WithStep("older-generation"), nil)
	class, driver := createTestClass(tCtx, ns)
	s := saScope{tCtx: tCtx, ns: ns, driver: driver, class: class}
	s.slice("gen1", "pool", 1, 1, nil, saDevice{name: "nic"})
	gen2 := s.slice("gen2", "pool", 2, 1, subnetExtractor(saSubnetExpr), saDevice{name: "nic"})

	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRASharingAffinity:    false,
		features.DRAConsumableCapacity: false,
	})
	env.startScheduler()

	env.tCtx.Run("delete-exposes-older-generation", func(tCtx ktesting.TContext) {
		s.tCtx = tCtx
		a := s.claim("a", []saRequest{{name: "a"}})
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.deleteSlice(gen2)
		s.scheduled(pa)
	})
}
