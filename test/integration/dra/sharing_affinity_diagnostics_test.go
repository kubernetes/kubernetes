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
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	klogtesting "k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
)

const (
	saDiagMismatch    = "incompatible with the device's sharing affinity lock"
	saDiagUnsatisfied = "does not produce a value for every sharing affinity extractor"
	saDiagUnknown     = "device sharing affinity is unknown because of legacy or inconsistent allocations"
	saDiagPending     = "not yet reconciled"
	saDiagRejectLog   = "Sharing affinity rejected candidate devices"
	saDiagExtractLog  = "Sharing affinity extraction failed"
)

// saEventRecord is one pre-coalescing Eventf call.
type saEventRecord struct {
	uid     types.UID
	reason  string
	message string
}

// saRecorderStore is shared by every recorder the spy hands out, including
// WithLogger derivatives.
type saRecorderStore struct {
	mutex    sync.Mutex
	inFlight int
	records  []saEventRecord
}

type saRecorderSpy struct {
	store    *saRecorderStore
	delegate events.EventRecorderLogger
}

var _ events.EventRecorderLogger = &saRecorderSpy{}

func (r *saRecorderSpy) Eventf(regarding runtime.Object, related runtime.Object, eventtype, reason, action, note string, args ...interface{}) {
	r.store.mutex.Lock()
	r.store.inFlight++
	r.store.mutex.Unlock()
	defer func() {
		r.store.mutex.Lock()
		r.store.inFlight--
		r.store.mutex.Unlock()
	}()
	var uid types.UID
	if obj, err := meta.Accessor(regarding); err == nil {
		uid = obj.GetUID()
	}
	r.store.mutex.Lock()
	r.store.records = append(r.store.records, saEventRecord{uid: uid, reason: reason, message: fmt.Sprintf(note, args...)})
	r.store.mutex.Unlock()
	r.delegate.Eventf(regarding, related, eventtype, reason, action, note, args...)
}

func (r *saRecorderSpy) WithLogger(logger klog.Logger) events.EventRecorderLogger {
	return &saRecorderSpy{store: r.store, delegate: r.delegate.WithLogger(logger)}
}

func (s *saRecorderStore) reset() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.records = nil
}

// failed returns the FailedScheduling records for uid and the in-flight count
// from one critical section, so callers get a consistent snapshot.
func (s *saRecorderStore) failed(uid types.UID) ([]string, int) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	var msgs []string
	for _, r := range s.records {
		if r.uid == uid && r.reason == "FailedScheduling" {
			msgs = append(msgs, r.message)
		}
	}
	return msgs, s.inFlight
}

// startSchedulerWithRecorderSpy mirrors util.StartSchedulerWithDone, but
// wraps the recorder factory so Eventf calls are counted before the
// broadcaster coalesces them, and runs with the given logger.
func (e *saEnv) startSchedulerWithRecorderSpy(logger klog.Logger) *saRecorderStore {
	gate := &sharingAffinityGate{waiting: map[string]bool{}, rejected: map[string]int{}, preBindHit: map[string]int{}}
	e.gate = gate
	registry := frameworkruntime.Registry{
		saGateName: func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			gate.handle = h
			return gate, nil
		},
	}
	tCtx := e.tCtx.WithLogger(logger).WithCancel()
	cfg := newSchedulerComponentConfig(tCtx, saSchedulerConfig)
	clientSet := tCtx.Client()
	informerFactory := scheduler.NewInformerFactory(clientSet, 0, nil)
	evtBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: clientSet.EventsV1()})
	go func() {
		<-tCtx.Done()
		evtBroadcaster.Shutdown()
	}()
	evtBroadcaster.StartRecordingToSink(tCtx.Done())
	store := &saRecorderStore{}
	realFactory := profile.NewRecorderFactory(evtBroadcaster)
	sched, err := scheduler.New(
		tCtx,
		clientSet,
		informerFactory,
		nil,
		func(name string) events.EventRecorderLogger {
			return &saRecorderSpy{store: store, delegate: realFactory(name)}
		},
		scheduler.WithKubeConfig(tCtx.RESTConfig()),
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithPercentageOfNodesToScore(cfg.PercentageOfNodesToScore),
		scheduler.WithPercentageOfPlacementsToScore(cfg.PercentageOfPlacementsToScore),
		scheduler.WithPodMaxBackoffSeconds(cfg.PodMaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(cfg.PodInitialBackoffSeconds),
		scheduler.WithExtenders(cfg.Extenders...),
		scheduler.WithParallelism(cfg.Parallelism),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
	)
	tCtx.ExpectNoError(err, "creating scheduler")
	informerFactory.Start(tCtx.Done())
	informerFactory.WaitForCacheSync(tCtx.Done())
	tCtx.ExpectNoError(sched.WaitForHandlersSync(tCtx), "waiting for handlers to sync")
	done := make(chan struct{})
	go func() {
		sched.Run(tCtx)
		close(done)
	}()
	e.tCtx.CleanupCtx(func(ktesting.TContext) {
		tCtx.Cancel("stop sharing affinity diagnostics scheduler")
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			e.tCtx.Fatal("sharing affinity scheduler did not stop within 30s")
		}
		informerFactory.Shutdown()
	})
	return store
}

// failedSchedulingEvents waits until the API has a FailedScheduling Event for
// the pod containing all fragments and checks none of them leak secrets.
// Persisted Events may be coalesced, so this is message evidence only.
func (s *saScope) failedSchedulingEvents(pod *v1.Pod, contains []string, absent ...string) []v1.Event {
	s.tCtx.Helper()
	var events []v1.Event
	s.tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		list, err := tCtx.Client().CoreV1().Events(s.ns).List(tCtx, metav1.ListOptions{
			FieldSelector: "involvedObject.kind=Pod,involvedObject.name=" + pod.Name + ",reason=FailedScheduling",
		})
		if err != nil {
			return "", err
		}
		events = list.Items
		var msgs []string
		for _, e := range list.Items {
			msgs = append(msgs, e.Message)
		}
		return strings.Join(msgs, "\n"), nil
	}).WithTimeout(schedulingTimeout).Should(gomega.And(append([]gomega.OmegaMatcher{gomega.Not(gomega.BeEmpty())},
		saContainAll(contains)...)...), "FailedScheduling event for pod %s", pod.Name)
	for _, e := range events {
		for _, a := range absent {
			if strings.Contains(e.Message, a) {
				s.tCtx.Fatalf("event for pod %s leaks %q: %s", pod.Name, a, e.Message)
			}
		}
	}
	return events
}

// saLogs gives access to the scheduler's buffered log output.
type saLogs struct {
	logger klog.Logger
}

func (l saLogs) text() string {
	return l.logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
}

// expect waits for a log line matching pattern (the scheduler logs
// asynchronously relative to the API status write) and asserts that no
// sharing affinity diagnostic line contains any of the secrets.
func (l saLogs) expect(s *saScope, pattern string, absent ...string) {
	s.tCtx.Helper()
	re := regexp.MustCompile(pattern)
	s.tCtx.Eventually(func(ktesting.TContext) bool {
		return re.MatchString(l.text())
	}).WithTimeout(schedulingTimeout).Should(gomega.BeTrueBecause("scheduler log should match %q", pattern))
	// The whole buffer is checked: any scheduler log entry, including V(5+)
	// allocation dumps, must not disclose CEL expressions or opaque values.
	text := l.text()
	for _, a := range absent {
		if strings.Contains(text, a) {
			s.tCtx.Fatalf("scheduler log leaks %q:\n%s", a, text)
		}
	}
}

var saLogHeader = regexp.MustCompile(`^\s*(\S+\.go:\d+: )?[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d+\]`)

func saLogEntries(text string) []string {
	var entries []string
	for _, line := range strings.Split(text, "\n") {
		if saLogHeader.MatchString(line) || len(entries) == 0 {
			entries = append(entries, line)
			continue
		}
		entries[len(entries)-1] += "\n" + line
	}
	return entries
}

// saBudgetFixture is the admissible shared-runtime-budget fixture from the
// design (§5.1): constant cost per evaluation, admission-accepted, exhausted
// by the fifth claim-wide config because one budget covers all configs.
func saBudgetExpression() string {
	l := "["
	for i := range 16 {
		if i > 0 {
			l += ","
		}
		l += fmt.Sprint(i)
	}
	l += "]"
	return `has(parameters.kind) && ` + l + `.all(i, ` + l + `.all(j, "` + strings.Repeat("a", 4000) + `".matches("a{1,3}"))) ? "x" : "x"`
}

func saAttempts(s *saScope, result string) float64 {
	s.tCtx.Helper()
	values, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "scheduler_schedule_attempts_total",
		map[string]string{"profile": "default-scheduler"}, "result")
	s.tCtx.ExpectNoError(err, "read scheduler_schedule_attempts_total")
	return values[result]
}

// TestSharingAffinityDiagnostics drives the real scheduler and checks that
// each KEP diagnostic category reaches the pod condition, the
// FailedScheduling Event and the scheduler log, without leaking claim
// configuration values or CEL source.
func TestSharingAffinityDiagnostics(t *testing.T) {
	env := newSAEnv(t)
	logs := saLogs{logger: klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.Verbosity(7), klogtesting.BufferLogs(true)))}
	spy := env.startSchedulerWithRecorderSpy(logs.logger)

	// Runs first: the attempt metric is global, so this window must be the
	// only pending pod in the scheduler.
	env.run("mismatch", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr),
			shared("nic-0", "8Gi"), shared("nic-1", "8Gi"), shared("nic-2", "8Gi"))
		var owners []*v1.Pod
		var ownerClaims []*resourceapi.ResourceClaim
		for _, n := range []string{"a", "b", "c"} {
			c := s.nicClaim(n, "1Gi", subnet("subnet-"+n))
			owners = append(owners, s.scheduled(s.pod("p"+n, "", c)))
			ownerClaims = append(ownerClaims, c)
		}
		secrets := []string{"secret-subnet-d", "subnet-a", "subnet-b", "subnet-c", saSubnetExpr}

		baseline := saAttempts(s, "unschedulable")
		spy.reset()
		d := s.nicClaim("d", "1Gi", subnet("secret-subnet-d"))
		pd := s.pod("pd", "", d)
		s.notScheduled(pd, v1.PodReasonUnschedulable, []string{saDiagMismatch}, secrets...)

		count := func() int {
			msgs, _ := spy.failed(pd.UID)
			return len(msgs)
		}
		s.tCtx.Eventually(func(ktesting.TContext) int { return count() }).WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">=", 1))
		// Node label updates requeue the unschedulable pod for new attempts.
		for i := range 2 {
			before := count()
			node, err := s.tCtx.Client().CoreV1().Nodes().Get(s.tCtx, "worker-1", metav1.GetOptions{})
			s.tCtx.ExpectNoError(err, "get unrelated node")
			node.Labels["sharing-affinity-diag-requeue"] = fmt.Sprintf("%s-%d", s.ns, i)
			_, err = s.tCtx.Client().CoreV1().Nodes().Update(s.tCtx, node, metav1.UpdateOptions{})
			s.tCtx.ExpectNoError(err, "update unrelated node label")
			s.tCtx.Eventually(func(ktesting.TContext) int { return count() }).WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">", before))
		}

		// Quiescent window: three consecutive equal reads spanning >=2s with
		// metric delta == recorder count and nothing inside Eventf.
		var attempts float64
		var msgs []string
		stable := 0
		var last time.Time
		var prevAttempts float64 = -1
		prevCount := -1
		err := wait.PollUntilContextTimeout(s.tCtx, 100*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
			a := saAttempts(s, "unschedulable") - baseline
			m, inFlight := spy.failed(pd.UID)
			if inFlight != 0 || a != float64(len(m)) || a != prevAttempts || len(m) != prevCount {
				prevAttempts, prevCount, stable, last = a, len(m), 0, time.Now()
				return false, nil
			}
			stable++
			if stable >= 3 && time.Since(last) >= 2*time.Second {
				attempts, msgs = a, m
				return true, nil
			}
			return false, nil
		})
		s.tCtx.ExpectNoError(err, "scheduler did not quiesce with matching attempt metric and FailedScheduling emissions")
		if attempts < 3 {
			s.tCtx.Fatalf("want >=3 unschedulable attempts for pod %s, got %v", pd.Name, attempts)
		}
		if float64(len(msgs)) != attempts {
			s.tCtx.Fatalf("FailedScheduling emissions %d != attempts %v: three rejected devices must not add per-device events", len(msgs), attempts)
		}
		for _, m := range msgs {
			if !strings.Contains(m, saDiagMismatch) {
				s.tCtx.Fatalf("emission lacks mismatch fragment: %s", m)
			}
			for _, a := range append([]string{"nic-0", "nic-1", "nic-2", saDiagUnknown, saDiagUnsatisfied}, secrets...) {
				if strings.Contains(m, a) {
					s.tCtx.Fatalf("emission contains %q: %s", a, m)
				}
			}
		}
		s.unallocated(d)
		s.failedSchedulingEvents(pd, []string{saDiagMismatch}, append([]string{"nic-0", "nic-1", "nic-2"}, secrets...)...)
		logs.expect(s, saDiagRejectLog+`.*"`+s.ns+`/pd".*mismatch=[1-9]`, secrets...)

		s.release(pd, d)
		for i, o := range owners {
			s.release(o, ownerClaims[i])
		}
	})

	env.run("alternative-candidate-succeeds", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "8Gi"), shared("nic-1", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.scheduled(s.pod("pb", "", b))
		if s.device(a) == s.device(b) {
			s.tCtx.Fatalf("mismatching claims co-allocated on %s", s.device(a))
		}
		if msgs, _ := spy.failed(pb.UID); len(msgs) > 0 {
			s.tCtx.Fatalf("rejected sibling candidate produced FailedScheduling for schedulable pod: %v", msgs)
		}
		list, err := s.tCtx.Client().CoreV1().Events(s.ns).List(s.tCtx, metav1.ListOptions{})
		s.tCtx.ExpectNoError(err, "list events")
		for _, e := range list.Items {
			if e.InvolvedObject.Name == pb.Name && e.Reason == "FailedScheduling" {
				s.tCtx.Fatalf("rejected sibling candidate produced FailedScheduling for schedulable pod: %s", e.Message)
			}
		}
	})

	env.run("unsatisfied", func(s *saScope) {
		vlanExpr := `parameters.?vlanID.orValue("")`
		extractors := append(subnetExtractor(saSubnetExpr),
			resourceapi.SharingAffinityExtractor{Name: "vlan", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: vlanExpr})
		s.slice("s", "pool", 1, 1, extractors, shared("nic", "8Gi"))
		// subnet extracts successfully; its vlan sibling yields "".
		a := s.nicClaim("a", "1Gi", subnet("secret-subnet-u"))
		pa := s.pod("pa", "", a)
		secrets := []string{"secret-subnet-u", saSubnetExpr, vlanExpr}
		s.notScheduled(pa, v1.PodReasonUnschedulable, []string{saDiagUnsatisfied}, secrets...)
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{saDiagUnsatisfied}, secrets...)
		logs.expect(s, saDiagRejectLog+`.*"`+s.ns+`/pa".*unsatisfied=[1-9]`, secrets...)
	})

	env.run("evaluation-error/non-string", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", nicParams(`"subnetID":987654321`))
		pa := s.pod("pa", "", a)
		cond := s.notScheduled(pa, v1.PodReasonSchedulerError, []string{`extractor "subnet"`, "non-string"}, "987654321")
		for _, c := range cond.Status.Conditions {
			if strings.Contains(c.Message, saDiagUnsatisfied) {
				s.tCtx.Fatalf("evaluation error reported as unsatisfied: %s", c.Message)
			}
		}
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{"non-string"}, "987654321")
		logs.expect(s, saDiagExtractLog+`.*non-string`, "987654321")
	})

	env.run("evaluation-error/missing-field", func(s *saScope) {
		expr := `parameters.subnetID`
		s.slice("s", "pool", 1, 1, subnetExtractor(expr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", nicParams(`"vlanID":"secret-vlan-m"`))
		pa := s.pod("pa", "", a)
		secrets := []string{"secret-vlan-m", expr}
		cond := s.notScheduled(pa, v1.PodReasonSchedulerError, []string{`extractor "subnet"`, "missing field"}, secrets...)
		for _, c := range cond.Status.Conditions {
			if strings.Contains(c.Message, saDiagUnsatisfied) {
				s.tCtx.Fatalf("missing field reported as unsatisfied: %s", c.Message)
			}
		}
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{"missing field"}, secrets...)
		logs.expect(s, saDiagExtractLog+`.*missing field`, secrets...)
	})

	env.run("evaluation-error/cost-exhausted", func(s *saScope) {
		expr := saBudgetExpression()
		if len(expr) > resourceapi.SharingAffinityCELExpressionMaxLength {
			s.tCtx.Fatalf("fixture expression length %d exceeds limit", len(expr))
		}
		extractors := []resourceapi.SharingAffinityExtractor{{Name: "b0", APIGroup: "example.com", Kind: "Config", Expression: expr}}
		// Creation through the API proves the fixture is admissible.
		s.slice("s", "pool", 1, 1, extractors, shared("nic", "8Gi"))
		var configs []saConfig
		for range 8 {
			configs = append(configs, saConfig{params: `{"apiVersion":"example.com/v1","kind":"Config"}`})
		}
		a := s.claim("a", []saRequest{{name: "nic", memory: "1Gi"}}, configs...)
		pa := s.pod("pa", "", a)
		re := regexp.MustCompile(`config [0-7]\b`)
		cond := s.notScheduled(pa, v1.PodReasonSchedulerError, []string{`extractor "b0"`, "cost exceeded"}, strings.Repeat("a", 64))
		for _, c := range cond.Status.Conditions {
			if c.Type == v1.PodScheduled && !re.MatchString(c.Message) {
				s.tCtx.Fatalf("cost exhaustion should name a config index < 8 (shared budget): %s", c.Message)
			}
		}
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{"cost exceeded"}, strings.Repeat("a", 64))
		logs.expect(s, saDiagExtractLog+`.*cost`, expr, strings.Repeat("a", 64))
	})

	env.run("unknown-legacy-owner", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("dev1"))
		legacy := saSeedOwner(s, "legacy", "pool", "dev1", "")
		next := saRestartClaim(s, "next", subnet("secret-next"), "dev1")
		pnext := s.pod("pnext", "", next)
		s.notScheduled(pnext, v1.PodReasonUnschedulable, []string{saDiagUnknown}, "secret-next", saDiagPending)
		s.staysPending(pnext, next)
		s.failedSchedulingEvents(pnext, []string{saDiagUnknown}, "secret-next")
		logs.expect(s, saDiagRejectLog+`.*"`+s.ns+`/pnext".*unknown=[1-9]`, "secret-next")
		// Releasing the legacy owner clears the unknown state.
		s.release(legacy.pod, legacy.claim)
		s.scheduled(pnext)
		saOnDevice(s, next, "pool/dev1")
	})
}
