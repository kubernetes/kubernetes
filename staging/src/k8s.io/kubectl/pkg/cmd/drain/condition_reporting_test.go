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

package drain

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	drainlib "k8s.io/kubectl/pkg/drain"
	"k8s.io/kubectl/pkg/util/interrupt"
)

type recordedCondition struct {
	node  string
	event drainlib.NodeConditionEvent
}

type recordingConditionReporter struct {
	lock      sync.Mutex
	events    []recordedCondition
	err       error
	onStarted func()
	onReport  func(context.Context, drainlib.NodeConditionEvent) error
}

func (r *recordingConditionReporter) Report(ctx context.Context, nodeName string, event drainlib.NodeConditionEvent) error {
	r.lock.Lock()
	r.events = append(r.events, recordedCondition{node: nodeName, event: event})
	onStarted := r.onStarted
	r.lock.Unlock()
	if event == drainlib.NodeConditionDrainStarted && onStarted != nil {
		onStarted()
	}
	if r.onReport != nil {
		if err := r.onReport(ctx, event); err != nil {
			return err
		}
	}
	return r.err
}

func (r *recordingConditionReporter) recordedEvents() []recordedCondition {
	r.lock.Lock()
	defer r.lock.Unlock()
	return append([]recordedCondition(nil), r.events...)
}

func TestRunDrainReportsConditionsPerNode(t *testing.T) {
	reporter := &recordingConditionReporter{}
	o := newConditionTestOptions(t, reporter, "node-1", "node-2")

	if err := o.RunDrain(); err != nil {
		t.Fatalf("RunDrain() error = %v", err)
	}
	want := []recordedCondition{
		{node: "node-1", event: drainlib.NodeConditionDrainStarted},
		{node: "node-1", event: drainlib.NodeConditionDrainCompleted},
		{node: "node-2", event: drainlib.NodeConditionDrainStarted},
		{node: "node-2", event: drainlib.NodeConditionDrainCompleted},
	}
	assertRecordedConditions(t, reporter.recordedEvents(), want)
}

func TestRunDrainReportingFailureIsWarningOnly(t *testing.T) {
	reporter := &recordingConditionReporter{err: errors.New("status denied")}
	o := newConditionTestOptions(t, reporter, "node")
	var errOut bytes.Buffer
	o.ErrOut = &errOut
	o.drainer.ErrOut = &errOut
	o.WarningPrinter = printers.NewWarningPrinter(&errOut, printers.WarningPrinterOptions{})

	if err := o.RunDrain(); err != nil {
		t.Fatalf("RunDrain() error = %v", err)
	}
	if got := errOut.String(); !bytes.Contains([]byte(got), []byte(`Warning: failed to update drain conditions on node "node": status denied`)) {
		t.Fatalf("warning output = %q", got)
	}
}

func TestRunDrainReportingFailureDoesNotReplaceDrainError(t *testing.T) {
	reporter := &recordingConditionReporter{err: errors.New("status denied")}
	o := newConditionTestOptions(t, reporter, "node")
	o.drainer.Client.(*fake.Clientset).PrependReactor("list", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("pods unavailable")
	})

	err := o.RunDrain()
	if err == nil || !strings.Contains(err.Error(), "pods unavailable") {
		t.Fatalf("RunDrain() error = %v, want original pod-list error", err)
	}
	if strings.Contains(err.Error(), "status denied") {
		t.Fatalf("RunDrain() error = %v, must not include reporting error", err)
	}
	want := []recordedCondition{
		{node: "node", event: drainlib.NodeConditionDrainStarted},
		{node: "node", event: drainlib.NodeConditionDrainFailed},
	}
	assertRecordedConditions(t, reporter.recordedEvents(), want)
}

func TestReportConditionUsesBoundedContext(t *testing.T) {
	reporter := &recordingConditionReporter{
		onReport: func(ctx context.Context, event drainlib.NodeConditionEvent) error {
			deadline, ok := ctx.Deadline()
			if !ok {
				t.Fatal("condition report context has no deadline")
			}
			remaining := time.Until(deadline)
			if remaining <= 0 || remaining > conditionReportTimeout {
				t.Fatalf("condition report deadline is %v away, want at most %v", remaining, conditionReportTimeout)
			}
			return nil
		},
	}
	o := newConditionTestOptions(t, reporter, "node")

	o.reportCondition("node", drainlib.NodeConditionDrainStarted)
}

func TestRunUncordonReportsConditionForNoOp(t *testing.T) {
	reporter := &recordingConditionReporter{}
	o := newConditionTestOptions(t, reporter, "node")
	node := o.nodeInfos[0].Object.(*corev1.Node)
	node.Spec.Unschedulable = false

	if err := o.RunCordonOrUncordon(false); err != nil {
		t.Fatalf("RunCordonOrUncordon() error = %v", err)
	}
	want := []recordedCondition{{node: "node", event: drainlib.NodeConditionUncordoned}}
	assertRecordedConditions(t, reporter.recordedEvents(), want)
}

func TestRunDrainInterruptReportsActiveNode(t *testing.T) {
	interrupted := make(chan struct{})
	reporter := &recordingConditionReporter{
		onReport: func(ctx context.Context, event drainlib.NodeConditionEvent) error {
			if event == drainlib.NodeConditionDrainInterrupted {
				close(interrupted)
			}
			return nil
		},
	}
	started := make(chan struct{})
	reporter.onStarted = sync.OnceFunc(func() { close(started) })
	o := newConditionTestOptions(t, reporter, "node-1", "node-2")

	releaseList := make(chan struct{})
	o.drainer.Client.(*fake.Clientset).PrependReactor("list", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		<-releaseList
		return true, nil, context.Canceled
	})
	exitCodes := make(chan int, 1)
	sendSignal := installTestSignalHandler(o, exitCodes)

	result := make(chan error, 1)
	go func() {
		result <- o.RunDrain()
	}()
	<-started
	sendSignal(syscall.SIGINT)
	select {
	case <-interrupted:
	case <-time.After(time.Second):
		t.Fatal("interrupted condition was not reported while pod processing was blocked")
	}
	close(releaseList)

	if err := <-result; err != nil {
		t.Fatalf("RunDrain() error = %v", err)
	}
	if got := <-exitCodes; got != 1 {
		t.Fatalf("exit code = %d, want 1", got)
	}
	want := []recordedCondition{
		{node: "node-1", event: drainlib.NodeConditionDrainStarted},
		{node: "node-1", event: drainlib.NodeConditionDrainInterrupted},
	}
	assertRecordedConditions(t, reporter.recordedEvents(), want)
}

func TestRunDrainInterruptDuringTerminalReportCleansUpActiveNode(t *testing.T) {
	completionStarted := make(chan struct{})
	reporter := &recordingConditionReporter{
		onReport: func(ctx context.Context, event drainlib.NodeConditionEvent) error {
			if event != drainlib.NodeConditionDrainCompleted {
				return nil
			}
			close(completionStarted)
			<-ctx.Done()
			return ctx.Err()
		},
	}
	o := newConditionTestOptions(t, reporter, "node")
	exitCodes := make(chan int, 1)
	sendSignal := installTestSignalHandler(o, exitCodes)

	result := make(chan error, 1)
	go func() {
		result <- o.RunDrain()
	}()
	<-completionStarted
	sendSignal(syscall.SIGINT)

	if err := <-result; err != nil {
		t.Fatalf("RunDrain() error = %v", err)
	}
	if got := <-exitCodes; got != 1 {
		t.Fatalf("exit code = %d, want 1", got)
	}
	want := []recordedCondition{
		{node: "node", event: drainlib.NodeConditionDrainStarted},
		{node: "node", event: drainlib.NodeConditionDrainCompleted},
		{node: "node", event: drainlib.NodeConditionDrainInterrupted},
	}
	assertRecordedConditions(t, reporter.recordedEvents(), want)
}

func installTestSignalHandler(o *DrainCmdOptions, exitCodes chan<- int) func(os.Signal) {
	ready := make(chan struct{})
	var handler *interrupt.Handler
	o.newSignalHandler = func(first func(os.Signal)) *interrupt.Handler {
		handler = interrupt.New(first)
		close(ready)
		return handler
	}
	o.exit = func(code int) {
		exitCodes <- code
	}
	return func(received os.Signal) {
		<-ready
		go handler.Signal(received)
	}
}

func newConditionTestOptions(t *testing.T, reporter drainlib.ConditionReporter, nodeNames ...string) *DrainCmdOptions {
	t.Helper()
	objects := make([]runtime.Object, 0, len(nodeNames))
	infos := make([]*resource.Info, 0, len(nodeNames))
	mapping := &meta.RESTMapping{
		Resource:         corev1.SchemeGroupVersion.WithResource("nodes"),
		GroupVersionKind: corev1.SchemeGroupVersion.WithKind("Node"),
		Scope:            meta.RESTScopeRoot,
	}
	for _, name := range nodeNames {
		node := &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec:       corev1.NodeSpec{Unschedulable: true},
		}
		objects = append(objects, node)
		infos = append(infos, &resource.Info{
			Name:    name,
			Object:  node,
			Mapping: mapping,
		})
	}
	client := fake.NewClientset(objects...)
	streams, _, out, errOut := genericiooptions.NewTestIOStreams()
	o := &DrainCmdOptions{
		IOStreams: streams,
		drainer: &drainlib.Helper{
			Client:            client,
			Ctx:               context.Background(),
			Out:               out,
			ErrOut:            errOut,
			DisableEviction:   true,
			ConditionReporter: reporter,
			DryRunStrategy:    cmdutil.DryRunNone,
		},
		nodeInfos: infos,
		ToPrinter: func(operation string) (printers.ResourcePrinterFunc, error) {
			return func(runtime.Object, io.Writer) error { return nil }, nil
		},
		WarningPrinter: printers.NewWarningPrinter(errOut, printers.WarningPrinterOptions{}),
	}
	return o
}

func assertRecordedConditions(t *testing.T, got, want []recordedCondition) {
	t.Helper()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %#v, want %#v", got, want)
	}
}
