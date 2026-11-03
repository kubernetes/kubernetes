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

package portforward

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	clientgofake "k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	streamhttp "k8s.io/streaming/pkg/httpstream"
	streamspdy "k8s.io/streaming/pkg/httpstream/spdy"
)

type forwardingAddressWriter struct {
	addresses chan string
}

func (w forwardingAddressWriter) Write(p []byte) (int, error) {
	var address string
	if n, _ := fmt.Sscanf(string(p), "Forwarding from %s ->", &address); n == 1 {
		select {
		case w.addresses <- address:
		default:
		}
	}
	return len(p), nil
}

type forwardingDiagnosticLogger struct {
	diagnostics chan<- string
}

func (forwardingDiagnosticLogger) Helper() {}

func (l forwardingDiagnosticLogger) Log(args ...interface{}) {
	select {
	case l.diagnostics <- fmt.Sprint(args...):
	default:
	}
}

type portForwardLifecycle struct {
	ctx           context.Context
	cancel        context.CancelFunc
	diagnostics   chan string
	pod           *corev1.Pod
	client        *clientgofake.Clientset
	stop          chan struct{}
	done          chan struct{}
	err           error
	address       string
	connection    net.Conn
	watches       chan *watch.RaceFreeFakeWatcher
	transportDone chan struct{}
	lists         atomic.Int32
	watchLock     sync.Mutex
	allWatches    []*watch.RaceFreeFakeWatcher
}

func startPortForwardLifecycle(t *testing.T, configure ...func(*portForwardLifecycle)) *portForwardLifecycle {
	t.Helper()
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, false)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	f := &portForwardLifecycle{
		ctx: ctx, cancel: cancel, diagnostics: make(chan string, 20),
		pod: &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", UID: "original", ResourceVersion: "1"},
			Status:     corev1.PodStatus{Phase: corev1.PodRunning},
		},
		stop: make(chan struct{}), done: make(chan struct{}),
		watches: make(chan *watch.RaceFreeFakeWatcher, 10), transportDone: make(chan struct{}),
	}
	ctx = klog.NewContext(ctx, ktesting.NewLogger(
		forwardingDiagnosticLogger{diagnostics: f.diagnostics},
		ktesting.NewConfig(ktesting.Verbosity(0)),
	))
	f.ctx = ctx

	f.client = clientgofake.NewSimpleClientset(f.pod)
	f.client.PrependReactor("list", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		f.lists.Add(1)
		obj, err := f.client.Tracker().List(corev1.SchemeGroupVersion.WithResource("pods"), corev1.SchemeGroupVersion.WithKind("Pod"), "test")
		if err == nil {
			obj.(*corev1.PodList).ResourceVersion = "1"
		}
		return true, obj, err
	})
	f.client.PrependWatchReactor("pods", func(clienttesting.Action) (bool, watch.Interface, error) {
		watcher := watch.NewRaceFreeFake()
		f.watchLock.Lock()
		f.allWatches = append(f.allWatches, watcher)
		f.watchLock.Unlock()
		select {
		case f.watches <- watcher:
		case <-ctx.Done():
			watcher.Stop()
		}
		return true, watcher, nil
	})
	for _, configureClient := range configure {
		configureClient(f)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Exercise the normal websocket-to-SPDY fallback against a real upgrade.
		if r.Method != http.MethodPost {
			http.Error(w, "this test server accepts SPDY only", http.StatusBadRequest)
			return
		}
		if _, err := streamhttp.Handshake(r, w, []string{portforward.PortForwardProtocolV1Name}); err != nil {
			return
		}
		conn := streamspdy.NewResponseUpgrader().UpgradeResponse(w, r, func(stream streamhttp.Stream, replySent <-chan struct{}) error {
			if stream.Headers().Get(corev1.StreamType) == corev1.StreamTypeData {
				go func() {
					<-replySent
					_, _ = io.Copy(stream, stream)
				}()
			}
			return nil
		})
		if conn == nil {
			return
		}
		defer close(f.transportDone)
		defer func() { _ = conn.Close() }()
		<-conn.CloseChan()
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() {
		cancel()
		select {
		case <-f.done:
		case <-time.After(5 * time.Second):
			t.Error("forwarding did not stop during cleanup")
		}
		if f.connection != nil {
			_ = f.connection.Close()
		}
	})
	addresses := make(chan string, 1)
	ready := make(chan struct{})
	opts := PortForwardOptions{
		PodClient: f.client.CoreV1(), Namespace: "test", PodName: f.pod.Name, PodUID: f.pod.UID,
		Config: &restclient.Config{Host: server.URL}, Address: []string{"127.0.0.1"}, Ports: []string{":8080"},
		StopChannel: f.stop, ReadyChannel: ready,
	}
	forwarder := &defaultPortForwarder{IOStreams: genericiooptions.IOStreams{
		Out: forwardingAddressWriter{addresses}, ErrOut: io.Discard,
	}}
	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		defer close(f.done)
		f.err = forwarder.ForwardPorts(ctx, http.MethodPost, serverURL, opts)
	}()
	select {
	case <-ready:
	case <-f.done:
		t.Fatalf("forwarding stopped before becoming ready: %v", f.err)
	case <-ctx.Done():
		t.Fatal("forwarding did not become ready")
	}
	select {
	case f.address = <-addresses:
	case <-ctx.Done():
		t.Fatal("forwarding did not report a listen address")
	}
	f.connection = f.dial(t)
	f.echo(t, f.connection)
	return f
}

func (f *portForwardLifecycle) nextWatch(t *testing.T) *watch.RaceFreeFakeWatcher {
	t.Helper()
	select {
	case w := <-f.watches:
		return w
	case <-f.done:
		t.Fatalf("forwarding stopped while waiting for a watch: %v", f.err)
	case <-f.ctx.Done():
		t.Fatal("pod watch did not start")
	}
	return nil
}

func (f *portForwardLifecycle) dial(t *testing.T) net.Conn {
	t.Helper()
	conn, err := net.DialTimeout("tcp4", f.address, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if err := conn.SetDeadline(time.Now().Add(15 * time.Second)); err != nil {
		t.Fatal(err)
	}
	return conn
}

func (f *portForwardLifecycle) echo(t *testing.T, conn net.Conn) {
	t.Helper()
	const payload = "still forwarding"
	if _, err := io.WriteString(conn, payload); err != nil {
		t.Fatal(err)
	}
	response := make([]byte, len(payload))
	if _, err := io.ReadFull(conn, response); err != nil {
		t.Fatal(err)
	}
	if string(response) != payload {
		t.Fatalf("forwarded response = %q, want %q", response, payload)
	}
}

func (f *portForwardLifecycle) assertStopped(t *testing.T, wantErr error) {
	t.Helper()
	select {
	case <-f.done:
	case <-time.After(5 * time.Second):
		t.Fatal("forwarding did not stop")
	}
	if !errors.Is(f.err, wantErr) {
		t.Errorf("ForwardPorts() = %v, want %v", f.err, wantErr)
	}
	select {
	case <-f.transportDone:
	case <-time.After(5 * time.Second):
		t.Error("shared streaming connection did not close")
	}
	if err := f.connection.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.connection.Read(make([]byte, 1)); err == nil {
		t.Error("existing local connection remained open")
	} else {
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			t.Errorf("existing local connection was not closed: %v", err)
		}
	}
	if conn, err := net.DialTimeout("tcp4", f.address, time.Second); err == nil {
		_ = conn.Close()
		t.Error("local listener still accepts connections")
	}
	f.watchLock.Lock()
	defer f.watchLock.Unlock()
	for _, w := range f.allWatches {
		if !w.IsStopped() {
			t.Error("pod watch was not stopped")
		}
	}
}

func TestPortForwardPodLifecycle(t *testing.T) {
	for _, scenario := range []string{"idle pod deletion", "watch reconnect", "expired resource version", "interrupt"} {
		t.Run(scenario, func(t *testing.T) {
			f := startPortForwardLifecycle(t)
			watcher := f.nextWatch(t)
			updated := f.pod.DeepCopy()
			updated.ResourceVersion = "2"
			updated.Labels = map[string]string{"updated": "true"}
			watcher.Modify(updated)
			select {
			case <-f.done:
				t.Fatalf("a Running pod update stopped forwarding: %v", f.err)
			case <-time.After(100 * time.Millisecond):
			}
			f.echo(t, f.connection)
			if scenario == "interrupt" {
				// RunPortForwardContext closes this channel when it receives Ctrl+C.
				close(f.stop)
				f.assertStopped(t, nil)
				return
			}
			if scenario == "watch reconnect" {
				watcher.Stop()
				watcher = f.nextWatch(t)
				f.echo(t, f.connection)
				another := f.dial(t)
				f.echo(t, another)
			}
			if err := f.client.Tracker().Delete(corev1.SchemeGroupVersion.WithResource("pods"), "test", f.pod.Name); err != nil {
				t.Fatal(err)
			}
			if scenario == "expired resource version" {
				// Deletion happens during the gap: only a relist can discover it.
				watcher.Error(&metav1.Status{Status: metav1.StatusFailure, Reason: metav1.StatusReasonExpired, Code: http.StatusGone})
			} else {
				watcher.Delete(updated)
			}
			f.assertStopped(t, portforward.ErrLostConnectionToPod)
			if scenario == "expired resource version" && f.lists.Load() < 2 {
				t.Error("expired resource version did not trigger a relist")
			}
		})
	}
}

func TestPortForwardForbiddenInitialList(t *testing.T) {
	f := startPortForwardLifecycle(t, func(f *portForwardLifecycle) {
		f.client.PrependReactor("list", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, f.pod.Name,
				fmt.Errorf("user cannot list resource pods"))
		})
	})
	for {
		select {
		case diagnostic := <-f.diagnostics:
			if strings.Contains(diagnostic, "forbidden") && strings.Contains(diagnostic, "cannot list resource pods") {
				// Monitoring failure is currently reported and retried; it does
				// not prevent forwarding or leave the command uncancellable.
				f.echo(t, f.connection)
				f.cancel()
				f.assertStopped(t, nil)
				return
			}
		case <-f.done:
			t.Fatalf("forwarding stopped on a forbidden list: %v", f.err)
		case <-time.After(5 * time.Second):
			t.Fatal("forbidden pod list did not produce an actionable diagnostic")
		}
	}
}

func TestPortForwardCancelDuringMonitoringRetries(t *testing.T) {
	for _, operation := range []string{"list", "watch"} {
		t.Run(operation, func(t *testing.T) {
			attempts := make(chan struct{}, 10)
			f := startPortForwardLifecycle(t, func(f *portForwardLifecycle) {
				failure := func() error {
					select {
					case attempts <- struct{}{}:
					default:
					}
					return apierrors.NewInternalError(fmt.Errorf("pod monitoring unavailable"))
				}
				if operation == "list" {
					f.client.PrependReactor("list", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
						return true, nil, failure()
					})
				} else {
					f.client.PrependWatchReactor("pods", func(clienttesting.Action) (bool, watch.Interface, error) {
						return true, nil, failure()
					})
				}
			})
			for range 2 {
				select {
				case <-attempts:
				case <-f.done:
					t.Fatalf("forwarding stopped during monitoring retries: %v", f.err)
				case <-f.ctx.Done():
					t.Fatalf("pod %s failure was not retried", operation)
				}
			}
			f.echo(t, f.connection)
			f.cancel()
			f.assertStopped(t, nil)
		})
	}
}

func TestPortForwardConcurrentPodDeletionAndInterrupt(t *testing.T) {
	for i := range 10 {
		t.Run(fmt.Sprintf("iteration-%d", i), func(t *testing.T) {
			f := startPortForwardLifecycle(t)
			watcher := f.nextWatch(t)
			start := make(chan struct{})
			var workers sync.WaitGroup
			workers.Add(2)
			go func() {
				defer workers.Done()
				<-start
				watcher.Delete(f.pod)
			}()
			go func() {
				defer workers.Done()
				<-start
				close(f.stop)
			}()
			close(start)
			workers.Wait()
			select {
			case <-f.done:
			case <-time.After(5 * time.Second):
				t.Fatal("concurrent pod deletion and interrupt did not stop forwarding")
			}
			// Either cancellation cause can win, but both must clean up the
			// same session without closing shared channels twice.
			if f.err != nil && !errors.Is(f.err, portforward.ErrLostConnectionToPod) {
				t.Fatalf("unexpected forwarding error: %v", f.err)
			}
			f.assertStopped(t, f.err)
		})
	}
}
