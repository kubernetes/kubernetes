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
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	resourcev1alpha3 "k8s.io/api/resource/v1alpha3"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/resourcepoolstatusrequest"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

const (
	scalePools           = 100
	scaleExpiredRequests = 1000
	scaleFreshRequests   = 10
	scaleDriver          = "scale.example.com"

	// Sweep pacing: 10/s plus a burst of 10.
	maxDeletesPerSecond = 20
	// The sweep interval.
	maxSweepDuration = 10 * time.Minute
)

// TestResourcePoolStatusRequestScale checks the TTL sweep with 100 pools and
// 1000 expired requests. Skipped with -short; run with:
//
//	make test-integration WHAT=./test/integration/dra SHORT=--short=false KUBE_TEST_ARGS='-run ^TestResourcePoolStatusRequestScale$'
func TestResourcePoolStatusRequestScale(t *testing.T) {
	if testing.Short() {
		t.Skip("takes about two minutes")
	}
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, map[featuregate.Feature]bool{
		features.DRAResourcePoolStatus: true,
	})
	apiServerFlags := append(framework.DefaultTestServerFlags(), "--runtime-config=resource.k8s.io/v1alpha3=true")
	server := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), apiServerFlags, framework.SharedEtcd())
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) { server.TearDownFn() })
	tCtx = tCtx.WithRESTConfig(server.ClientConfig)
	requests := tCtx.Client().ResourceV1alpha3().ResourcePoolStatusRequests()

	parallel(tCtx, scalePools, func(i int) error {
		devices := make([]resourceapi.Device, 8)
		for d := range devices {
			devices[d] = resourceapi.Device{Name: fmt.Sprintf("device-%d", d)}
		}
		_, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("scale-slice-%03d", i)},
			Spec: resourceapi.ResourceSliceSpec{
				Driver:   scaleDriver,
				Pool:     resourceapi.ResourcePool{Name: fmt.Sprintf("pool-%03d", i), ResourceSliceCount: 1, Generation: 1},
				NodeName: new(fmt.Sprintf("node-%03d", i)),
				Devices:  devices,
			},
		}, metav1.CreateOptions{})
		return err
	})

	// Expired: completed 2h ago (TTL 1h). Fresh: completed now.
	createCompleted := func(name string, completedAt time.Time) error {
		request, err := requests.Create(tCtx, &resourcev1alpha3.ResourcePoolStatusRequest{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec:       resourcev1alpha3.ResourcePoolStatusRequestSpec{Driver: scaleDriver},
		}, metav1.CreateOptions{})
		if err != nil {
			return err
		}
		request.Status = &resourcev1alpha3.ResourcePoolStatusRequestStatus{
			PoolCount: ptr.To[int32](0),
			Conditions: []metav1.Condition{{
				Type:               resourcev1alpha3.ResourcePoolStatusRequestConditionComplete,
				Status:             metav1.ConditionTrue,
				Reason:             "CalculationComplete",
				Message:            "Calculated status for 0 pools",
				LastTransitionTime: metav1.NewTime(completedAt),
			}},
		}
		_, err = requests.UpdateStatus(tCtx, request, metav1.UpdateOptions{})
		return err
	}
	parallel(tCtx, scaleExpiredRequests, func(i int) error {
		return createCompleted(fmt.Sprintf("expired-%04d", i), time.Now().Add(-2*time.Hour))
	})
	parallel(tCtx, scaleFreshRequests, func(i int) error {
		return createCompleted(fmt.Sprintf("fresh-%02d", i), time.Now())
	})
	tCtx.Logf("Seeded %d pools, %d expired and %d fresh requests", scalePools, scaleExpiredRequests, scaleFreshRequests)

	list, err := requests.List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list requests")
	watcher, err := requests.Watch(tCtx, metav1.ListOptions{ResourceVersion: list.ResourceVersion})
	tCtx.ExpectNoError(err, "watch requests")
	defer watcher.Stop()

	// KCM default client QPS.
	controllerConfig := rest.CopyConfig(server.ClientConfig)
	controllerConfig.QPS = 20
	controllerConfig.Burst = 30
	// Time deletes when sent; watch events can arrive bunched.
	deletes := &deleteRecorder{}
	controllerConfig.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if req.Method == http.MethodDelete && strings.Contains(req.URL.Path, "/resourcepoolstatusrequests/") {
				deletes.record(time.Now())
			}
			return rt.RoundTrip(req)
		})
	})
	controllerClient := kubernetes.NewForConfigOrDie(controllerConfig)
	controllerCtx := tCtx.WithCancel()
	defer controllerCtx.Cancel("test is done")
	informerFactory := informers.NewSharedInformerFactory(controllerClient, 0)
	controller, err := resourcepoolstatusrequest.NewController(controllerCtx, controllerClient,
		informerFactory.Resource().V1alpha3().ResourcePoolStatusRequests(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1().ResourceClaims(),
		informerFactory.Resource().V1().DeviceTaintRules(),
	)
	tCtx.ExpectNoError(err, "create controller")
	informerFactory.Start(controllerCtx.Done())
	var wg sync.WaitGroup
	defer wg.Wait()
	defer controllerCtx.Cancel("test is done")
	// The first sweep runs after cache sync.
	wg.Go(func() { controller.Run(controllerCtx, 1) })

	var deleted int
	probeCreated := false
	var probeDone sync.WaitGroup
	timeout := time.After(maxSweepDuration)
	for deleted < scaleExpiredRequests {
		select {
		case event, ok := <-watcher.ResultChan():
			if !ok {
				tCtx.Fatalf("watch closed after %d deletes", deleted)
			}
			if event.Type != watch.Deleted {
				continue
			}
			request := event.Object.(*resourcev1alpha3.ResourcePoolStatusRequest)
			if !strings.HasPrefix(request.Name, "expired-") {
				tCtx.Errorf("unexpected delete of %s", request.Name)
				continue
			}
			deleted++
			// Mid-sweep, new requests must still be processed.
			if deleted == 100 && !probeCreated {
				probeCreated = true
				probeDone.Go(func() { checkProbeDuringSweep(tCtx, requests) })
			}
		case <-timeout:
			tCtx.Fatalf("only %d of %d expired requests deleted within %v", deleted, scaleExpiredRequests, maxSweepDuration)
		}
	}
	probeDone.Wait()

	deleteTimes := deletes.times()
	sweep := deleteTimes[len(deleteTimes)-1].Sub(deleteTimes[0])
	maxInWindow := 0
	start := 0
	for end := range deleteTimes {
		for deleteTimes[end].Sub(deleteTimes[start]) >= time.Second {
			start++
		}
		maxInWindow = max(maxInWindow, end-start+1)
	}
	tCtx.Logf("Sent %d deletes in %v, at most %d in any one-second window", len(deleteTimes), sweep, maxInWindow)
	if maxInWindow > maxDeletesPerSecond {
		tCtx.Errorf("sweep deleted %d requests within one second, want at most %d", maxInWindow, maxDeletesPerSecond)
	}

	remaining, err := requests.List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list requests after the sweep")
	// The fresh requests and the probe survive.
	if got, want := len(remaining.Items), scaleFreshRequests+1; got != want {
		tCtx.Errorf("%d requests left after the sweep, want %d", got, want)
	}
}

// checkProbeDuringSweep expects a new request to complete during the sweep.
func checkProbeDuringSweep(tCtx ktesting.TContext, requests interface {
	Create(ctx context.Context, request *resourcev1alpha3.ResourcePoolStatusRequest, opts metav1.CreateOptions) (*resourcev1alpha3.ResourcePoolStatusRequest, error)
	Get(ctx context.Context, name string, opts metav1.GetOptions) (*resourcev1alpha3.ResourcePoolStatusRequest, error)
}) {
	created := time.Now()
	probe, err := requests.Create(tCtx, &resourcev1alpha3.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "probe"},
		Spec:       resourcev1alpha3.ResourcePoolStatusRequestSpec{Driver: scaleDriver, Limit: ptr.To[int32](scalePools)},
	}, metav1.CreateOptions{})
	if err != nil {
		tCtx.Errorf("create probe request: %v", err)
		return
	}
	var status *resourcev1alpha3.ResourcePoolStatusRequestStatus
	for status == nil && time.Since(created) < 10*time.Second {
		time.Sleep(100 * time.Millisecond)
		got, err := requests.Get(tCtx, probe.Name, metav1.GetOptions{})
		if err != nil {
			tCtx.Errorf("get probe request: %v", err)
			return
		}
		status = got.Status
	}
	if status == nil {
		tCtx.Errorf("probe request got no status within 10s while the sweep was running")
		return
	}
	tCtx.Logf("Probe request completed %v after creation, during the sweep", time.Since(created))
	if !apimeta.IsStatusConditionTrue(status.Conditions, resourcev1alpha3.ResourcePoolStatusRequestConditionComplete) {
		tCtx.Errorf("probe conditions = %+v, want Complete", status.Conditions)
	}
	if got := ptr.Deref(status.PoolCount, 0); got != scalePools || len(status.Pools) != scalePools {
		tCtx.Errorf("probe reports poolCount=%d with %d pools, want %d", got, len(status.Pools), scalePools)
	}
}

// parallel runs fn for 0..n-1 concurrently.
func parallel(tCtx ktesting.TContext, n int, fn func(i int) error) {
	tCtx.Helper()
	work := make(chan int)
	errs := make(chan error, n)
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			for i := range work {
				if err := fn(i); err != nil {
					errs <- err
				}
			}
		})
	}
	for i := range n {
		work <- i
	}
	close(work)
	wg.Wait()
	close(errs)
	for err := range errs {
		tCtx.Fatalf("seeding failed: %v", err)
	}
}

type deleteRecorder struct {
	mutex sync.Mutex
	sent  []time.Time
}

func (d *deleteRecorder) record(t time.Time) {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	d.sent = append(d.sent, t)
}

func (d *deleteRecorder) times() []time.Time {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	return slices.Clone(d.sent)
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }
