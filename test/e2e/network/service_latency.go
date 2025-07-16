/*
Copyright 2015 The Kubernetes Authors.

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

package network

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

type durations []time.Duration

func (d durations) Len() int           { return len(d) }
func (d durations) Less(i, j int) bool { return d[i] < d[j] }
func (d durations) Swap(i, j int)      { d[i], d[j] = d[j], d[i] }

var _ = common.SIGDescribe("Service endpoints latency", func() {
	f := framework.NewDefaultFramework("svc-latency")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release: v1.9
		Testname: Service endpoint latency, thresholds
		Description: Run 100 iterations of create service with the Pod running the pause image, measure the time it takes for creating the service and the endpoint with the service name is available. These durations are captured for 100 iterations, then the durations are sorted to compute 50th, 90th and 99th percentile. The single server latency MUST not exceed liberally set thresholds of 20s for 50th percentile and 50s for the 90th percentile.
	*/
	framework.ConformanceIt("should not be very high", func(ctx context.Context) {
		const (
			// These are very generous criteria. Ideally we will
			// get this much lower in the future. See issue
			// #10436.
			limitMedian = time.Second * 20
			limitTail   = time.Second * 50

			// Numbers chosen to make the test complete in a short amount
			// of time. This sample size is not actually large enough to
			// reliably measure tails (it may give false positives, but not
			// false negatives), but it should catch low hanging fruit.
			//
			// Note that these are fixed and do not depend on the
			// size of the cluster. Setting parallelTrials larger
			// distorts the measurements. Perhaps this wouldn't be
			// true on HA clusters.
			totalTrials    = 200
			parallelTrials = 15
			minSampleSize  = 100

			// Acceptable failure ratio for getting service latencies.
			acceptableFailureRatio = .05
		)

		// Turn off rate limiting--it interferes with our measurements.
		cfg, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("Unable to load config: %v", err)
		}
		cfg.RateLimiter = flowcontrol.NewFakeAlwaysRateLimiter()
		f.ClientSet = kubernetes.NewForConfigOrDie(cfg)

		failing := sets.NewString()
		d, err := runServiceLatencies(ctx, f, parallelTrials, totalTrials, acceptableFailureRatio)
		if err != nil {
			failing.Insert(fmt.Sprintf("Not all RC/pod/service trials succeeded: %v", err))
		}
		dSorted := durations(d)
		sort.Sort(dSorted)
		n := len(dSorted)
		if n < minSampleSize {
			failing.Insert(fmt.Sprintf("Did not get a good sample size: %v", dSorted))
		}
		if n < 2 {
			failing.Insert("Less than two runs succeeded; aborting.")
			framework.Fail(strings.Join(failing.List(), "\n"))
		}
		percentile := func(p int) time.Duration {
			est := n * p / 100
			if est >= n {
				return dSorted[n-1]
			}
			return dSorted[est]
		}
		framework.Logf("Latencies: %v", dSorted)
		p50 := percentile(50)
		p90 := percentile(90)
		p99 := percentile(99)
		framework.Logf("50 %%ile: %v", p50)
		framework.Logf("90 %%ile: %v", p90)
		framework.Logf("99 %%ile: %v", p99)
		framework.Logf("Total sample count: %v", len(dSorted))

		if p50 > limitMedian {
			failing.Insert("Median latency should be less than " + limitMedian.String())
		}
		if p99 > limitTail {
			failing.Insert("Tail (99 percentile) latency should be less than " + limitTail.String())
		}
		if failing.Len() > 0 {
			errList := strings.Join(failing.List(), "\n")
			helpfulInfo := fmt.Sprintf("\n50, 90, 99 percentiles: %v %v %v", p50, p90, p99)
			framework.Fail(errList + helpfulInfo)
		}
	})
})

func runServiceLatencies(ctx context.Context, f *framework.Framework, inParallel, total int, acceptableFailureRatio float32) (output []time.Duration, err error) {
	name := "svc-latency-rc"
	deploymentConf := e2edeployment.NewDeployment(name, 1, map[string]string{"name": name}, name, imageutils.GetPauseImageName(), appsv1.RecreateDeploymentStrategyType)
	deployment, err := f.ClientSet.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deploymentConf, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = e2edeployment.WaitForDeploymentComplete(f.ClientSet, deployment)
	framework.ExpectNoError(err)
	// Run a single watcher, to reduce the number of API calls we have to
	// make; this is to minimize the timing error. It's how kube-proxy
	// consumes the endpoints data, so it seems like the right thing to
	// test.
	endpointQueries := newQuerier()
	startEndpointWatcher(ctx, f, endpointQueries)
	defer close(endpointQueries.stop)

	// run one test and throw it away-- this is to make sure that the pod's
	// ready status has propagated.
	_, err = singleServiceLatency(ctx, f, name, endpointQueries)
	framework.ExpectNoError(err)

	// These channels are never closed, and each attempt sends on exactly
	// one of these channels, so the sum of the things sent over them will
	// be exactly total.
	errs := make(chan error, total)
	durations := make(chan time.Duration, total)

	blocker := make(chan struct{}, inParallel)
	for i := 0; i < total; i++ {
		go func() {
			defer ginkgo.GinkgoRecover()
			blocker <- struct{}{}
			defer func() { <-blocker }()
			if d, err := singleServiceLatency(ctx, f, name, endpointQueries); err != nil {
				errs <- err
			} else {
				durations <- d
			}
		}()
	}

	errCount := 0
	for i := 0; i < total; i++ {
		select {
		case e := <-errs:
			framework.Logf("Got error: %v", e)
			errCount++
		case d := <-durations:
			output = append(output, d)
		}
	}
	if errCount != 0 {
		framework.Logf("Got %d errors out of %d tries", errCount, total)
		errRatio := float32(errCount) / float32(total)
		if errRatio > acceptableFailureRatio {
			return output, fmt.Errorf("error ratio %g is higher than the acceptable ratio %g", errRatio, acceptableFailureRatio)
		}
	}
	return output, nil
}

type endpointQuery struct {
	endpointsName string
	endpoints     *v1.Endpoints
	result        chan<- struct{}
}

type endpointQueries struct {
	requests map[string]*endpointQuery

	stop        chan struct{}
	requestChan chan *endpointQuery
	seenChan    chan *v1.Endpoints
}

func newQuerier() *endpointQueries {
	eq := &endpointQueries{
		requests: map[string]*endpointQuery{},

		stop:        make(chan struct{}, 100),
		requestChan: make(chan *endpointQuery),
		seenChan:    make(chan *v1.Endpoints, 100),
	}
	go eq.join()
	return eq
}

// join merges the incoming streams of requests and added endpoints. It has
// nice properties like:
//   - remembering an endpoint if it happens to arrive before it is requested.
//   - closing all outstanding requests (returning nil) if it is stopped.
func (eq *endpointQueries) join() {
	defer func() {
		// Terminate all pending requests, so that no goroutine will
		// block indefinitely.
		for _, req := range eq.requests {
			if req.result != nil {
				close(req.result)
			}
		}
	}()

	for {
		select {
		case <-eq.stop:
			return
		case req := <-eq.requestChan:
			if cur, ok := eq.requests[req.endpointsName]; ok && cur.endpoints != nil {
				// We've already gotten the result, so we can
				// immediately satisfy this request.
				delete(eq.requests, req.endpointsName)
				req.endpoints = cur.endpoints
				close(req.result)
			} else {
				// Save this request.
				eq.requests[req.endpointsName] = req
			}
		case got := <-eq.seenChan:
			if req, ok := eq.requests[got.Name]; ok {
				if req.result != nil {
					// Satisfy a request.
					delete(eq.requests, got.Name)
					req.endpoints = got
					close(req.result)
				}
				// We've already recorded a result, but
				// haven't gotten the request yet. Only
				// keep the first result.
			} else {
				// We haven't gotten the corresponding request
				// yet, save this result.
				eq.requests[got.Name] = &endpointQuery{
					endpoints: got,
				}
			}
		}
	}
}

// request blocks until the requested endpoint is seen.
func (eq *endpointQueries) request(endpointsName string) *v1.Endpoints {
	result := make(chan struct{})
	req := &endpointQuery{
		endpointsName: endpointsName,
		result:        result,
	}
	eq.requestChan <- req
	<-result
	return req.endpoints
}

// marks e as added; does not block.
func (eq *endpointQueries) added(e *v1.Endpoints) {
	eq.seenChan <- e
}

// blocks until it has finished syncing.
func startEndpointWatcher(ctx context.Context, f *framework.Framework, q *endpointQueries) {
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				obj, err := f.ClientSet.CoreV1().Endpoints(f.Namespace.Name).List(ctx, options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return f.ClientSet.CoreV1().Endpoints(f.Namespace.Name).Watch(ctx, options)
			},
		},
		&v1.Endpoints{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				if e, ok := obj.(*v1.Endpoints); ok {
					if len(e.Subsets) > 0 && len(e.Subsets[0].Addresses) > 0 {
						q.added(e)
					}
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				if e, ok := cur.(*v1.Endpoints); ok {
					if len(e.Subsets) > 0 && len(e.Subsets[0].Addresses) > 0 {
						q.added(e)
					}
				}
			},
		},
	)

	go controller.Run(q.stop)

	// Wait for the controller to sync, so that we don't count any warm-up time.
	for !controller.HasSynced() {
		time.Sleep(100 * time.Millisecond)
	}
}

func singleServiceLatency(ctx context.Context, f *framework.Framework, name string, q *endpointQueries) (time.Duration, error) {
	// Make a service that points to that pod.
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "latency-svc-",
		},
		Spec: v1.ServiceSpec{
			Ports:           []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: 80}},
			Selector:        map[string]string{"name": name},
			Type:            v1.ServiceTypeClusterIP,
			SessionAffinity: v1.ServiceAffinityNone,
		},
	}
	startTime := time.Now()
	gotSvc, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
	if err != nil {
		return 0, err
	}
	framework.Logf("Created: %v", gotSvc.Name)

	if e := q.request(gotSvc.Name); e == nil {
		return 0, fmt.Errorf("never got a result for endpoint %v", gotSvc.Name)
	}
	stopTime := time.Now()
	d := stopTime.Sub(startTime)
	framework.Logf("Got endpoints: %v [%v]", gotSvc.Name, d)
	return d, nil
}
