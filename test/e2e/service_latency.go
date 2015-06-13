/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
)

type durations []time.Duration

func (d durations) Len() int           { return len(d) }
func (d durations) Less(i, j int) bool { return d[i] < d[j] }
func (d durations) Swap(i, j int)      { d[i], d[j] = d[j], d[i] }

var _ = Describe("Service endpoints latency", func() {
	f := NewFramework("svc-latency")

	It("should not be very high", func() {
		nodes, err := f.Client.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			Failf("Failed to list nodes: %v", err)
		}
		count := len(nodes.Items)

		// Numbers chosen to make the test complete in a short amount
		// of time. This sample size is not actually large enough to
		// reliably measure tails on a reasonably sized test cluster,
		// but it should catch low hanging fruit.
		var (
			totalTrials    = 20 * count
			parallelTrials = 8 * count
			minSampleSize  = 10 * count
		)

		failing := util.NewStringSet()
		d, err := runServiceLatencies(f, parallelTrials, totalTrials)
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
			Fail(strings.Join(failing.List(), "\n"))
		}
		percentile := func(p int) time.Duration {
			est := n * p / 100
			if est >= n {
				return dSorted[n-1]
			}
			return dSorted[est]
		}
		Logf("Latencies: %v", dSorted)
		p50 := percentile(50)
		p90 := percentile(90)
		p99 := percentile(99)
		Logf("50 %%ile: %v", p50)
		Logf("90 %%ile: %v", p90)
		Logf("99 %%ile: %v", p99)

		if p99 > 4*p50 {
			failing.Insert("Tail latency is > 4x median latency")
		}

		if p50 > time.Second*20 {
			failing.Insert("Median latency should be less than 20 seconds")
		}
		if failing.Len() > 0 {
			Fail(strings.Join(failing.List(), "\n"))
		}
	})
})

func runServiceLatencies(f *Framework, inParallel, total int) (output []time.Duration, err error) {
	next := make(chan int, total)
	go func() {
		for i := 0; i < total; i++ {
			next <- i
		}
		close(next)
	}()

	errs := make(chan error, total)
	durations := make(chan time.Duration, total)

	for i := 0; i < inParallel; i++ {
		go func() {
			defer GinkgoRecover()
			for {
				i, ok := <-next
				if !ok {
					return
				}
				if d, err := singleServiceLatency(f, i); err != nil {
					errs <- err
				} else {
					durations <- d
				}
			}
		}()
	}

	errCount := 0
	for i := 0; i < total; i++ {
		select {
		case e := <-errs:
			Logf("Got error: %v", e)
			errCount += 1
		case d := <-durations:
			output = append(output, d)
		}
	}
	if errCount != 0 {
		return output, fmt.Errorf("got %v errors", errCount)
	}
	return output, nil
}

func singleServiceLatency(f *Framework, i int) (time.Duration, error) {
	// Make an RC with a single pod.
	cfg := RCConfig{
		Client:       f.Client,
		Image:        "gcr.io/google_containers/pause:1.0",
		Name:         fmt.Sprintf("trial-%v", i),
		Namespace:    f.Namespace.Name,
		Replicas:     1,
		PollInterval: time.Second,
	}
	if err := RunRC(cfg); err != nil {
		return 0, err
	}
	defer DeleteRC(f.Client, f.Namespace.Name, cfg.Name)

	// Now make a service that points to that pod.
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: cfg.Name,
		},
		Spec: api.ServiceSpec{
			Ports:           []api.ServicePort{{Protocol: api.ProtocolTCP, Port: 80}},
			Selector:        map[string]string{"name": cfg.Name},
			Type:            api.ServiceTypeClusterIP,
			SessionAffinity: api.ServiceAffinityNone,
		},
	}
	gotSvc, err := f.Client.Services(f.Namespace.Name).Create(svc)
	if err != nil {
		return 0, err
	}

	// Now time how long it takes for the endpoints to show up.
	startTime := time.Now()
	defer f.Client.Services(f.Namespace.Name).Delete(gotSvc.Name)
	w, err := f.Client.Endpoints(f.Namespace.Name).Watch(labels.Everything(), fields.Set{"metadata.name": cfg.Name}.AsSelector(), gotSvc.ResourceVersion)
	if err != nil {
		return 0, err
	}
	defer w.Stop()

	for {
		val, ok := <-w.ResultChan()
		if !ok {
			return 0, fmt.Errorf("watch closed")
		}
		if e, ok := val.Object.(*api.Endpoints); ok {
			if e.Name == cfg.Name && len(e.Subsets) > 0 && len(e.Subsets[0].Addresses) > 0 {
				stopTime := time.Now()
				return stopTime.Sub(startTime), nil
			}
		}
	}

}
