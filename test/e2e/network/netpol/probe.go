/*
Copyright 2020 The Kubernetes Authors.

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

package netpol

import (
	"fmt"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	netutils "k8s.io/utils/net"
)

// decouple us from k8smanager.go
type Prober interface {
	probeConnectivity(args *probeConnectivityArgs) (bool, string, error)
}

// ProbeJob packages the data for the input of a pod->pod connectivity probe
type ProbeJob struct {
	PodFrom        TestPod
	PodTo          TestPod
	PodToServiceIP string
	ToPort         int
	ToPodDNSDomain string
	Protocol       v1.Protocol
}

// ProbeJobResults packages the data for the results of a pod->pod connectivity probe
type ProbeJobResults struct {
	Job         *ProbeJob
	IsConnected bool
	Err         error
	Command     string
}

// ProbePodToPodConnectivity runs a series of probes in kube, and records the results in `testCase.Reachability`
func ProbePodToPodConnectivity(prober Prober, allPods []TestPod, dnsDomain string, testCase *TestCase) {
	size := len(allPods) * len(allPods)
	jobs := make(chan *ProbeJob, size)
	results := make(chan *ProbeJobResults, size)
	for i := 0; i < getWorkers(); i++ {
		go probeWorker(prober, jobs, results, getProbeTimeoutSeconds())
	}
	for _, podFrom := range allPods {
		for _, podTo := range allPods {
			jobs <- &ProbeJob{
				PodFrom:        podFrom,
				PodTo:          podTo,
				ToPort:         testCase.ToPort,
				ToPodDNSDomain: dnsDomain,
				Protocol:       testCase.Protocol,
			}
		}
	}
	close(jobs)

	for i := 0; i < size; i++ {
		result := <-results
		job := result.Job
		if result.Err != nil {
			framework.Logf("unable to perform probe %s -> %s: %v", job.PodFrom.PodString(), job.PodTo.PodString(), result.Err)
		}
		testCase.Reachability.Observe(job.PodFrom.PodString(), job.PodTo.PodString(), result.IsConnected)
		expected := testCase.Reachability.Expected.Get(job.PodFrom.PodString().String(), job.PodTo.PodString().String())
		if result.IsConnected != expected {
			framework.Logf("Validation of %s -> %s FAILED !!!", job.PodFrom.PodString(), job.PodTo.PodString())
			framework.Logf("error %v ", result.Err)
			if expected {
				framework.Logf("Expected allowed pod connection was instead BLOCKED --- run '%v'", result.Command)
			} else {
				framework.Logf("Expected blocked pod connection was instead ALLOWED --- run '%v'", result.Command)
			}
		}
	}
}

// probeWorker continues polling a pod connectivity status, until the incoming "jobs" channel is closed, and writes results back out to the "results" channel.
// it only writes pass/fail status to a channel and has no failure side effects, this is by design since we do not want to fail inside a goroutine.
func probeWorker(prober Prober, jobs <-chan *ProbeJob, results chan<- *ProbeJobResults, timeoutSeconds int) {
	defer ginkgo.GinkgoRecover()
	for job := range jobs {
		podFrom := job.PodFrom
		// defensive programming: this should not be possible as we already check in initializeClusterFromModel
		if netutils.ParseIPSloppy(job.PodTo.ServiceIP) == nil {
			results <- &ProbeJobResults{
				Job:         job,
				IsConnected: false,
				Err:         fmt.Errorf("empty service ip"),
			}
		}
		// note that we can probe a dnsName instead of ServiceIP by using dnsName like so:
		// we stopped doing this because we wanted to support netpol testing in non dns enabled
		// clusters, but might re-enable it later.
		// dnsName := job.PodTo.QualifiedServiceAddress(job.ToPodDNSDomain)

		// TODO make this work on dual-stack clusters...
		connected, command, err := prober.probeConnectivity(&probeConnectivityArgs{
			nsFrom:         podFrom.Namespace,
			podFrom:        podFrom.Name,
			containerFrom:  podFrom.ContainerName,
			addrTo:         job.PodTo.ServiceIP,
			protocol:       job.Protocol,
			toPort:         job.ToPort,
			timeoutSeconds: timeoutSeconds,
		})
		result := &ProbeJobResults{
			Job:         job,
			IsConnected: connected,
			Err:         err,
			Command:     command,
		}
		results <- result
	}
}
