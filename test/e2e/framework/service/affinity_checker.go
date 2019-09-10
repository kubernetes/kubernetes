/*
Copyright 2019 The Kubernetes Authors.

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

package service

import (
	"fmt"
	"net"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

// CheckAffinity function tests whether the service affinity works as expected.
// If affinity is expected, the test will return true once affinityConfirmCount
// number of same response observed in a row. If affinity is not expected, the
// test will keep observe until different responses observed. The function will
// return false only in case of unexpected errors.
func CheckAffinity(execPod *v1.Pod, serviceIP string, servicePort int, shouldHold bool) bool {
	serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	cmd := fmt.Sprintf(`curl -q -s --connect-timeout 2 http://%s/`, serviceIPPort)
	timeout := TestTimeout
	if execPod == nil {
		timeout = LoadBalancerPollTimeout
	}
	var tracker affinityTracker
	if pollErr := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		if execPod != nil {
			stdout, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
			if err != nil {
				framework.Logf("Failed to get response from %s. Retry until timeout", serviceIPPort)
				return false, nil
			}
			tracker.recordHost(stdout)
		} else {
			rawResponse := GetHTTPContent(serviceIP, servicePort, timeout, "")
			tracker.recordHost(rawResponse.String())
		}
		trackerFulfilled, affinityHolds := tracker.checkHostTrace(AffinityConfirmCount)
		if !shouldHold && !affinityHolds {
			return true, nil
		}
		if shouldHold && trackerFulfilled && affinityHolds {
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		trackerFulfilled, _ := tracker.checkHostTrace(AffinityConfirmCount)
		if pollErr != wait.ErrWaitTimeout {
			checkAffinityFailed(tracker, pollErr.Error())
			return false
		}
		if !trackerFulfilled {
			checkAffinityFailed(tracker, fmt.Sprintf("Connection to %s timed out or not enough responses.", serviceIPPort))
		}
		if shouldHold {
			checkAffinityFailed(tracker, "Affinity should hold but didn't.")
		} else {
			checkAffinityFailed(tracker, "Affinity shouldn't hold but did.")
		}
		return true
	}
	return true
}

// affinityTracker tracks the destination of a request for the affinity tests.
type affinityTracker struct {
	hostTrace []string
}

// Record the response going to a given host.
func (at *affinityTracker) recordHost(host string) {
	at.hostTrace = append(at.hostTrace, host)
	framework.Logf("Received response from host: %s", host)
}

// Check that we got a constant count requests going to the same host.
func (at *affinityTracker) checkHostTrace(count int) (fulfilled, affinityHolds bool) {
	fulfilled = (len(at.hostTrace) >= count)
	if len(at.hostTrace) == 0 {
		return fulfilled, true
	}
	last := at.hostTrace[0:]
	if len(at.hostTrace)-count >= 0 {
		last = at.hostTrace[len(at.hostTrace)-count:]
	}
	host := at.hostTrace[len(at.hostTrace)-1]
	for _, h := range last {
		if h != host {
			return fulfilled, false
		}
	}
	return fulfilled, true
}

func checkAffinityFailed(tracker affinityTracker, err string) {
	framework.Logf("%v", tracker.hostTrace)
	framework.Failf(err)
}
