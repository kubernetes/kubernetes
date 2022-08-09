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
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"strings"
)

// TestCase describes the data for a netpol test
type TestCase struct {
	ToPort       int
	Protocol     v1.Protocol
	Reachability *Reachability
}

// PodString represents a namespace 'x' + pod 'a' as "x/a".
type PodString string

// NewPodString instantiates a PodString from the given namespace and name.
func NewPodString(namespace string, podName string) PodString {
	return PodString(fmt.Sprintf("%s/%s", namespace, podName))
}

// String converts back to a string
func (pod PodString) String() string {
	return string(pod)
}

func (pod PodString) split() (string, string) {
	pieces := strings.Split(string(pod), "/")
	if len(pieces) != 2 {
		framework.Failf("expected ns/pod, found %+v", pieces)
	}
	return pieces[0], pieces[1]
}

// Namespace extracts the namespace
func (pod PodString) Namespace() string {
	ns, _ := pod.split()
	return ns
}

// PodName extracts the pod name
func (pod PodString) PodName() string {
	_, podName := pod.split()
	return podName
}

// Peer is used for matching pods by either or both of the pod's namespace and name.
type Peer struct {
	Namespace string
	Pod       string
}

// Matches checks whether the Peer matches the PodString:
// - an empty namespace means the namespace will always match
// - otherwise, the namespace must match the PodString's namespace
// - same goes for Pod: empty matches everything, otherwise must match exactly
func (p *Peer) Matches(pod PodString) bool {
	return (p.Namespace == "" || p.Namespace == pod.Namespace()) && (p.Pod == "" || p.Pod == pod.PodName())
}

// Reachability packages the data for a cluster-wide connectivity probe
type Reachability struct {
	Expected   *TruthTable
	Observed   *TruthTable
	PodStrings []PodString
}

// NewReachability instantiates a reachability
func NewReachability(podStrings []PodString, defaultExpectation bool) *Reachability {
	var podNames []string
	for _, podString := range podStrings {
		podNames = append(podNames, podString.String())
	}
	r := &Reachability{
		Expected:   NewTruthTableFromItems(podNames, &defaultExpectation),
		Observed:   NewTruthTableFromItems(podNames, nil),
		PodStrings: podStrings,
	}
	return r
}

// AllowLoopback expects all communication from a pod to itself to be allowed.
// In general, call it after setting up any other rules since loopback logic follows no policy.
func (r *Reachability) AllowLoopback() {
	for _, podString := range r.PodStrings {
		podName := podString.String()
		r.Expected.Set(podName, podName, true)
	}
}

// Expect sets the expected value for a single observation
func (r *Reachability) Expect(from PodString, to PodString, isConnected bool) {
	r.Expected.Set(string(from), string(to), isConnected)
}

// ExpectAllIngress defines that any traffic going into the pod will be allowed/denied (true/false)
func (r *Reachability) ExpectAllIngress(pod PodString, connected bool) {
	r.Expected.SetAllTo(string(pod), connected)
	if !connected {
		framework.Logf("Denying all traffic *to* %s", pod)
	}
}

// ExpectAllEgress defines that any traffic going out of the pod will be allowed/denied (true/false)
func (r *Reachability) ExpectAllEgress(pod PodString, connected bool) {
	r.Expected.SetAllFrom(string(pod), connected)
	if !connected {
		framework.Logf("Denying all traffic *from* %s", pod)
	}
}

// ExpectPeer sets expected values using Peer matchers
func (r *Reachability) ExpectPeer(from *Peer, to *Peer, connected bool) {
	for _, fromPod := range r.PodStrings {
		if from.Matches(fromPod) {
			for _, toPod := range r.PodStrings {
				if to.Matches(toPod) {
					r.Expected.Set(fromPod.String(), toPod.String(), connected)
				}
			}
		}
	}
}

// Observe records a single connectivity observation
func (r *Reachability) Observe(fromPod PodString, toPod PodString, isConnected bool) {
	r.Observed.Set(fromPod.String(), toPod.String(), isConnected)
}

// Summary produces a useful summary of expected and observed data
func (r *Reachability) Summary(ignoreLoopback bool) (trueObs int, falseObs int, ignoredObs int, comparison *TruthTable) {
	comparison = r.Expected.Compare(r.Observed)
	if !comparison.IsComplete() {
		framework.Failf("observations not complete!")
	}
	falseObs, trueObs, ignoredObs = 0, 0, 0
	for from, dict := range comparison.Values {
		for to, val := range dict {
			if ignoreLoopback && from == to {
				// Never fail on loopback, because its not yet defined.
				ignoredObs++
			} else if val {
				trueObs++
			} else {
				falseObs++
			}
		}
	}
	return
}

// PrintSummary prints the summary
func (r *Reachability) PrintSummary(printExpected bool, printObserved bool, printComparison bool) {
	right, wrong, ignored, comparison := r.Summary(ignoreLoopback)
	if ignored > 0 {
		framework.Logf("warning: this test doesn't take into consideration hairpin traffic, i.e. traffic whose source and destination is the same pod: %d cases ignored", ignored)
	}
	framework.Logf("reachability: correct:%v, incorrect:%v, result=%t\n\n", right, wrong, wrong == 0)
	if printExpected {
		framework.Logf("expected:\n\n%s\n\n\n", r.Expected.PrettyPrint(""))
	}
	if printObserved {
		framework.Logf("observed:\n\n%s\n\n\n", r.Observed.PrettyPrint(""))
	}
	if printComparison {
		framework.Logf("comparison:\n\n%s\n\n\n", comparison.PrettyPrint(""))
	}
}
