/*
Copyright 2016 The Kubernetes Authors.

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

package healthcheck

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

type TestCaseData struct {
	nodePorts    int
	numEndpoints int
	nodePortList []int
	svcNames     []types.NamespacedName
}

const (
	startPort = 20000
	endPort   = 40000
)

var (
	choices = []byte("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
)

func generateRandomString(n int) string {

	b := make([]byte, n)
	l := len(choices)
	for i := range b {
		b[i] = choices[rand.Intn(l)]
	}
	return string(b)
}

func chooseServiceName(tc int, hint int) types.NamespacedName {
	var svc types.NamespacedName
	svc.Namespace = fmt.Sprintf("ns_%d", tc)
	svc.Name = fmt.Sprintf("name_%d", hint)
	return svc
}

func generateEndpointSet(max int) sets.String {
	s := sets.NewString()
	for i := 0; i < max; i++ {
		s.Insert(fmt.Sprintf("%d%s", i, generateRandomString(8)))
	}
	return s
}

func verifyHealthChecks(tc *TestCaseData, t *testing.T) bool {
	var success = true
	time.Sleep(100 * time.Millisecond)
	for i := 0; i < tc.nodePorts; i++ {
		t.Logf("Validating HealthCheck works for svc %s nodePort %d\n", tc.svcNames[i], tc.nodePortList[i])
		res, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/", tc.nodePortList[i]))
		if err != nil {
			t.Logf("ERROR: Failed to connect to listening port")
			success = false
			continue
		}
		robots, err := ioutil.ReadAll(res.Body)
		if res.StatusCode == http.StatusServiceUnavailable {
			t.Logf("ERROR: HealthCheck returned %s: %s", res.Status, string(robots))
			success = false
			continue
		}
		res.Body.Close()
		if err != nil {
			t.Logf("Error: reading body of response (%s)", err)
			success = false
			continue
		}
	}
	if success {
		t.Logf("Success: All nodePorts found active")
	}
	return success
}

func TestHealthChecker(t *testing.T) {
	testcases := []TestCaseData{
		{
			nodePorts:    1,
			numEndpoints: 2,
		},
		{
			nodePorts:    10,
			numEndpoints: 6,
		},
		{
			nodePorts:    100,
			numEndpoints: 1,
		},
	}

	Run()

	ports := startPort
	for n, tc := range testcases {
		tc.nodePortList = make([]int, tc.nodePorts)
		tc.svcNames = make([]types.NamespacedName, tc.nodePorts)
		for i := 0; i < tc.nodePorts; i++ {
			tc.svcNames[i] = chooseServiceName(n, i)
			t.Logf("Updating endpoints map for %s %d", tc.svcNames[i], tc.numEndpoints)
			for {
				UpdateEndpoints(tc.svcNames[i], generateEndpointSet(tc.numEndpoints))
				tc.nodePortList[i] = ports
				ports++
				if AddServiceListener(tc.svcNames[i], tc.nodePortList[i]) {
					break
				}
				DeleteServiceListener(tc.svcNames[i], tc.nodePortList[i])
				// Keep searching for a port that works
				t.Logf("Failed to bind/listen on port %d...trying next port", ports-1)
				if ports > endPort {
					t.Errorf("Exhausted range of ports available for tests")
					return
				}
			}
		}
		t.Logf("Validating if all nodePorts for tc %d work", n)
		if !verifyHealthChecks(&tc, t) {
			t.Errorf("Healthcheck validation failed")
		}

		for i := 0; i < tc.nodePorts; i++ {
			DeleteServiceListener(tc.svcNames[i], tc.nodePortList[i])
			UpdateEndpoints(tc.svcNames[i], sets.NewString())
		}

		// Ensure that all listeners have been shutdown
		if verifyHealthChecks(&tc, t) {
			t.Errorf("Healthcheck validation failed")
		}
	}
}
