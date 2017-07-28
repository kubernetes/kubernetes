/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import "testing"

func TestEnsureStaticIP(t *testing.T) {
	fcas := NewFakeCloudAddressService()
	ipName := "some-static-ip"
	serviceName := ""
	region := "us-central1"

	// First ensure call
	ip, existed, err := ensureStaticIP(fcas, ipName, serviceName, region, "")
	if err != nil || existed || ip == "" {
		t.Fatalf(`ensureStaticIP(%v, %v, %v, %v, "") = %v, %v, %v; want valid ip, false, nil`, fcas, ipName, serviceName, region, ip, existed, err)
	}

	// Second ensure call
	var ipPrime string
	ipPrime, existed, err = ensureStaticIP(fcas, ipName, serviceName, region, ip)
	if err != nil || !existed || ip != ipPrime {
		t.Fatalf(`ensureStaticIP(%v, %v, %v, %v, %v) = %v, %v, %v; want %v, true, nil`, fcas, ipName, serviceName, region, ip, ipPrime, existed, err, ip)
	}
}
