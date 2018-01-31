/*
Copyright 2018 The Kubernetes Authors.

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

package etcd

import (
	"math"
	"testing"
	"testing/quick"
)

func TestFeistelObfuscatorSymmetric(t *testing.T) {
	o := NewFeistelObfuscator()
	checkRoundTrip := func(key string, etcdResourceVersion uint64) bool {
		clientResourceVersion := o.Encode(key, etcdResourceVersion)
		roundTripResourceVersion := o.Decode(key, clientResourceVersion)
		return roundTripResourceVersion == etcdResourceVersion
	}
	testCases := []struct {
		Key                 string
		EtcdResourceVersion uint64
	}{
		{Key: "aaaa", EtcdResourceVersion: 0},
		{Key: "bbb", EtcdResourceVersion: 1},
		{Key: "cc", EtcdResourceVersion: math.MaxUint64},
		{Key: "d", EtcdResourceVersion: math.MaxUint64 - 1},
		{Key: "", EtcdResourceVersion: 1000000},
	}
	for _, testCase := range testCases {
		if !checkRoundTrip(testCase.Key, testCase.EtcdResourceVersion) {
			t.Errorf("%q %v: version returned after a round trip conversion was different",
				testCase.Key,
				testCase.EtcdResourceVersion,
			)
		}
	}
	if err := quick.Check(checkRoundTrip, nil); err != nil {
		t.Error(err)
	}
}

func TestFeistelObfuscatorFixedAtZero(t *testing.T) {
	o := NewFeistelObfuscator()
	checkZeroFixed := func(key string) bool {
		clientResourceVersion := o.Encode(key, 0)
		return clientResourceVersion == 0
	}
	if err := quick.Check(checkZeroFixed, nil); err != nil {
		t.Error(err)
	}
}
