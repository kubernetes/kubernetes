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

package preflight

import (
	"testing"
	"time"
)

type testTimer struct {}

func (testTimer) Sleep(time.Duration) {
	timerCallCount += 1
}

type testEtcdConnection struct {}

func (testEtcdConnection) checkConnection(connString string) bool {
	checkCallCount += 1
	return (checkCallCount >= (timesToFail + 1))
}

var checkCallCount int
var timesToFail int

var origTimer clock
var origRetryInterval time.Duration
var timerCallCount int
var origEtcdConnection connection

func setUp() {
	origTimer = timer
	origRetryInterval = retryInterval
	timer = new(testTimer)
	timerCallCount = 0
	checkCallCount = 0
	timesToFail = 0
	origEtcdConnection = etcdConnection
	etcdConnection = new(testEtcdConnection)
}

func tearDown() {
	timer = origTimer
	retryInterval = origRetryInterval
	etcdConnection = origEtcdConnection
}

func TestWaitForEtcdSuccess(t *testing.T) {
	setUp()
	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers)
	if checkCallCount != 1 { t.Fatal("Check Etcd was called an unexpected number of times") }
	if err != nil { t.Fatalf("Unexpected error: %v", err) }
	tearDown()
}

func TestWaitForEtcdFail(t *testing.T) {
	setUp()
	timesToFail = retryLimit + 5
	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers)
	if checkCallCount != retryLimit + 1 { t.Fatalf("Expected Etcd connection to be tested %d times. It was checked %d", retryLimit + 1, checkCallCount) }
	if err == nil { t.Fatal("Expected retry error to occur") }
	tearDown()
}

func TestBadConnectionString(t *testing.T) {
	setUp()
	servers := []string{"-invalid uri$@#%"}
	err := WaitForEtcd(servers)
	if err == nil { t.Fatal("Expected bad URI to raise parse error") }
	tearDown()
}

func TestDelayedStart(t *testing.T) {
	setUp()
	timesToFail = 2
	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers)
	if checkCallCount != 3 { t.Fatalf("Expected 3 calls to check Etcd. Got: %d", checkCallCount) }
	if err != nil { t.Fatalf("Unexpected error: %v", err) }
	tearDown()
}
