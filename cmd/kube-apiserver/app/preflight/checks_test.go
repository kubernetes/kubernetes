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

type testTimer struct {
	timerCallCount int
}

func (t testTimer) Sleep(time.Duration) {
	t.timerCallCount += 1
}

type mockEtcdConnection struct {
	shouldSucceed bool
}

func (m mockEtcdConnection) checkConnection(address string) bool{
	return m.shouldSucceed
}

func makeEtcdConnection(shouldSucceed bool) mockEtcdConnection {
	conn := mockEtcdConnection{}
	conn.shouldSucceed = shouldSucceed
	return conn
}

func setUp() func() {
	origTimer := timer
	origEtcdConnection := etcd
	return func() {
		timer = origTimer
		etcd = origEtcdConnection
	}
}

func TestWaitForEtcdSuccess(t *testing.T) {
	tearDown := setUp()
	defer tearDown()
	etcd = makeEtcdConnection(true)
	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers)

	if err != nil { t.Fatalf("Unexpected error: %v", err) }
}


func TestWaitForEtcdFail(t *testing.T) {
	tearDown := setUp()
	defer tearDown()
	timer = new(testTimer)

	etcd = makeEtcdConnection(false)
	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers)
	if err == nil { t.Fatal("Expected 'Unable to reach Etcd' error to occur") }
}

func TestCheckEtcdServer(t *testing.T) {
	tearDown := setUp()
	defer tearDown()
	timer = new(testTimer)

	mockEtcd := makeEtcdConnection(false)
	etcd = mockEtcd
	address := "https://127.0.0.1:2379"
	ch := make(chan struct{})
	checkEtcdServer(address, ch)
	select {
		case <-ch:
			{t.Error("received unexpected channel activity")}
		default:
	}
}

func testEtcdConnection(t *testing.T) {
	tearDown := setUp()
	defer tearDown()
	result := etcd.checkConnection("-not a real network address-")
	if result {t.Error("checkConnection should not have succeeded")}
}
