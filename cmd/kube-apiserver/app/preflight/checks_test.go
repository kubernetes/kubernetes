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

func (t testTimer) After(time.Duration) <-chan time.Time {
	ch := make(chan time.Time, 1)
	ch <- time.Now()
	return ch
}

type mockEtcdConnection struct {
	shouldSucceed bool
}

func (m mockEtcdConnection) serverReachable(address string) bool{
	return m.shouldSucceed
}

func makeEtcdConnection(shouldSucceed bool) mockEtcdConnection {
	conn := mockEtcdConnection{}
	conn.shouldSucceed = shouldSucceed
	return conn
}

func TestWaitForEtcdSuccess(t *testing.T) {
	timer := new(connectionTimer)
	etcd := makeEtcdConnection(true)
	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers, timer, etcd)

	if err != nil { t.Fatalf("Unexpected error: %v", err) }
}


func TestWaitForEtcdFail(t *testing.T) {
	timer := new(testTimer)
	etcd := makeEtcdConnection(false)

	servers := []string{"https://127.0.0.1:2379"}
	err := WaitForEtcd(servers, timer, etcd)
	if err == nil { t.Fatal("Expected 'Unable to reach Etcd' error to occur") }
}

func TestCheckEtcdServerExpired(t *testing.T) {
	timer := new(testTimer)
	etcd := makeEtcdConnection(false)

	address := "https://127.0.0.1:2379"
	ch := make(chan struct{})
	stop := make(chan struct{})
	defer close(stop)
	checkEtcdServer(address, ch, stop, timer, etcd)
	select {
		case <-ch:
			{t.Fatal("received unexpected channel activity")}
		default:
	}
}

func TestCheckEtcdServerStop(t *testing.T) {
	// use a timer that really sleeps
	timer := new(connectionTimer)
	etcd := makeEtcdConnection(false)

	address := "https://127.0.0.1:2379"
	ch := make(chan struct{})
	stop := make(chan struct{})
	// close the channel before its even sent
	close(stop)
	checkEtcdServer(address, ch, stop, timer, etcd)
	select {
	case <-ch:
		{t.Fatal("received unexpected channel activity")}
	default:
	}
}

func testEtcdConnection(t *testing.T) {
	etcd := new(etcdConnection)

	result := etcd.serverReachable("-not a real network address-")
	if result {t.Fatal("checkConnection should not have succeeded")}
}

func TestBadConnectionString(t *testing.T) {
	timer := new(testTimer)
	etcd := makeEtcdConnection(false)
	servers := []string{"-invalid uri$@#%"}
	err := WaitForEtcd(servers, timer, etcd)
	if err == nil { t.Fatal("Expected bad URI to raise parse error") }
}
