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
	"errors"
	"testing"
)

type mockEtcdConnection struct {
	shouldSucceed bool
}

func (m mockEtcdConnection) serverReachable(address string) bool {
	return m.shouldSucceed
}

func (m mockEtcdConnection) parseServerList(serverList []string) error {
	return nil
}

func (m mockEtcdConnection) checkEtcdServers() (bool, error) {
	if !m.shouldSucceed {
		return false, errors.New("artificial error for testing")
	}
	return true, nil
}

func makeEtcdConnection(shouldSucceed bool) mockEtcdConnection {
	conn := mockEtcdConnection{}
	conn.shouldSucceed = shouldSucceed
	return conn
}


func TestParseServerListGood(t *testing.T) {
	etcd := new(etcdConnection)
	servers := []string{"https://127.0.0.1:2379"}
	err := etcd.parseServerList(servers)
	if err != nil { t.Fatalf("unexpected error: %v", err) }
	host := "127.0.0.1:2379"
	if etcd.hosts[0] != host {t.Fatal("server uri was not parsed correctly")}
}

func TestParseServerListBad(t *testing.T) {
	etcd := new(etcdConnection)
	servers := []string{"-invalid uri$@#%"}
	err := etcd.parseServerList(servers)
	if err == nil { t.Fatal("expected bad uri to raise parse error") }
}

func TestEtcdConnection(t *testing.T) {
	etcd := new(etcdConnection)

	result := etcd.serverReachable("-not a real network address-")
	if result {t.Fatal("checkConnection should not have succeeded")}
}

func TestCheckEtcdServersEmpty(t *testing.T) {
	etcd := new(etcdConnection)
	result, err := etcd.checkEtcdServers()
	if err != nil { t.Fatalf("unexpected error: %v", err) }
	if result {t.Fatal("checkEtcdServers should not have succeeded")}
}

func TestCheckEtcdServers(t *testing.T) {
	etcd := new(etcdConnection)
	etcd.hosts = []string{"-invalid uri$@#%"}
	result, err := etcd.checkEtcdServers()
	if err != nil { t.Fatalf("unexpected error: %v", err) }
	if result {t.Fatal("checkEtcdServers should not have succeeded")}
}


func TestWaitForEtcdSuccess(t *testing.T) {
	etcd := makeEtcdConnection(true)
	err := waitForAvailableEtcd(etcd)
	if err != nil { t.Fatalf("unexpected error: %v", err) }
}

func TestWaitForEtcdFail(t *testing.T) {
	etcd := makeEtcdConnection(false)
	err := waitForAvailableEtcd(etcd)
	if err == nil { t.Fatal("expected 'unable to reach etcd' error to occur") }
}
