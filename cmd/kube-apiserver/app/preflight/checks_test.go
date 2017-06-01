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

	utilwait "k8s.io/apimachinery/pkg/util/wait"
)

func TestParseServerURIGood(t *testing.T) {
	host, err := parseServerURI("https://127.0.0.1:2379")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	reference := "127.0.0.1:2379"
	if host != reference {
		t.Fatal("server uri was not parsed correctly")
	}
}

func TestParseServerURIBad(t *testing.T) {
	_, err := parseServerURI("-invalid uri$@#%")
	if err == nil {
		t.Fatal("expected bad uri to raise parse error")
	}
}

func TestEtcdConnection(t *testing.T) {
	etcd := new(EtcdConnection)

	result := etcd.serverReachable("-not a real network address-")
	if result {
		t.Fatal("checkConnection should not have succeeded")
	}
}

func TestCheckEtcdServersEmpty(t *testing.T) {
	etcd := new(EtcdConnection)
	result, err := etcd.CheckEtcdServers()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result {
		t.Fatal("CheckEtcdServers should not have succeeded")
	}
}

func TestCheckEtcdServersUri(t *testing.T) {
	etcd := new(EtcdConnection)
	etcd.ServerList = []string{"-invalid uri$@#%"}
	result, err := etcd.CheckEtcdServers()
	if err == nil {
		t.Fatalf("expected bad uri to raise parse error")
	}
	if result {
		t.Fatal("CheckEtcdServers should not have succeeded")
	}
}

func TestCheckEtcdServers(t *testing.T) {
	etcd := new(EtcdConnection)
	etcd.ServerList = []string{""}
	result, err := etcd.CheckEtcdServers()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result {
		t.Fatal("CheckEtcdServers should not have succeeded")
	}
}

func TestPollCheckServer(t *testing.T) {
	err := utilwait.PollImmediate(1*time.Microsecond,
		2*time.Microsecond,
		EtcdConnection{ServerList: []string{""}}.CheckEtcdServers)
	if err == nil {
		t.Fatal("expected check to time out")
	}
}
