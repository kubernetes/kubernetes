// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"io/ioutil"
	"log"
	"net/rpc"
	"os"
	"testing"

	"github.com/coreos/etcd/tools/functional-tester/etcd-agent/client"
)

func init() {
	defaultAgent, err := newAgent(AgentConfig{EtcdPath: etcdPath, LogDir: "etcd.log"})
	if err != nil {
		log.Panic(err)
	}
	defaultAgent.serveRPC(":9027")
}

func TestRPCStart(t *testing.T) {
	c, err := rpc.DialHTTP("tcp", ":9027")
	if err != nil {
		t.Fatal(err)
	}

	dir, err := ioutil.TempDir(os.TempDir(), "etcd-agent")
	if err != nil {
		t.Fatal(err)
	}
	var pid int
	err = c.Call("Agent.RPCStart", []string{"--data-dir", dir}, &pid)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Call("Agent.RPCTerminate", struct{}{}, nil)

	_, err = os.FindProcess(pid)
	if err != nil {
		t.Errorf("unexpected error %v when find process %d", err, pid)
	}
}

func TestRPCRestart(t *testing.T) {
	c, err := rpc.DialHTTP("tcp", ":9027")
	if err != nil {
		t.Fatal(err)
	}

	dir, err := ioutil.TempDir(os.TempDir(), "etcd-agent")
	if err != nil {
		t.Fatal(err)
	}
	var pid int
	err = c.Call("Agent.RPCStart", []string{"--data-dir", dir}, &pid)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Call("Agent.RPCTerminate", struct{}{}, nil)

	err = c.Call("Agent.RPCStop", struct{}{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var npid int
	err = c.Call("Agent.RPCRestart", struct{}{}, &npid)
	if err != nil {
		t.Fatal(err)
	}

	if npid == pid {
		t.Errorf("pid = %v, want not equal to %d", npid, pid)
	}

	s, err := os.FindProcess(pid)
	if err != nil {
		t.Errorf("unexpected error %v when find process %d", err, pid)
	}
	_, err = s.Wait()
	if err == nil {
		t.Errorf("err = nil, want killed error")
	}
	_, err = os.FindProcess(npid)
	if err != nil {
		t.Errorf("unexpected error %v when find process %d", err, npid)
	}
}

func TestRPCTerminate(t *testing.T) {
	c, err := rpc.DialHTTP("tcp", ":9027")
	if err != nil {
		t.Fatal(err)
	}

	dir, err := ioutil.TempDir(os.TempDir(), "etcd-agent")
	if err != nil {
		t.Fatal(err)
	}
	var pid int
	err = c.Call("Agent.RPCStart", []string{"--data-dir", dir}, &pid)
	if err != nil {
		t.Fatal(err)
	}

	err = c.Call("Agent.RPCTerminate", struct{}{}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal(err)
	}
}

func TestRPCStatus(t *testing.T) {
	c, err := rpc.DialHTTP("tcp", ":9027")
	if err != nil {
		t.Fatal(err)
	}

	var s client.Status
	err = c.Call("Agent.RPCStatus", struct{}{}, &s)
	if err != nil {
		t.Fatal(err)
	}
	if s.State != stateTerminated {
		t.Errorf("state = %s, want %s", s.State, stateTerminated)
	}

	dir, err := ioutil.TempDir(os.TempDir(), "etcd-agent")
	if err != nil {
		t.Fatal(err)
	}
	var pid int
	err = c.Call("Agent.RPCStart", []string{"--data-dir", dir}, &pid)
	if err != nil {
		t.Fatal(err)
	}

	err = c.Call("Agent.RPCStatus", struct{}{}, &s)
	if err != nil {
		t.Fatal(err)
	}
	if s.State != stateStarted {
		t.Errorf("state = %s, want %s", s.State, stateStarted)
	}

	err = c.Call("Agent.RPCTerminate", struct{}{}, nil)
	if err != nil {
		t.Fatal(err)
	}
}
