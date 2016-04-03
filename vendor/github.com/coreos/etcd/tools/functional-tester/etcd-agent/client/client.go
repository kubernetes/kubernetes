// Copyright 2015 CoreOS, Inc.
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

package client

import "net/rpc"

type Status struct {
	// TODO: gather more informations
	// TODO: memory usage, raft information, etc..
	State string
}

type Agent interface {
	ID() uint64
	// Start starts a new etcd with the given args on the agent machine.
	Start(args ...string) (int, error)
	// Stop stops the existing etcd the agent started.
	Stop() error
	// Restart restarts the existing etcd the agent stopped.
	Restart() (int, error)
	// Cleanup stops the exiting etcd the agent started, then archives log and its data dir.
	Cleanup() error
	// Terminate stops the exiting etcd the agent started and removes its data dir.
	Terminate() error
	// DropPort drops all network packets at the given port.
	DropPort(port int) error
	// RecoverPort stops dropping all network packets at the given port.
	RecoverPort(port int) error
	// Status returns the status of etcd on the agent
	Status() (Status, error)
}

type agent struct {
	endpoint  string
	rpcClient *rpc.Client
}

func NewAgent(endpoint string) (Agent, error) {
	c, err := rpc.DialHTTP("tcp", endpoint)
	if err != nil {
		return nil, err
	}
	return &agent{endpoint, c}, nil
}

func (a *agent) Start(args ...string) (int, error) {
	var pid int
	err := a.rpcClient.Call("Agent.RPCStart", args, &pid)
	if err != nil {
		return -1, err
	}
	return pid, nil
}

func (a *agent) Stop() error {
	return a.rpcClient.Call("Agent.RPCStop", struct{}{}, nil)
}

func (a *agent) Restart() (int, error) {
	var pid int
	err := a.rpcClient.Call("Agent.RPCRestart", struct{}{}, &pid)
	if err != nil {
		return -1, err
	}
	return pid, nil
}

func (a *agent) Cleanup() error {
	return a.rpcClient.Call("Agent.RPCCleanup", struct{}{}, nil)
}

func (a *agent) Terminate() error {
	return a.rpcClient.Call("Agent.RPCTerminate", struct{}{}, nil)
}

func (a *agent) DropPort(port int) error {
	return a.rpcClient.Call("Agent.RPCDropPort", port, nil)
}

func (a *agent) RecoverPort(port int) error {
	return a.rpcClient.Call("Agent.RPCRecoverPort", port, nil)
}

func (a *agent) Status() (Status, error) {
	var s Status
	err := a.rpcClient.Call("Agent.RPCStatus", struct{}{}, &s)
	if err != nil {
		return s, err
	}
	return s, nil
}

func (a *agent) ID() uint64 {
	panic("not implemented")
}
