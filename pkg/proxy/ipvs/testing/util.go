//go:build linux

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

package testing

import (
	utilipset "k8s.io/kubernetes/pkg/proxy/ipvs/ipset"
)

// ExpectedVirtualServer is the expected ipvs rules with VirtualServer and RealServer
// VSNum is the expected ipvs virtual server number
// IP:Port protocol is the expected ipvs vs info
// RS is the RealServer of this expected VirtualServer
type ExpectedVirtualServer struct {
	VSNum    int
	IP       string
	Port     uint16
	Protocol string
	RS       []ExpectedRealServer
}

// ExpectedRealServer is the expected ipvs RealServer
type ExpectedRealServer struct {
	IP   string
	Port uint16
}

// ExpectedIptablesChain is a map of expected iptables chain and jump rules
type ExpectedIptablesChain map[string][]ExpectedIptablesRule

// ExpectedIptablesRule is the expected iptables rules with jump chain and match ipset name
type ExpectedIptablesRule struct {
	SourceAddress string
	JumpChain     string
	MatchSet      string
}

// ExpectedIPSet is the expected ipset with set name and entries name
type ExpectedIPSet map[string][]*utilipset.Entry
