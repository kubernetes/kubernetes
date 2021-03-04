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

package iptables

import (
	"fmt"
	"net"

	"k8s.io/klog/v2"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnet "k8s.io/utils/net"
)

// LocalTrafficDetector in a interface to take action (jump) based on whether traffic originated locally
// at the node or not
type LocalTrafficDetector interface {
	// IsImplemented returns true if the implementation does something, false otherwise
	IsImplemented() bool

	// JumpIfLocal appends conditions to jump to a target chain if traffic detected to be
	// of local origin
	JumpIfLocal(args []string, toChain string) []string

	// JumpINotfLocal appends conditions to jump to a target chain if traffic detected not to be
	// of local origin
	JumpIfNotLocal(args []string, toChain string) []string
}

type noOpLocalDetector struct{}

// NewNoOpLocalDetector is a no-op implementation of LocalTrafficDetector
func NewNoOpLocalDetector() LocalTrafficDetector {
	return &noOpLocalDetector{}
}

func (n *noOpLocalDetector) IsImplemented() bool {
	return false
}

func (n *noOpLocalDetector) JumpIfLocal(args []string, toChain string) []string {
	return args // no-op
}

func (n *noOpLocalDetector) JumpIfNotLocal(args []string, toChain string) []string {
	return args // no-op
}

type detectLocalByCIDR struct {
	cidr string
}

// NewDetectLocalByCIDR implements the LocalTrafficDetector interface using a CIDR. This can be used when a single CIDR
// range can be used to capture the notion of local traffic.
func NewDetectLocalByCIDR(cidr string, ipt utiliptables.Interface) (LocalTrafficDetector, error) {
	if utilnet.IsIPv6CIDRString(cidr) != ipt.IsIPv6() {
		return nil, fmt.Errorf("CIDR %s has incorrect IP version: expect isIPv6=%t", cidr, ipt.IsIPv6())
	}
	_, _, err := net.ParseCIDR(cidr)
	if err != nil {
		return nil, err
	}
	return &detectLocalByCIDR{cidr: cidr}, nil
}

func (d *detectLocalByCIDR) IsImplemented() bool {
	return true
}

func (d *detectLocalByCIDR) JumpIfLocal(args []string, toChain string) []string {
	line := append(args, "-s", d.cidr, "-j", toChain)
	klog.V(4).Info("[DetectLocalByCIDR (", d.cidr, ")", " Jump Local: ", line)
	return line
}

func (d *detectLocalByCIDR) JumpIfNotLocal(args []string, toChain string) []string {
	line := append(args, "!", "-s", d.cidr, "-j", toChain)
	klog.V(4).Info("[DetectLocalByCIDR (", d.cidr, ")]", " Jump Not Local: ", line)
	return line
}
