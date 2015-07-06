/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package main

import (
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/iptables"
	"github.com/golang/glog"
	"math/rand"
	"strings"
	"sync/atomic"
	"time"
)

const (
	ChainFirewall iptables.Chain = "K8S-FIREWALL"
)

var letters = []rune("abcdefghijklmnopqrstuvwxyz")

var chainNumber uint64 = 0

// An injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type IptablesExecutor interface {
	CreateNewChainWithRandomName() (iptables.Chain, error)
	EnsureFirewallChain() (bool, error)
	PrependJumpRuleInFirewallChain(jumpTarget iptables.Chain) (bool, error)
	AppendDenyAllRule(chain iptables.Chain) (bool, error)
	AppendAllowLocalTrafficRule(chain iptables.Chain) error
	AppendAllowTrafficRule(chain iptables.Chain, comment string, sourceIPs []string, protocol string, ports []string) error
	AppendAllowTrafficByDestIPRule(chain iptables.Chain, comment string, sourceIPs []string, destIPAddress, protocol string, ports []string) error
	ExculsiveChainToFirewallChain(chain iptables.Chain) (bool, error)
	FlushFirewallChain() error
}

// runner implements Interface in terms of exec("iptables").
type runner struct {
	iptables iptables.Interface
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func NewIpTablesExecutor() IptablesExecutor {
	return &runner{
		iptables: iptables.New(exec.New(), iptables.ProtocolIpv4),
	}
}

func (runner *runner) CreateNewChainWithRandomName() (iptables.Chain, error) {
	atomic.AddUint64(&chainNumber, 1)
	newChainNumber := atomic.LoadUint64(&chainNumber)
	randomSuffix := randSeq(6)
	newChainName := iptables.Chain(fmt.Sprintf("K8S-FIREWALL-%d-%s", newChainNumber, randomSuffix))
	existed, err := runner.iptables.EnsureChain(iptables.TableFILTER, newChainName)
	if err != nil {
		return "", err
	} else if existed {
		return "", fmt.Errorf("Chain %s already exists. Retry.", newChainName)
	} else {
		return newChainName, nil
	}
}

func (runner *runner) EnsureFirewallChain() (bool, error) {
	chainExisted, err := runner.iptables.EnsureChain(iptables.TableFILTER, ChainFirewall)
	if err != nil {
		return false, err
	}
	ruleExisted, err := runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, iptables.ChainInput, "-j", string(ChainFirewall))
	if err != nil {
		return false, err
	}
	return chainExisted && ruleExisted, nil
}

func (runner *runner) PrependJumpRuleInFirewallChain(jumpTarget iptables.Chain) (bool, error) {
	return runner.iptables.EnsureRule(iptables.Prepend, iptables.TableFILTER, ChainFirewall, "-j", string(jumpTarget))
}

func (runner *runner) AppendDenyAllRule(chain iptables.Chain) (bool, error) {
	return runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-j", "DROP")
}

func (runner *runner) AppendAllowLocalTrafficRule(chain iptables.Chain) error {
	_, err := runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", "allow-outbound-traffic", "-m", "state", "--state", "ESTABLISHED,RELATED", "-j", "RETURN")
	if err != nil {
		return err
	}
	_, err = runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", "allow-local-src-addreses", "-m", "addrtype", "--src-type", "LOCAL", "-j", "RETURN")
	if err != nil {
		return err
	}

	localCIDRs := []string{"10.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.168.0.0/16"}
	for _, localCIDR := range localCIDRs {
		comment := fmt.Sprintf("allow-%s-addresses", localCIDR)
		_, err = runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", comment, "-s", localCIDR, "-j", "RETURN")
		if err != nil {
			return err
		}
	}

	return nil
}

func (runner *runner) AppendAllowTrafficRule(chain iptables.Chain, comment string, sourceIPs []string, protocol string, ports []string) error {
	if ports == nil || len(ports) == 0 {
		for _, sourceIP := range sourceIPs {
			_, err := runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", comment, "-p", protocol, "-s", sourceIP, "-j", "RETURN")
			if err != nil {
				return err
			}
		}
		return nil
	}

	for _, port := range ports {
		port = strings.Replace(port, "-", ":", 1)
		for _, sourceIP := range sourceIPs {
			_, err := runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", comment, "-p", protocol, "-s", sourceIP, "-m", protocol, "--dport", port, "-j", "RETURN")
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (runner *runner) AppendAllowTrafficByDestIPRule(chain iptables.Chain, comment string, sourceIPs []string, destIPAddress, protocol string, ports []string) error {
	if ports == nil || len(ports) == 0 {
		for _, sourceIP := range sourceIPs {
			_, err := runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", comment, "-p", protocol, "-s", sourceIP, "-m", protocol, "-d", "destIPAddress", "-j", "RETURN")
			if err != nil {
				return err
			}
		}
		return nil
	}
	for _, port := range ports {
		port = strings.Replace(port, "-", ":", 1)
		for _, sourceIP := range sourceIPs {
			_, err := runner.iptables.EnsureRule(iptables.Append, iptables.TableFILTER, chain, "-m", "comment", "--comment", comment, "-p", protocol, "-s", sourceIP, "-d", destIPAddress, "-m", protocol, "--dport", port, "-j", "RETURN")
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (runner *runner) FlushFirewallChain() error {
	existed, err := runner.EnsureFirewallChain()
	if err != nil {
		return err
	}
	if !existed {
		glog.Infof("Firewall chain does not exist")
		return nil
	} else {
		runner.removeAllFirewallSubChains()
		return runner.iptables.FlushChain(iptables.TableFILTER, ChainFirewall)
	}
}

func (runner *runner) ExculsiveChainToFirewallChain(chain iptables.Chain) (bool, error) {
	targets, targetsFetchError := runner.extractTargetsFromFirewallChain()
	glog.Infof("Prepending %s chain to firewall chain", chain)
	_, prependError := runner.iptables.EnsureRule(iptables.Prepend, iptables.TableFILTER, ChainFirewall, "-j", string(chain))
	if prependError != nil {
		return false, prependError
	}

	if targetsFetchError != nil {
		return true, targetsFetchError
	}
	glog.Infof("Removing SubChains within Firewall Chain")
	for i := len(targets) - 1; i >= 0; i-- {
		glog.Infof("Removing %s chain and all references to it", targets[i])
		runner.removeFirewallSubChain(targets[i], i+2)
	}
	return true, nil
}

func (runner *runner) removeAllFirewallSubChains() {
	targets, err := runner.extractTargetsFromFirewallChain()
	if err != nil {
		glog.Errorf("Could not extract targets from firewall chain")
		return
	}
	for i := len(targets) - 1; i >= 0; i-- {
		runner.removeFirewallSubChain(targets[i], i+1)
	}
	return
}

func (runner *runner) removeFirewallSubChain(chainName string, ruleNumber int) (bool, error) {
	chain := iptables.Chain(chainName)
	err := runner.iptables.DeleteRuleNumber(iptables.TableFILTER, ChainFirewall, ruleNumber)
	removed := false
	if err != nil {
		glog.Warningf("Could not remove chain:'%s' from firewall chain, err:%v", chain, err)
	} else {
		glog.Infof("Removed chain:%s from firewall chain", chain)
		removed = true
	}
	err = runner.iptables.FlushChain(iptables.TableFILTER, chain)
	if err != nil {
		glog.Warningf("Could not flush chain:%s, err:%v", chain, err)
		return removed, err
	}
	glog.Infof("Flushed unused chain:%s", chain)
	err = runner.iptables.DeleteChain(iptables.TableFILTER, chain)
	if err != nil {
		glog.Warningf("Could not delete chain:%s, err:%v", chain, err)
		return removed, err
	}
	glog.Infof("Deleted chain:%s", chain)
	return removed, nil
}

func (runner *runner) extractTargetsFromFirewallChain() ([]string, error) {
	output, err := runner.iptables.ListChain(iptables.TableFILTER, ChainFirewall)
	if err != nil {
		return nil, err
	}
	list := strings.Split(output, "\n")
	retval := []string{}
	found := false
	for i := 0; i < len(list); i++ {
		if !found {
			if strings.Contains(list[i], "target") {
				found = true
			}
		} else {
			columns := strings.Fields(list[i])
			if len(columns) > 0 {
				glog.Infof("Adding target '%s'", columns[0])
				chainName := columns[0]
				retval = append(retval, chainName)
			}
		}
	}
	return retval, nil
}

func randSeq(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}
