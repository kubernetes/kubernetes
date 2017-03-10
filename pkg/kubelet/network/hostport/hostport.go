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

package hostport

import (
	"bytes"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/v1"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

const (
	// the hostport chain
	kubeHostportsChain utiliptables.Chain = "KUBE-HOSTPORTS"
	// the "is a local destination" chain, which pre-filters hostports
	kubeLocalDestChain utiliptables.Chain = "KUBE-LOCALDEST"
	// prefix for hostport chains
	kubeHostportChainPrefix string = "KUBE-HP-"
)

// PortMapping represents a network port in a container
type PortMapping struct {
	Name          string
	HostPort      int32
	ContainerPort int32
	Protocol      v1.Protocol
	HostIP        string
}

// PodPortMapping represents a pod's network state and associated container port mappings
type PodPortMapping struct {
	Namespace    string
	Name         string
	PortMappings []*PortMapping
	HostNetwork  bool
	IP           net.IP
}

type hostport struct {
	port     int32
	protocol string
}

type hostportOpener func(*hostport) (closeable, error)

type closeable interface {
	Close() error
}

func openLocalPort(hp *hostport) (closeable, error) {
	// For ports on node IPs, open the actual port and hold it, even though we
	// use iptables to redirect traffic.
	// This ensures a) that it's safe to use that port and b) that (a) stays
	// true.  The risk is that some process on the node (e.g. sshd or kubelet)
	// is using a port and we give that same port out to a Service.  That would
	// be bad because iptables would silently claim the traffic but the process
	// would never know.
	// NOTE: We should not need to have a real listen()ing socket - bind()
	// should be enough, but I can't figure out a way to e2e test without
	// it.  Tools like 'ss' and 'netstat' do not show sockets that are
	// bind()ed but not listen()ed, and at least the default debian netcat
	// has no way to avoid about 10 seconds of retries.
	var socket closeable
	switch hp.protocol {
	case "tcp":
		listener, err := net.Listen("tcp", fmt.Sprintf(":%d", hp.port))
		if err != nil {
			return nil, err
		}
		socket = listener
	case "udp":
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf(":%d", hp.port))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", hp.protocol)
	}
	glog.V(3).Infof("Opened local port %s", hp.String())
	return socket, nil
}

// openHostports opens all given hostports using the given hostportOpener
// If encounter any error, clean up and return the error
// If all ports are opened successfully, return the hostport and socket mapping
// TODO: move openHostports and closeHostports into a common struct
func openHostports(portOpener hostportOpener, podPortMapping *PodPortMapping) (map[hostport]closeable, error) {
	var retErr error
	ports := make(map[hostport]closeable)
	for _, pm := range podPortMapping.PortMappings {
		if pm.HostPort <= 0 {
			continue
		}
		hp := portMappingToHostport(pm)
		socket, err := portOpener(&hp)
		if err != nil {
			retErr = fmt.Errorf("cannot open hostport %d for pod %s: %v", pm.HostPort, getPodFullName(podPortMapping), err)
			break
		}
		ports[hp] = socket
	}

	// If encounter any error, close all hostports that just got opened.
	if retErr != nil {
		for hp, socket := range ports {
			if err := socket.Close(); err != nil {
				glog.Errorf("Cannot clean up hostport %d for pod %s: %v", hp.port, getPodFullName(podPortMapping), err)
			}
		}
		return nil, retErr
	}
	return ports, nil
}

// portMappingToHostport creates hostport structure based on input portmapping
func portMappingToHostport(portMapping *PortMapping) hostport {
	return hostport{
		port:     portMapping.HostPort,
		protocol: strings.ToLower(string(portMapping.Protocol)),
	}
}

// ensureKubeHostportChainLinked ensures the KUBE-HOSTPORTS chain is linked into the root of iptables
func ensureKubeHostportChainLinked(iptables utiliptables.Interface) error {
	glog.V(4).Info("Ensuring kubelet hostport chains")
	// Ensure kubeHostportsChain exists
	if _, err := iptables.EnsureChain(utiliptables.TableNAT, kubeHostportsChain); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, kubeHostportsChain, err)
	}

	// Enusure and fill kubeLocalDestChain with current info
	if err := syncLocaldest(iptables); err != nil {
		return fmt.Errorf("Failed to populate %s chain %s: %v", utiliptables.TableNAT, kubeLocalDestChain, err)
	}

	// Ensure it is linked from the root.
	tableChainsNeedJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableNAT, utiliptables.ChainOutput},
		{utiliptables.TableNAT, utiliptables.ChainPrerouting},
	}
	args := []string{
		"-m", "comment", "--comment", "maybe kube hostport",
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(kubeLocalDestChain),
	}
	for _, tc := range tableChainsNeedJumpServices {
		if _, err := iptables.EnsureRule(utiliptables.Prepend, tc.table, tc.chain, args...); err != nil {
			return fmt.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", tc.table, tc.chain, kubeHostportsChain, err)
		}
		removeOldLink(iptables, tc.table, tc.chain) // ignore errors
	}
	removeOldSNAT(iptables) // ignore errors

	return nil
}

func syncLocaldest(iptables utiliptables.Interface) error {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("syncLocaldest took %v", time.Since(start))
	}()

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingNATChains := make(map[utiliptables.Chain]string)
	iptablesSaveRaw, err := iptables.Save(utiliptables.TableNAT)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, iptablesSaveRaw)
	}

	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)
	writeLine(natChains, "*nat")
	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	if chain, ok := existingNATChains[kubeLocalDestChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubeLocalDestChain))
	}

	// Populate the LOCALDEST chain.
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return err
	}
	// Only consider packets with dest of a real interface addr for hostports.
	for _, a := range addrs {
		astr := a.String() // CIDR notation
		ip, _, err := net.ParseCIDR(astr)
		if err != nil {
			return err
		}
		if ip.To4() != nil {
			args := []string{
				"-A", string(kubeLocalDestChain),
				"-m", "comment", "--comment", `"maybe kube hostport"`,
				"-d", ip.String(),
				"-j", string(kubeHostportsChain),
			}
			writeLine(natRules, args...)
		}
	}
	writeLine(natRules, "COMMIT")

	natLines := append(natChains.Bytes(), natRules.Bytes()...)
	glog.V(3).Infof("Restoring iptables rules: %s", natLines)
	err = iptables.RestoreAll(natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		return fmt.Errorf("Failed to execute iptables-restore: %v", err)
	}
	return nil
}

// This can be removed after v1.8.
func removeOldLink(iptables utiliptables.Interface, table utiliptables.Table, chain utiliptables.Chain) error {
	if err := iptables.DeleteRule(table, chain,
		"-m", "comment", "--comment", "kube hostport portals",
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(kubeHostportsChain)); err != nil {
		return fmt.Errorf("Failed to remove old link from %s chain %s to %s: %v", table, chain, kubeHostportsChain, err)
	}
	return nil
}

// This can be removed after v1.8.
func removeOldSNAT(iptables utiliptables.Interface) error {
	if err := iptables.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPostrouting,
		"-s", "127.0.0.0/8",
		"-o", "cbr0",
		"-m", "comment", "--comment", "SNAT for localhost access to hostports",
		"-j", "MASQUERADE"); err != nil {
		return fmt.Errorf("Failed to remove old SNAT from nat chain POSTROUTING: %v", err)
	}
	return nil
}
