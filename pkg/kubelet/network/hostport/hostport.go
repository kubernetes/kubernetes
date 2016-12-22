/*
Copyright 2014 The Kubernetes Authors.

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
	"crypto/sha256"
	"encoding/base32"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	iptablesproxy "k8s.io/kubernetes/pkg/proxy/iptables"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

const (
	// the hostport chain
	kubeHostportsChain utiliptables.Chain = "KUBE-HOSTPORTS"
	// prefix for hostport chains
	kubeHostportChainPrefix string = "KUBE-HP-"
)

type HostportHandler interface {
	OpenPodHostportsAndSync(newPod *ActivePod, natInterfaceName string, activePods []*ActivePod) error
	SyncHostports(natInterfaceName string, activePods []*ActivePod) error
}

type ActivePod struct {
	Pod *v1.Pod
	IP  net.IP
}

type hostportOpener func(*hostport) (closeable, error)

type handler struct {
	hostPortMap map[hostport]closeable
	iptables    utiliptables.Interface
	portOpener  hostportOpener
}

func NewHostportHandler() HostportHandler {
	iptInterface := utiliptables.New(utilexec.New(), utildbus.New(), utiliptables.ProtocolIpv4)
	return &handler{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptInterface,
		portOpener:  openLocalPort,
	}
}

type closeable interface {
	Close() error
}

type hostport struct {
	port     int32
	protocol string
}

type targetPod struct {
	podFullName string
	podIP       string
}

func (hp *hostport) String() string {
	return fmt.Sprintf("%s:%d", hp.protocol, hp.port)
}

//openPodHostports opens all hostport for pod and returns the map of hostport and socket
func (h *handler) openHostports(pod *v1.Pod) error {
	var retErr error
	ports := make(map[hostport]closeable)
	for _, container := range pod.Spec.Containers {
		for _, port := range container.Ports {
			if port.HostPort <= 0 {
				// Ignore
				continue
			}
			hp := hostport{
				port:     port.HostPort,
				protocol: strings.ToLower(string(port.Protocol)),
			}
			socket, err := h.portOpener(&hp)
			if err != nil {
				retErr = fmt.Errorf("Cannot open hostport %d for pod %s: %v", port.HostPort, kubecontainer.GetPodFullName(pod), err)
				break
			}
			ports[hp] = socket
		}
		if retErr != nil {
			break
		}
	}
	// If encounter any error, close all hostports that just got opened.
	if retErr != nil {
		for hp, socket := range ports {
			if err := socket.Close(); err != nil {
				glog.Errorf("Cannot clean up hostport %d for pod %s: %v", hp.port, kubecontainer.GetPodFullName(pod), err)
			}
		}
		return retErr
	}

	for hostPort, socket := range ports {
		h.hostPortMap[hostPort] = socket
	}

	return nil
}

// gatherAllHostports returns all hostports that should be presented on node,
// given the list of pods running on that node and ignoring host network
// pods (which don't need hostport <-> container port mapping).
func gatherAllHostports(activePods []*ActivePod) (map[v1.ContainerPort]targetPod, error) {
	podHostportMap := make(map[v1.ContainerPort]targetPod)
	for _, r := range activePods {
		if r.IP.To4() == nil {
			return nil, fmt.Errorf("Invalid or missing pod %s IP", kubecontainer.GetPodFullName(r.Pod))
		}

		// should not handle hostports for hostnetwork pods
		if r.Pod.Spec.HostNetwork {
			continue
		}

		for _, container := range r.Pod.Spec.Containers {
			for _, port := range container.Ports {
				if port.HostPort != 0 {
					podHostportMap[port] = targetPod{podFullName: kubecontainer.GetPodFullName(r.Pod), podIP: r.IP.String()}
				}
			}
		}
	}
	return podHostportMap, nil
}

// Join all words with spaces, terminate with newline and write to buf.
func writeLine(buf *bytes.Buffer, words ...string) {
	buf.WriteString(strings.Join(words, " ") + "\n")
}

//hostportChainName takes containerPort for a pod and returns associated iptables chain.
// This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-SVC-".  We do
// this because IPTables Chain Names must be <= 28 chars long, and the longer
// they are the harder they are to read.
func hostportChainName(cp v1.ContainerPort, podFullName string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(string(cp.HostPort) + string(cp.Protocol) + podFullName))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain(kubeHostportChainPrefix + encoded[:16])
}

// OpenPodHostportsAndSync opens hostports for a new pod, gathers all hostports on
// node, sets up iptables rules enable them. And finally clean up stale hostports.
// 'newPod' must also be present in 'activePods'.
func (h *handler) OpenPodHostportsAndSync(newPod *ActivePod, natInterfaceName string, activePods []*ActivePod) error {
	// try to open pod host port if specified
	if err := h.openHostports(newPod.Pod); err != nil {
		return err
	}

	// Add the new pod to active pods if it's not present.
	var found bool
	for _, p := range activePods {
		if p.Pod.UID == newPod.Pod.UID {
			found = true
			break
		}
	}
	if !found {
		activePods = append(activePods, newPod)
	}

	return h.SyncHostports(natInterfaceName, activePods)
}

// SyncHostports gathers all hostports on node and setup iptables rules enable them. And finally clean up stale hostports
func (h *handler) SyncHostports(natInterfaceName string, activePods []*ActivePod) error {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("syncHostportsRules took %v", time.Since(start))
	}()

	containerPortMap, err := gatherAllHostports(activePods)
	if err != nil {
		return err
	}

	glog.V(4).Info("Ensuring kubelet hostport chains")
	// Ensure kubeHostportChain
	if _, err := h.iptables.EnsureChain(utiliptables.TableNAT, kubeHostportsChain); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, kubeHostportsChain, err)
	}
	tableChainsNeedJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableNAT, utiliptables.ChainOutput},
		{utiliptables.TableNAT, utiliptables.ChainPrerouting},
	}
	args := []string{"-m", "comment", "--comment", "kube hostport portals",
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(kubeHostportsChain)}
	for _, tc := range tableChainsNeedJumpServices {
		if _, err := h.iptables.EnsureRule(utiliptables.Prepend, tc.table, tc.chain, args...); err != nil {
			return fmt.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", tc.table, tc.chain, kubeHostportsChain, err)
		}
	}
	// Need to SNAT traffic from localhost
	args = []string{"-m", "comment", "--comment", "SNAT for localhost access to hostports", "-o", natInterfaceName, "-s", "127.0.0.0/8", "-j", "MASQUERADE"}
	if _, err := h.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s jumps to MASQUERADE: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, err)
	}

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingNATChains := make(map[utiliptables.Chain]string)
	iptablesSaveRaw, err := h.iptables.Save(utiliptables.TableNAT)
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
	if chain, ok := existingNATChains[kubeHostportsChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubeHostportsChain))
	}

	// Accumulate NAT chains to keep.
	activeNATChains := map[utiliptables.Chain]bool{} // use a map as a set

	for containerPort, target := range containerPortMap {
		protocol := strings.ToLower(string(containerPort.Protocol))
		hostportChain := hostportChainName(containerPort, target.podFullName)
		if chain, ok := existingNATChains[hostportChain]; ok {
			writeLine(natChains, chain)
		} else {
			writeLine(natChains, utiliptables.MakeChainLine(hostportChain))
		}

		activeNATChains[hostportChain] = true

		// Redirect to hostport chain
		args := []string{
			"-A", string(kubeHostportsChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, containerPort.HostPort),
			"-m", protocol, "-p", protocol,
			"--dport", fmt.Sprintf("%d", containerPort.HostPort),
			"-j", string(hostportChain),
		}
		writeLine(natRules, args...)

		// Assuming kubelet is syncing iptables KUBE-MARK-MASQ chain
		// If the request comes from the pod that is serving the hostport, then SNAT
		args = []string{
			"-A", string(hostportChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, containerPort.HostPort),
			"-s", target.podIP, "-j", string(iptablesproxy.KubeMarkMasqChain),
		}
		writeLine(natRules, args...)

		// Create hostport chain to DNAT traffic to final destination
		// IPTables will maintained the stats for this chain
		args = []string{
			"-A", string(hostportChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, containerPort.HostPort),
			"-m", protocol, "-p", protocol,
			"-j", "DNAT", fmt.Sprintf("--to-destination=%s:%d", target.podIP, containerPort.ContainerPort),
		}
		writeLine(natRules, args...)
	}

	// Delete chains no longer in use.
	for chain := range existingNATChains {
		if !activeNATChains[chain] {
			chainString := string(chain)
			if !strings.HasPrefix(chainString, kubeHostportChainPrefix) {
				// Ignore chains that aren't ours.
				continue
			}
			// We must (as per iptables) write a chain-line for it, which has
			// the nice effect of flushing the chain.  Then we can remove the
			// chain.
			writeLine(natChains, existingNATChains[chain])
			writeLine(natRules, "-X", chainString)
		}
	}
	writeLine(natRules, "COMMIT")

	natLines := append(natChains.Bytes(), natRules.Bytes()...)
	glog.V(3).Infof("Restoring iptables rules: %s", natLines)
	err = h.iptables.RestoreAll(natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		return fmt.Errorf("Failed to execute iptables-restore: %v", err)
	}

	h.cleanupHostportMap(containerPortMap)
	return nil
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

// cleanupHostportMap closes obsolete hostports
func (h *handler) cleanupHostportMap(containerPortMap map[v1.ContainerPort]targetPod) {
	// compute hostports that are supposed to be open
	currentHostports := make(map[hostport]bool)
	for containerPort := range containerPortMap {
		hp := hostport{
			port:     containerPort.HostPort,
			protocol: strings.ToLower(string(containerPort.Protocol)),
		}
		currentHostports[hp] = true
	}

	// close and delete obsolete hostports
	for hp, socket := range h.hostPortMap {
		if _, ok := currentHostports[hp]; !ok {
			socket.Close()
			glog.V(3).Infof("Closed local port %s", hp.String())
			delete(h.hostPortMap, hp)
		}
	}
}
