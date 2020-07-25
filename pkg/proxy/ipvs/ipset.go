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

package ipvs

import (
	"bytes"
	"fmt"
	"strings"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/sets"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
)

const (
	// MinIPSetCheckVersion is the min ipset version we need.  IPv6 is supported in ipset 6.x
	MinIPSetCheckVersion = "6.0"

	kubeLoopBackIPSetComment = "Kubernetes endpoints dst ip:port, source ip for solving hairpin purpose"
	kubeLoopBackIPSet        = "KUBE-LOOP-BACK"

	kubeClusterIPSetComment = "Kubernetes service cluster ip + port for masquerade purpose"
	kubeClusterIPSet        = "KUBE-CLUSTER-IP"

	kubeExternalIPSetComment = "Kubernetes service external ip + port for masquerade and filter purpose"
	kubeExternalIPSet        = "KUBE-EXTERNAL-IP"

	kubeExternalIPLocalSetComment = "Kubernetes service external ip + port with externalTrafficPolicy=local"
	kubeExternalIPLocalSet        = "KUBE-EXTERNAL-IP-LOCAL"

	kubeLoadBalancerSetComment = "Kubernetes service lb portal"
	kubeLoadBalancerSet        = "KUBE-LOAD-BALANCER"

	kubeLoadBalancerLocalSetComment = "Kubernetes service load balancer ip + port with externalTrafficPolicy=local"
	kubeLoadBalancerLocalSet        = "KUBE-LOAD-BALANCER-LOCAL"

	kubeLoadbalancerFWSetComment = "Kubernetes service load balancer ip + port for load balancer with sourceRange"
	kubeLoadbalancerFWSet        = "KUBE-LOAD-BALANCER-FW"

	kubeLoadBalancerSourceIPSetComment = "Kubernetes service load balancer ip + port + source IP for packet filter purpose"
	kubeLoadBalancerSourceIPSet        = "KUBE-LOAD-BALANCER-SOURCE-IP"

	kubeLoadBalancerSourceCIDRSetComment = "Kubernetes service load balancer ip + port + source cidr for packet filter purpose"
	kubeLoadBalancerSourceCIDRSet        = "KUBE-LOAD-BALANCER-SOURCE-CIDR"

	kubeNodePortSetTCPComment = "Kubernetes nodeport TCP port for masquerade purpose"
	kubeNodePortSetTCP        = "KUBE-NODE-PORT-TCP"

	kubeNodePortLocalSetTCPComment = "Kubernetes nodeport TCP port with externalTrafficPolicy=local"
	kubeNodePortLocalSetTCP        = "KUBE-NODE-PORT-LOCAL-TCP"

	kubeNodePortSetUDPComment = "Kubernetes nodeport UDP port for masquerade purpose"
	kubeNodePortSetUDP        = "KUBE-NODE-PORT-UDP"

	kubeNodePortLocalSetUDPComment = "Kubernetes nodeport UDP port with externalTrafficPolicy=local"
	kubeNodePortLocalSetUDP        = "KUBE-NODE-PORT-LOCAL-UDP"

	kubeNodePortSetSCTPComment = "Kubernetes nodeport SCTP port for masquerade purpose with type 'hash ip:port'"
	kubeNodePortSetSCTP        = "KUBE-NODE-PORT-SCTP-HASH"

	kubeNodePortLocalSetSCTPComment = "Kubernetes nodeport SCTP port with externalTrafficPolicy=local with type 'hash ip:port'"
	kubeNodePortLocalSetSCTP        = "KUBE-NODE-PORT-LOCAL-SCTP-HASH"

	addCommand    = "add"
	deleteCommand = "del"
	createCommand = "create"
)

// IPSetVersioner can query the current ipset version.
type IPSetVersioner interface {
	// returns "X.Y"
	GetVersion() (string, error)
}

type ipsetInfo struct {
	name    string
	setType utilipset.Type
	comment string
}

// ipsetManager wraps util/ipset which is used by IPVS proxier.
type ipsetManager struct {
	// setList a map of current sets list, with key of set name and value of set content
	setList map[string]*utilipset.IPSet
	// activeEntries is a map of current active entries, with key of set name and value of set entries
	activeEntries map[string]sets.String
	// handler is the util ipset interface handler.
	handler utilipset.Interface
}

func newIPSetManager(isIPv6 bool, setInfoList []ipsetInfo, ipset utilipset.Interface) *ipsetManager {
	m := &ipsetManager{
		setList:       make(map[string]*utilipset.IPSet, len(setInfoList)),
		activeEntries: make(map[string]sets.String, len(setInfoList)),
		handler:       ipset,
	}
	for _, info := range setInfoList {
		set := newIPSet(info.name, info.setType, isIPv6, info.comment)
		m.setList[info.name] = set
		m.activeEntries[info.name] = sets.NewString()
	}
	return m
}

// newIPSet initialize a new IPSet
func newIPSet(name string, setType utilipset.Type, isIPv6 bool, comment string) *utilipset.IPSet {
	hashFamily := utilipset.ProtocolFamilyIPV4
	if isIPv6 {
		hashFamily = utilipset.ProtocolFamilyIPV6
		// In dual-stack both ipv4 and ipv6 ipset's can co-exist. To
		// ensure unique names the prefix for ipv6 is changed from
		// "KUBE-" to "KUBE-6-". The "KUBE-" prefix is kept for
		// backward compatibility. The maximum name length of an ipset
		// is 31 characters which must be taken into account.  The
		// ipv4 names are not altered to minimize the risk for
		// problems on upgrades.
		if strings.HasPrefix(name, "KUBE-") {
			name = strings.Replace(name, "KUBE-", "KUBE-6-", 1)
			if len(name) > 31 {
				klog.Warningf("ipset name truncated; [%s] -> [%s]", name, name[:31])
				name = name[:31]
			}
		}
	}
	set := &utilipset.IPSet{
		Name:       name,
		SetType:    setType,
		HashFamily: hashFamily,
		Comment:    comment,
	}
	set.SetIPSetDefaults()
	return set
}

func (s *ipsetManager) isSetEmpty(setName string) bool {
	return s.activeEntries[setName].Len() == 0
}

func (s *ipsetManager) getSetComment(setName string) string {
	return fmt.Sprintf("\"%s\"", s.setList[setName].Comment)
}

func (s *ipsetManager) resetEntries() {
	for setName := range s.activeEntries {
		s.activeEntries[setName] = sets.NewString()
	}
}

func (s *ipsetManager) entryValid(setName string, entry *utilipset.Entry) bool {
	if s.setList[setName] == nil {
		return false
	}
	return entry.Validate(s.setList[setName])
}

func (s *ipsetManager) insertIPSetEntry(setName string, entry *utilipset.Entry) error {
	if !s.entryValid(setName, entry) {
		return fmt.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, setName))
	}
	s.activeEntries[setName].Insert(entry.String())
	return nil
}

func (s *ipsetManager) getIPSet(setName string) *utilipset.IPSet {
	return s.setList[setName]
}

func (s *ipsetManager) syncIPSetEntries() {
	existingIPSetEntries := make(map[string]sets.String)
	buf := bytes.NewBuffer(nil)
	data, err := s.handler.SaveAllSets()
	if err != nil {
		klog.Errorf("Failed to exec ipset save, syncing all set entries: %v", err)
	} else {
		existingIPSetEntries = readSetsEntries(data)
	}
	// make sure ip sets exists in the system.
	for _, set := range s.setList {
		writeCreateSet(buf, set)
	}

	// cleanup ip sets entries and add new entries
	for set, entries := range s.activeEntries {
		if existingEntries, found := existingIPSetEntries[set]; found {
			for entry := range existingEntries.Difference(entries) {
				writeDelEntry(buf, set, entry)
			}
			for entry := range entries.Difference(existingEntries) {
				writeAddEntry(buf, set, entry)
			}
		} else {
			for entry := range entries {
				writeAddEntry(buf, set, entry)
			}
		}
	}

	klog.V(5).Infof("Restoring ipset sets and entries: %s", buf.Bytes())
	if err := s.handler.RestoreSets(buf.Bytes()); err != nil {
		klog.Errorf("Failed to exec ipset restore, sync all set entries err: %v", err)
	}
}

// checkMinVersion checks if ipset current version satisfies required min version
func checkMinVersion(vstring string) bool {
	version, err := utilversion.ParseGeneric(vstring)
	if err != nil {
		klog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return false
	}

	minVersion, err := utilversion.ParseGeneric(MinIPSetCheckVersion)
	if err != nil {
		klog.Errorf("MinCheckVersion (%s) is not a valid version string: %v", MinIPSetCheckVersion, err)
		return false
	}
	return !version.LessThan(minVersion)
}

func readSetsEntries(data []byte) map[string]sets.String {
	buffer := bytes.NewBuffer(data)
	setEntries := make(map[string]sets.String)
	for {
		line, err := buffer.ReadString('\n')
		if err != nil || len(line) == 0 {
			break
		}
		line = strings.TrimSuffix(line, "\n")
		if strings.HasPrefix(line, addCommand) {
			entrySlices := strings.Split(line, " ")
			if len(entrySlices) < 3 {
				klog.Errorf("Read set entries, got invalid entry: %s", line)
				continue
			}
			set, entry := entrySlices[1], entrySlices[2]
			if setEntries[set] == nil {
				setEntries[set] = sets.NewString(entry)
			} else {
				setEntries[set].Insert(entry)
			}
		}
	}
	return setEntries
}

func writeCreateSet(buf *bytes.Buffer, set *utilipset.IPSet) {
	line := strings.Join([]string{createCommand, set.String(), "-exist"}, " ")
	buf.WriteString(line)
	buf.WriteByte('\n')
}

func writeAddEntry(buf *bytes.Buffer, setName, entry string) {
	line := strings.Join([]string{addCommand, setName, entry, "-exist"}, " ")
	buf.WriteString(line)
	buf.WriteByte('\n')
}

func writeDelEntry(buf *bytes.Buffer, setName, entry string) {
	line := strings.Join([]string{deleteCommand, setName, entry}, " ")
	buf.WriteString(line)
	buf.WriteByte('\n')
}
