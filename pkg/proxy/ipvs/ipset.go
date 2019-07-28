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
	"k8s.io/apimachinery/pkg/util/sets"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"

	"fmt"
	"k8s.io/klog"
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

	// This ipset is no longer active but still used in previous versions.
	// DO NOT create an ipset using this name
	legacyKubeNodePortSetSCTPComment = "Kubernetes nodeport SCTP port for masquerade purpose"
	legacyKubeNodePortSetSCTP        = "KUBE-NODE-PORT-SCTP"

	// This ipset is no longer active but still used in previous versions.
	// DO NOT create an ipset using this name
	legacyKubeNodePortLocalSetSCTPComment = "Kubernetes nodeport SCTP port with externalTrafficPolicy=local"
	legacyKubeNodePortLocalSetSCTP        = "KUBE-NODE-PORT-LOCAL-SCTP"

	kubeNodePortSetSCTPComment = "Kubernetes nodeport SCTP port for masquerade purpose with type 'hash ip:port'"
	kubeNodePortSetSCTP        = "KUBE-NODE-PORT-SCTP-HASH"

	kubeNodePortLocalSetSCTPComment = "Kubernetes nodeport SCTP port with externalTrafficPolicy=local with type 'hash ip:port'"
	kubeNodePortLocalSetSCTP        = "KUBE-NODE-PORT-LOCAL-SCTP-HASH"
)

// IPSetVersioner can query the current ipset version.
type IPSetVersioner interface {
	// returns "X.Y"
	GetVersion() (string, error)
}

// IPSet wraps util/ipset which is used by IPVS proxier.
type IPSet struct {
	utilipset.IPSet
	// activeEntries is the current active entries of the ipset.
	activeEntries sets.String
	// handle is the util ipset interface handle.
	handle utilipset.Interface
}

// NewIPSet initialize a new IPSet struct
func NewIPSet(handle utilipset.Interface, name string, setType utilipset.Type, isIPv6 bool, comment string) *IPSet {
	hashFamily := utilipset.ProtocolFamilyIPV4
	if isIPv6 {
		hashFamily = utilipset.ProtocolFamilyIPV6
	}
	set := &IPSet{
		IPSet: utilipset.IPSet{
			Name:       name,
			SetType:    setType,
			HashFamily: hashFamily,
			Comment:    comment,
		},
		activeEntries: sets.NewString(),
		handle:        handle,
	}
	return set
}

func (set *IPSet) validateEntry(entry *utilipset.Entry) bool {
	return entry.Validate(&set.IPSet)
}

func (set *IPSet) isEmpty() bool {
	return len(set.activeEntries.UnsortedList()) == 0
}

func (set *IPSet) getComment() string {
	return fmt.Sprintf("\"%s\"", set.Comment)
}

func (set *IPSet) resetEntries() {
	set.activeEntries = sets.NewString()
}

func (set *IPSet) syncIPSetEntries() {
	appliedEntries, err := set.handle.ListEntries(set.Name)
	if err != nil {
		klog.Errorf("Failed to list ip set entries, error: %v", err)
		return
	}

	// currentIPSetEntries represents Endpoints watched from API Server.
	currentIPSetEntries := sets.NewString()
	for _, appliedEntry := range appliedEntries {
		currentIPSetEntries.Insert(appliedEntry)
	}

	if !set.activeEntries.Equal(currentIPSetEntries) {
		// Clean legacy entries
		for _, entry := range currentIPSetEntries.Difference(set.activeEntries).List() {
			if err := set.handle.DelEntry(entry, set.Name); err != nil {
				if !utilipset.IsNotFoundError(err) {
					klog.Errorf("Failed to delete ip set entry: %s from ip set: %s, error: %v", entry, set.Name, err)
				}
			} else {
				klog.V(3).Infof("Successfully delete legacy ip set entry: %s from ip set: %s", entry, set.Name)
			}
		}
		// Create active entries
		for _, entry := range set.activeEntries.Difference(currentIPSetEntries).List() {
			if err := set.handle.AddEntry(entry, &set.IPSet, true); err != nil {
				klog.Errorf("Failed to add entry: %v to ip set: %s, error: %v", entry, set.Name, err)
			} else {
				klog.V(3).Infof("Successfully add entry: %v to ip set: %s", entry, set.Name)
			}
		}
	}
}

func ensureIPSet(set *IPSet) error {
	if err := set.handle.CreateSet(&set.IPSet, true); err != nil {
		klog.Errorf("Failed to make sure ip set: %v exist, error: %v", set, err)
		return err
	}
	return nil
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
