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
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utilversion "k8s.io/kubernetes/pkg/util/version"

	"github.com/golang/glog"
)

const (
	// MinIPSetCheckVersion is the min ipset version we need.  IPv6 is supported in ipset 6.x
	MinIPSetCheckVersion = "6.0"

	// KubeLoopBackIPSet is used to store endpoints dst ip:port, source ip for solving hairpin purpose.
	KubeLoopBackIPSet = "KUBE-LOOP-BACK"

	// KubeClusterIPSet is used to store service cluster ip + port for masquerade purpose.
	KubeClusterIPSet = "KUBE-CLUSTER-IP"

	// KubeExternalIPSet is used to store service external ip + port for masquerade and filter purpose.
	KubeExternalIPSet = "KUBE-EXTERNAL-IP"

	// KubeLoadBalancerSet is used to store service load balancer ingress ip + port, it is the service lb portal.
	KubeLoadBalancerSet = "KUBE-LOAD-BALANCER"

	// KubeLoadBalancerMasqSet is used to store service load balancer ingress ip + port for masquerade purpose.
	KubeLoadBalancerMasqSet = "KUBE-LOAD-BALANCER-MASQ"

	// KubeLoadBalancerSourceIPSet is used to store service load balancer ingress ip + port + source IP for packet filter purpose.
	KubeLoadBalancerSourceIPSet = "KUBE-LOAD-BALANCER-SOURCE-IP"

	// KubeLoadBalancerSourceCIDRSet is used to store service load balancer ingress ip + port + source cidr for packet filter purpose.
	KubeLoadBalancerSourceCIDRSet = "KUBE-LOAD-BALANCER-SOURCE-CIDR"

	// KubeNodePortSetTCP is used to store nodeport TCP port for masquerade purpose.
	KubeNodePortSetTCP = "KUBE-NODE-PORT-TCP"

	// KubeNodePortSetUDP is used to store nodeport UDP port for masquerade purpose.
	KubeNodePortSetUDP = "KUBE-NODE-PORT-UDP"
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
func NewIPSet(handle utilipset.Interface, name string, setType utilipset.Type, isIPv6 bool) *IPSet {
	hashFamily := utilipset.ProtocolFamilyIPV4
	if isIPv6 {
		hashFamily = utilipset.ProtocolFamilyIPV6
	}
	set := &IPSet{
		IPSet: utilipset.IPSet{
			Name:       name,
			SetType:    setType,
			HashFamily: hashFamily,
		},
		activeEntries: sets.NewString(),
		handle:        handle,
	}
	return set
}

func (set *IPSet) isEmpty() bool {
	return len(set.activeEntries.UnsortedList()) == 0
}

func (set *IPSet) resetEntries() {
	set.activeEntries = sets.NewString()
}

func (set *IPSet) syncIPSetEntries() {
	appliedEntries, err := set.handle.ListEntries(set.Name)
	if err != nil {
		glog.Errorf("Failed to list ip set entries, error: %v", err)
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
					glog.Errorf("Failed to delete ip set entry: %s from ip set: %s, error: %v", entry, set.Name, err)
				}
			} else {
				glog.V(3).Infof("Successfully delete legacy ip set entry: %s from ip set: %s", entry, set.Name)
			}
		}
		// Create active entries
		for _, entry := range set.activeEntries.Difference(currentIPSetEntries).List() {
			if err := set.handle.AddEntry(entry, set.Name, true); err != nil {
				glog.Errorf("Failed to add entry: %v to ip set: %s, error: %v", entry, set.Name, err)
			} else {
				glog.V(3).Infof("Successfully add entry: %v to ip set: %s", entry, set.Name)
			}
		}
	}
}

func ensureIPSets(ipSets ...*IPSet) error {
	for _, set := range ipSets {
		if err := set.handle.CreateSet(&set.IPSet, true); err != nil {
			glog.Errorf("Failed to make sure ip set: %v exist, error: %v", set, err)
			return err
		}
	}
	return nil
}

// checkMinVersion checks if ipset current version satisfies required min version
func checkMinVersion(vstring string) bool {
	version, err := utilversion.ParseGeneric(vstring)
	if err != nil {
		glog.Errorf("vstring (%s) is not a valid version string: %v", vstring, err)
		return false
	}

	minVersion, err := utilversion.ParseGeneric(MinIPSetCheckVersion)
	if err != nil {
		glog.Errorf("MinCheckVersion (%s) is not a valid version string: %v", MinIPSetCheckVersion, err)
		return false
	}
	return !version.LessThan(minVersion)
}
