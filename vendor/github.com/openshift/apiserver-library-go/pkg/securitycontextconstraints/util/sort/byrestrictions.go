package sort

import (
	"strings"

	"k8s.io/klog"

	corev1 "k8s.io/api/core/v1"

	securityv1 "github.com/openshift/api/security/v1"
)

// ByRestrictions is a helper to sort SCCs in order of most restrictive to least restrictive.
type ByRestrictions []*securityv1.SecurityContextConstraints

func (s ByRestrictions) Len() int {
	return len(s)
}
func (s ByRestrictions) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ByRestrictions) Less(i, j int) bool {
	return pointValue(s[i]) < pointValue(s[j])
}

// The following constants define the weight of the restrictions and used for
// calculating the points of the particular SCC. The lower the number, the more
// restrictive SCC is. Make sure that weak restrictions are always valued
// higher than the combination of the strong restrictions.

type points int

const (
	privilegedPoints points = 1000000

	hostNetworkPoints points = 200000
	hostPortsPoints   points = 400000

	hostVolumePoints       points = 100000
	nonTrivialVolumePoints points = 50000

	runAsAnyUserPoints points = 40000
	runAsNonRootPoints points = 30000
	runAsRangePoints   points = 20000
	runAsUserPoints    points = 10000

	capDefaultPoints  points = 5000
	capAddOnePoints   points = 300
	capAllowAllPoints points = 4000
	capAllowOnePoints points = 10
	capDropAllPoints  points = -3000
	capDropOnePoints  points = -50
	capMaxPoints      points = 9999
	capMinPoints      points = 0

	noPoints points = 0
)

// pointValue places a value on the SCC based on the settings of the SCC that can be used
// to determine how restrictive it is.  The lower the number, the more restrictive it is.
func pointValue(constraint *securityv1.SecurityContextConstraints) points {
	totalPoints := noPoints

	if constraint.AllowPrivilegedContainer {
		totalPoints += privilegedPoints
	}

	// add points based on volume requests
	totalPoints += volumePointValue(constraint)

	if constraint.AllowHostNetwork {
		totalPoints += hostNetworkPoints
	}
	if constraint.AllowHostPorts {
		totalPoints += hostPortsPoints
	}

	// add points based on capabilities
	totalPoints += capabilitiesPointValue(constraint)

	// the map contains points for both RunAsUser and SELinuxContext
	// strategies by taking advantage that they have identical strategy names
	strategiesPoints := map[string]points{
		string(securityv1.RunAsUserStrategyRunAsAny):         runAsAnyUserPoints,
		string(securityv1.RunAsUserStrategyMustRunAsNonRoot): runAsNonRootPoints,
		string(securityv1.RunAsUserStrategyMustRunAsRange):   runAsRangePoints,
		string(securityv1.RunAsUserStrategyMustRunAs):        runAsUserPoints,
	}

	strategyType := string(constraint.SELinuxContext.Type)
	points, found := strategiesPoints[strategyType]
	if found {
		totalPoints += points
	} else {
		klog.Warningf("SELinuxContext type %q has no point value, this may cause issues in sorting SCCs by restriction", strategyType)
	}

	strategyType = string(constraint.RunAsUser.Type)
	points, found = strategiesPoints[strategyType]
	if found {
		totalPoints += points
	} else {
		klog.Warningf("RunAsUser type %q has no point value, this may cause issues in sorting SCCs by restriction", strategyType)
	}

	return totalPoints
}

// volumePointValue returns a score based on the volumes allowed by the SCC.
// Allowing a host volume will return a score of 100000.  Allowance of anything other
// than Secret, ConfigMap, EmptyDir, DownwardAPI, Projected, and None will result in
// a score of 50000.  If the SCC only allows these trivial types, it will have a
// score of 0.
func volumePointValue(scc *securityv1.SecurityContextConstraints) points {
	hasHostVolume := false
	hasNonTrivialVolume := false
	for _, v := range scc.Volumes {
		switch v {
		case securityv1.FSTypeHostPath, securityv1.FSTypeAll:
			hasHostVolume = true
			// nothing more to do, this is the max point value
			break
		// it is easier to specifically list the trivial volumes and allow the
		// default case to be non-trivial so we don't have to worry about adding
		// volumes in the future unless they're trivial.
		case securityv1.FSTypeSecret, securityv1.FSTypeConfigMap, securityv1.FSTypeEmptyDir,
			securityv1.FSTypeDownwardAPI, securityv1.FSProjected, securityv1.FSTypeNone:
			// do nothing
		default:
			hasNonTrivialVolume = true
		}
	}

	if hasHostVolume {
		return hostVolumePoints
	}
	if hasNonTrivialVolume {
		return nonTrivialVolumePoints
	}
	return noPoints
}

// hasCap checks for needle in haystack.
func hasCap(needle string, haystack []corev1.Capability) bool {
	for _, c := range haystack {
		if needle == strings.ToUpper(string(c)) {
			return true
		}
	}
	return false
}

// capabilitiesPointValue returns a score based on the capabilities allowed,
// added, or removed by the SCC. This allow us to prefer the more restrictive
// SCC.
func capabilitiesPointValue(scc *securityv1.SecurityContextConstraints) points {
	capsPoints := capDefaultPoints
	capsPoints += capAddOnePoints * points(len(scc.DefaultAddCapabilities))
	if hasCap(string(securityv1.AllowAllCapabilities), scc.AllowedCapabilities) {
		capsPoints += capAllowAllPoints
	} else if hasCap("ALL", scc.AllowedCapabilities) {
		capsPoints += capAllowAllPoints
	} else {
		capsPoints += capAllowOnePoints * points(len(scc.AllowedCapabilities))
	}
	if hasCap("ALL", scc.RequiredDropCapabilities) {
		capsPoints += capDropAllPoints
	} else {
		capsPoints += capDropOnePoints * points(len(scc.RequiredDropCapabilities))
	}
	if capsPoints > capMaxPoints {
		return capMaxPoints
	} else if capsPoints < capMinPoints {
		return capMinPoints
	}
	return capsPoints
}
