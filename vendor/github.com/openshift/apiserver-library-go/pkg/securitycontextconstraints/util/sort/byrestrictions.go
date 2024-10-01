package sort

import (
	"fmt"
	"strings"

	"k8s.io/klog/v2"

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

// To be able to reason about what restriction was favored to be more restrictive
// ensure that number ranges between distinct restrictions are mutually exclusive.

type points int

const (
	// max total 3_189_999 = 1_600_000 + 1_589_999
	privilegedPoints points = 1_600_000

	// max total: 1_589_999 = 800_000 + 789_999
	hostPortsPoints points = 800_000

	// max total: 789_999 = 400_000 + 389_999
	hostNetworkPoints points = 400_000

	// max total: 389_999 = 200_000 + 189_999
	hostVolumePoints points = 200_000

	// max total 189_999 = 100_000 + 89_999
	nonTrivialVolumePoints points = 100_000

	// Note: boundaries for runAs* must be considered twice,
	// because they are accumulated for both SELinuxContext.Type
	// and RunAsUser.Type.
	//
	// max total 89_999 = (40_000 * 2) + 9999
	runAsAnyUserPoints points = 40_000
	runAsNonRootPoints points = 30_000
	runAsRangePoints   points = 20_000
	runAsUserPoints    points = 10_000

	// cap* max points = 9999
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

func moreRestrictiveReason(p, q points) string {
	if p >= q {
		return ""
	}

	var done bool
	var reason string
	dueTo := func(x points, what string) (points, points, string, bool) {
		switch {
		case p >= x && q >= x:
			p -= x
			q -= x
		case p < x && q >= x:
			return p, q, fmt.Sprintf("forbids %s", what), true
		}
		return p, q, "", false
	}
	if p, q, reason, done = dueTo(privilegedPoints, "privileged"); done {
		return reason
	}
	if p, q, reason, done = dueTo(hostPortsPoints, "host ports"); done {
		return reason
	}
	if p, q, reason, done = dueTo(hostNetworkPoints, "host networking"); done {
		return reason
	}
	if p, q, reason, done = dueTo(hostVolumePoints, "host volume mounts"); done {
		return reason
	}
	if p, q, reason, done = dueTo(nonTrivialVolumePoints, "non-trivial volume mounts"); done {
		return reason
	}

	runsAsP, capP := p/10000, p%10000
	runsAsQ, capQ := q/10000, q%10000

	if runsAsP < runsAsQ {
		// this can be either SELinuxContext.Type or RunAsUser.Type
		return "permits less runAs strategies"
	}
	if capP < capQ {
		return "permits less capabilities"
	}

	// this should never happen due to the comparison at the very top
	return "is equally restrictive"
}

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
// Allowing a host volume will return a score of 200_000.  Allowance of anything other
// than Secret, ConfigMap, EmptyDir, DownwardAPI, Projected, and None will result in
// a score of 100_000.  If the SCC only allows these trivial types, it will have a
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
// It never returns a score higher than capMaxPoints and lower than capMinPoints.
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
