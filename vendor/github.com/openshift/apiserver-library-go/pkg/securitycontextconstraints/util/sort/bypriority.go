package sort

import (
	securityv1 "github.com/openshift/api/security/v1"
)

// ByPriority is a helper to sort SCCs based on priority.  If priorities are equal
// a string compare of the name is used.
type ByPriority []*securityv1.SecurityContextConstraints

func (s ByPriority) Len() int {
	return len(s)
}
func (s ByPriority) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ByPriority) Less(i, j int) bool {
	ret, _ := s.LessWithReason(i, j)
	return ret
}
func (s ByPriority) LessWithReason(i, j int) (bool, string) {
	iSCC := s[i]
	jSCC := s[j]

	iSCCPriority := getPriority(iSCC)
	jSCCPriority := getPriority(jSCC)

	// a higher priority is considered "less" so that it moves to the front of the line
	if iSCCPriority > jSCCPriority {
		return true, "has higher priority"
	}

	if iSCCPriority < jSCCPriority {
		return false, "has lower priority"
	}

	// priorities are equal, let's try point values
	iRestrictionScore := pointValue(iSCC)
	jRestrictionScore := pointValue(jSCC)

	// a lower restriction score is considered "less" so that it moves to the front of the line
	// (the greater the score, the more lax the SCC is)
	if iRestrictionScore < jRestrictionScore {
		return true, moreRestrictiveReason(iRestrictionScore, jRestrictionScore)
	}

	if iRestrictionScore > jRestrictionScore {
		return false, moreRestrictiveReason(jRestrictionScore, iRestrictionScore)
	}

	// they are still equal, sort by name
	return iSCC.Name < jSCC.Name, "ordered by name"
}

func getPriority(scc *securityv1.SecurityContextConstraints) int {
	if scc.Priority == nil {
		return 0
	}
	return int(*scc.Priority)
}
