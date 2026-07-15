package v1

const (
	UIDRangeAnnotation = "openshift.io/sa.scc.uid-range"
	// SupplementalGroupsAnnotation contains a comma delimited list of allocated supplemental groups
	// for the namespace.  Groups are in the form of a Block which supports {start}/{length} or {start}-{end}
	SupplementalGroupsAnnotation = "openshift.io/sa.scc.supplemental-groups"
	MCSAnnotation                = "openshift.io/sa.scc.mcs"
	ValidatedSCCAnnotation       = "openshift.io/scc"
	// This annotation pins required SCCs for core OpenShift workloads to prevent preemption of custom SCCs.
	// It is being used in the SCC admission plugin.
	RequiredSCCAnnotation = "openshift.io/required-scc"

	// MinimallySufficientPodSecurityStandard indicates the PodSecurityStandard that matched the SCCs available to the users of the namespace.
	MinimallySufficientPodSecurityStandard = "security.openshift.io/MinimallySufficientPodSecurityStandard"

	// ValidatedSCCSubjectTypeAnnotation indicates the subject type that allowed the
	// SCC admission. This can be used by controllers to detect potential issues
	// between user-driven SCC usage and the ServiceAccount-driven SCC usage.
	ValidatedSCCSubjectTypeAnnotation = "security.openshift.io/validated-scc-subject-type"
)
