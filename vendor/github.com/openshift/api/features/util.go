package features

import (
	"fmt"
	"net/url"
	"strings"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// FeatureGateDescription is a golang-only interface used to contains details for a feature gate.
type FeatureGateDescription struct {
	// FeatureGateAttributes is the information that appears in the API
	FeatureGateAttributes configv1.FeatureGateAttributes

	// OwningJiraComponent is the jira component that owns most of the impl and first assignment for the bug.
	// This is the team that owns the feature long term.
	OwningJiraComponent string
	// ResponsiblePerson is the person who is on the hook for first contact.  This is often, but not always, a team lead.
	// It is someone who can make the promise on the behalf of the team.
	ResponsiblePerson string
	// OwningProduct is the product that owns the lifecycle of the gate.
	OwningProduct OwningProduct
	// EnhancementPR is the PR for the enhancement.
	EnhancementPR string
}

type FeatureGateEnabledDisabled struct {
	Enabled  []FeatureGateDescription
	Disabled []FeatureGateDescription
}

type ClusterProfileName string

var (
	Hypershift         = ClusterProfileName("include.release.openshift.io/ibm-cloud-managed")
	SelfManaged        = ClusterProfileName("include.release.openshift.io/self-managed-high-availability")
	AllClusterProfiles = []ClusterProfileName{Hypershift, SelfManaged}
)

type OwningProduct string

var (
	ocpSpecific = OwningProduct("OCP")
	kubernetes  = OwningProduct("Kubernetes")
)

type featureGateEnableOption func(s *featureGateStatus)

type versionOperator string

var (
	equal              = versionOperator("=")
	greaterThan        = versionOperator(">")
	greaterThanOrEqual = versionOperator(">=")
	lessThan           = versionOperator("<")
	lessThanOrEqual    = versionOperator("<=")
)

func inVersion(version uint64, op versionOperator) featureGateEnableOption {
	return func(s *featureGateStatus) {
		switch op {
		case equal:
			s.version.Insert(version)
		case greaterThan:
			for v := version + 1; v <= maxOpenshiftVersion; v++ {
				s.version.Insert(v)
			}
		case greaterThanOrEqual:
			for v := version; v <= maxOpenshiftVersion; v++ {
				s.version.Insert(v)
			}
		case lessThan:
			for v := minOpenshiftVersion; v < version; v++ {
				s.version.Insert(v)
			}
		case lessThanOrEqual:
			for v := minOpenshiftVersion; v <= version; v++ {
				s.version.Insert(v)
			}
		default:
			panic(fmt.Sprintf("invalid version operator: %s", op))
		}
	}
}

func inClusterProfile(clusterProfile ClusterProfileName) featureGateEnableOption {
	return func(s *featureGateStatus) {
		s.clusterProfile.Insert(clusterProfile)
	}
}

func withFeatureSet(featureSet configv1.FeatureSet) featureGateEnableOption {
	return func(s *featureGateStatus) {
		s.featureSets.Insert(featureSet)
	}
}

func inDefault() featureGateEnableOption {
	return withFeatureSet(configv1.Default)
}

func inTechPreviewNoUpgrade() featureGateEnableOption {
	return withFeatureSet(configv1.TechPreviewNoUpgrade)
}

func inDevPreviewNoUpgrade() featureGateEnableOption {
	return withFeatureSet(configv1.DevPreviewNoUpgrade)
}

func inCustomNoUpgrade() featureGateEnableOption {
	return withFeatureSet(configv1.CustomNoUpgrade)
}

func inOKD() featureGateEnableOption {
	return withFeatureSet(configv1.OKD)
}

type featureGateBuilder struct {
	name                string
	owningJiraComponent string
	responsiblePerson   string
	owningProduct       OwningProduct
	enhancementPRURL    string

	status []featureGateStatus
}
type featureGateStatus struct {
	version        sets.Set[uint64]
	clusterProfile sets.Set[ClusterProfileName]
	featureSets    sets.Set[configv1.FeatureSet]
}

func (s *featureGateStatus) isEnabled(version uint64, clusterProfile ClusterProfileName, featureSet configv1.FeatureSet) bool {
	// If either version or clusterprofile are empty, match all.
	matchesVersion := len(s.version) == 0 || s.version.Has(version)
	matchesClusterProfile := len(s.clusterProfile) == 0 || s.clusterProfile.Has(clusterProfile)

	matchesFeatureSet := s.featureSets.Has(featureSet)

	return matchesVersion && matchesClusterProfile && matchesFeatureSet
}

const (
	legacyFeatureGateWithoutEnhancement = "FeatureGate predates 4.18"
)

// newFeatureGate featuregate are disabled in every FeatureSet and selectively enabled
func newFeatureGate(name string) *featureGateBuilder {
	return &featureGateBuilder{
		name: name,
	}
}

func (b *featureGateBuilder) reportProblemsToJiraComponent(owningJiraComponent string) *featureGateBuilder {
	b.owningJiraComponent = owningJiraComponent
	return b
}

func (b *featureGateBuilder) contactPerson(responsiblePerson string) *featureGateBuilder {
	b.responsiblePerson = responsiblePerson
	return b
}

func (b *featureGateBuilder) productScope(owningProduct OwningProduct) *featureGateBuilder {
	b.owningProduct = owningProduct
	return b
}

func (b *featureGateBuilder) enhancementPR(url string) *featureGateBuilder {
	b.enhancementPRURL = url
	return b
}

func (b *featureGateBuilder) enable(opts ...featureGateEnableOption) *featureGateBuilder {
	status := featureGateStatus{
		version:        sets.New[uint64](),
		clusterProfile: sets.New[ClusterProfileName](),
		featureSets:    sets.New[configv1.FeatureSet](),
	}

	for _, opt := range opts {
		opt(&status)
	}

	b.status = append(b.status, status)

	return b
}

func (b *featureGateBuilder) register() (configv1.FeatureGateName, error) {
	if len(b.name) == 0 {
		return "", fmt.Errorf("missing name")
	}
	if len(b.owningJiraComponent) == 0 {
		return "", fmt.Errorf("missing owningJiraComponent")
	}
	if len(b.responsiblePerson) == 0 {
		return "", fmt.Errorf("missing responsiblePerson")
	}
	if len(b.owningProduct) == 0 {
		return "", fmt.Errorf("missing owningProduct")
	}
	_, enhancementPRErr := url.Parse(b.enhancementPRURL)
	switch {
	case b.enhancementPRURL == legacyFeatureGateWithoutEnhancement:
		if !legacyFeatureGates.Has(b.name) {
			return "", fmt.Errorf("FeatureGate/%s is a new feature gate, not an existing one.  It must have an enhancementPR with GA Graduation Criteria like https://github.com/openshift/enhancements/pull/#### or https://github.com/kubernetes/enhancements/issues/####", b.name)
		}

	case len(b.enhancementPRURL) == 0:
		return "", fmt.Errorf("FeatureGate/%s is missing an enhancementPR with GA Graduation Criteria like https://github.com/openshift/enhancements/pull/#### or https://github.com/kubernetes/enhancements/issues/####", b.name)

	case !strings.HasPrefix(b.enhancementPRURL, "https://github.com/openshift/enhancements/pull/") &&
		!strings.HasPrefix(b.enhancementPRURL, "https://github.com/kubernetes/enhancements/issues/") &&
		!strings.HasPrefix(b.enhancementPRURL, "https://github.com/ovn-kubernetes/ovn-kubernetes/pull/"):
		return "", fmt.Errorf("FeatureGate/%s enhancementPR format is incorrect; must be like https://github.com/openshift/enhancements/pull/#### or https://github.com/kubernetes/enhancements/issues/#### or https://github.com/ovn-kubernetes/ovn-kubernetes/pull/####", b.name)

	case enhancementPRErr != nil:
		return "", fmt.Errorf("FeatureGate/%s is enhancementPR is invalid: %w", b.name, enhancementPRErr)
	}

	featureGateName := configv1.FeatureGateName(b.name)

	allFeatureGates[featureGateName] = b.status

	return featureGateName, nil
}

func (b *featureGateBuilder) mustRegister() configv1.FeatureGateName {
	ret, err := b.register()
	if err != nil {
		panic(err)
	}
	return ret
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *FeatureGateEnabledDisabled) DeepCopyInto(out *FeatureGateEnabledDisabled) {
	*out = *in
	if in.Enabled != nil {
		in, out := &in.Enabled, &out.Enabled
		*out = make([]FeatureGateDescription, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
	if in.Disabled != nil {
		in, out := &in.Disabled, &out.Disabled
		*out = make([]FeatureGateDescription, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
	return
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new FeatureGateEnabledDisabled.
func (in *FeatureGateEnabledDisabled) DeepCopy() *FeatureGateEnabledDisabled {
	if in == nil {
		return nil
	}
	out := new(FeatureGateEnabledDisabled)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *FeatureGateDescription) DeepCopyInto(out *FeatureGateDescription) {
	*out = *in
	out.FeatureGateAttributes = in.FeatureGateAttributes
	return
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new FeatureGateDescription.
func (in *FeatureGateDescription) DeepCopy() *FeatureGateDescription {
	if in == nil {
		return nil
	}
	out := new(FeatureGateDescription)
	in.DeepCopyInto(out)
	return out
}
