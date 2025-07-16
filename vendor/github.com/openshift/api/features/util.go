package features

import (
	"fmt"
	configv1 "github.com/openshift/api/config/v1"
	"net/url"
	"strings"
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

type featureGateBuilder struct {
	name                string
	owningJiraComponent string
	responsiblePerson   string
	owningProduct       OwningProduct
	enhancementPRURL    string

	statusByClusterProfileByFeatureSet map[ClusterProfileName]map[configv1.FeatureSet]bool
}

const (
	legacyFeatureGateWithoutEnhancement = "FeatureGate predates 4.18"
)

// newFeatureGate featuregate are disabled in every FeatureSet and selectively enabled
func newFeatureGate(name string) *featureGateBuilder {
	b := &featureGateBuilder{
		name:                               name,
		statusByClusterProfileByFeatureSet: map[ClusterProfileName]map[configv1.FeatureSet]bool{},
	}
	for _, clusterProfile := range AllClusterProfiles {
		byFeatureSet := map[configv1.FeatureSet]bool{}
		for _, featureSet := range configv1.AllFixedFeatureSets {
			byFeatureSet[featureSet] = false
		}
		b.statusByClusterProfileByFeatureSet[clusterProfile] = byFeatureSet
	}
	return b
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

func (b *featureGateBuilder) enableIn(featureSets ...configv1.FeatureSet) *featureGateBuilder {
	for clusterProfile := range b.statusByClusterProfileByFeatureSet {
		for _, featureSet := range featureSets {
			b.statusByClusterProfileByFeatureSet[clusterProfile][featureSet] = true
		}
	}
	return b
}

func (b *featureGateBuilder) enableForClusterProfile(clusterProfile ClusterProfileName, featureSets ...configv1.FeatureSet) *featureGateBuilder {
	for _, featureSet := range featureSets {
		b.statusByClusterProfileByFeatureSet[clusterProfile][featureSet] = true
	}
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

	case !strings.HasPrefix(b.enhancementPRURL, "https://github.com/openshift/enhancements/pull/") && !strings.HasPrefix(b.enhancementPRURL, "https://github.com/kubernetes/enhancements/issues/"):
		return "", fmt.Errorf("FeatureGate/%s enhancementPR format is incorrect; must be like https://github.com/openshift/enhancements/pull/#### or https://github.com/kubernetes/enhancements/issues/####", b.name)

	case enhancementPRErr != nil:
		return "", fmt.Errorf("FeatureGate/%s is enhancementPR is invalid: %w", b.name, enhancementPRErr)
	}

	featureGateName := configv1.FeatureGateName(b.name)
	description := FeatureGateDescription{
		FeatureGateAttributes: configv1.FeatureGateAttributes{
			Name: featureGateName,
		},
		OwningJiraComponent: b.owningJiraComponent,
		ResponsiblePerson:   b.responsiblePerson,
		OwningProduct:       b.owningProduct,
		EnhancementPR:       b.enhancementPRURL,
	}

	// statusByClusterProfileByFeatureSet is initialized by constructor to be false for every combination
	for clusterProfile, byFeatureSet := range b.statusByClusterProfileByFeatureSet {
		for featureSet, enabled := range byFeatureSet {
			if _, ok := allFeatureGates[clusterProfile]; !ok {
				allFeatureGates[clusterProfile] = map[configv1.FeatureSet]*FeatureGateEnabledDisabled{}
			}
			if _, ok := allFeatureGates[clusterProfile][featureSet]; !ok {
				allFeatureGates[clusterProfile][featureSet] = &FeatureGateEnabledDisabled{}
			}

			if enabled {
				allFeatureGates[clusterProfile][featureSet].Enabled = append(allFeatureGates[clusterProfile][featureSet].Enabled, description)
			} else {
				allFeatureGates[clusterProfile][featureSet].Disabled = append(allFeatureGates[clusterProfile][featureSet].Disabled, description)
			}
		}
	}

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
		copy(*out, *in)
	}
	if in.Disabled != nil {
		in, out := &in.Disabled, &out.Disabled
		*out = make([]FeatureGateDescription, len(*in))
		copy(*out, *in)
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
