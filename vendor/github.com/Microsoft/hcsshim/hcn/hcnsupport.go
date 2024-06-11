//go:build windows

package hcn

import (
	"sync"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"

	"github.com/Microsoft/hcsshim/internal/log"
)

var (
	// featuresOnce handles assigning the supported features and printing the supported info to stdout only once to avoid unnecessary work
	// multiple times.
	featuresOnce      sync.Once
	featuresErr       error
	supportedFeatures SupportedFeatures
)

// SupportedFeatures are the features provided by the Service.
type SupportedFeatures struct {
	Acl                      AclFeatures `json:"ACL"`
	Api                      ApiSupport  `json:"API"`
	RemoteSubnet             bool        `json:"RemoteSubnet"`
	HostRoute                bool        `json:"HostRoute"`
	DSR                      bool        `json:"DSR"`
	Slash32EndpointPrefixes  bool        `json:"Slash32EndpointPrefixes"`
	AclSupportForProtocol252 bool        `json:"AclSupportForProtocol252"`
	SessionAffinity          bool        `json:"SessionAffinity"`
	IPv6DualStack            bool        `json:"IPv6DualStack"`
	SetPolicy                bool        `json:"SetPolicy"`
	VxlanPort                bool        `json:"VxlanPort"`
	L4Proxy                  bool        `json:"L4Proxy"`    // network policy that applies VFP rules to all endpoints on the network to redirect traffic
	L4WfpProxy               bool        `json:"L4WfpProxy"` // endpoint policy that applies WFP filters to redirect traffic to/from that endpoint
	TierAcl                  bool        `json:"TierAcl"`
	NetworkACL               bool        `json:"NetworkACL"`
	NestedIpSet              bool        `json:"NestedIpSet"`
	DisableHostPort          bool        `json:"DisableHostPort"`
	ModifyLoadbalancer       bool        `json:"ModifyLoadbalancer"`
}

// AclFeatures are the supported ACL possibilities.
type AclFeatures struct {
	AclAddressLists       bool `json:"AclAddressLists"`
	AclNoHostRulePriority bool `json:"AclHostRulePriority"`
	AclPortRanges         bool `json:"AclPortRanges"`
	AclRuleId             bool `json:"AclRuleId"`
}

// ApiSupport lists the supported API versions.
type ApiSupport struct {
	V1 bool `json:"V1"`
	V2 bool `json:"V2"`
}

// GetCachedSupportedFeatures returns the features supported by the Service and an error if the query failed. If this has been called
// before it will return the supported features and error received from the first call. This can be used to optimize if many calls to the
// various hcn.IsXSupported methods need to be made.
func GetCachedSupportedFeatures() (SupportedFeatures, error) {
	// Only query the HCN version and features supported once, instead of everytime this is invoked. The logs are useful to
	// debug incidents where there's confusion on if a feature is supported on the host machine. The sync.Once helps to avoid redundant
	// spam of these anytime a check needs to be made for if an HCN feature is supported. This is a common occurrence in kube-proxy
	// for example.
	featuresOnce.Do(func() {
		supportedFeatures, featuresErr = getSupportedFeatures()
	})

	return supportedFeatures, featuresErr
}

// GetSupportedFeatures returns the features supported by the Service.
//
// Deprecated: Use GetCachedSupportedFeatures instead.
func GetSupportedFeatures() SupportedFeatures {
	features, err := GetCachedSupportedFeatures()
	if err != nil {
		// Expected on pre-1803 builds, all features will be false/unsupported
		logrus.WithError(err).Errorf("unable to obtain supported features")
		return features
	}
	return features
}

func getSupportedFeatures() (SupportedFeatures, error) {
	var features SupportedFeatures
	globals, err := GetGlobals()
	if err != nil {
		// It's expected if this fails once, it should always fail. It should fail on pre 1803 builds for example.
		return SupportedFeatures{}, errors.Wrap(err, "failed to query HCN version number: this is expected on pre 1803 builds.")
	}
	features.Acl = AclFeatures{
		AclAddressLists:       isFeatureSupported(globals.Version, HNSVersion1803),
		AclNoHostRulePriority: isFeatureSupported(globals.Version, HNSVersion1803),
		AclPortRanges:         isFeatureSupported(globals.Version, HNSVersion1803),
		AclRuleId:             isFeatureSupported(globals.Version, HNSVersion1803),
	}

	features.Api = ApiSupport{
		V2: isFeatureSupported(globals.Version, V2ApiSupport),
		V1: true, // HNSCall is still available.
	}

	features.RemoteSubnet = isFeatureSupported(globals.Version, RemoteSubnetVersion)
	features.HostRoute = isFeatureSupported(globals.Version, HostRouteVersion)
	features.DSR = isFeatureSupported(globals.Version, DSRVersion)
	features.Slash32EndpointPrefixes = isFeatureSupported(globals.Version, Slash32EndpointPrefixesVersion)
	features.AclSupportForProtocol252 = isFeatureSupported(globals.Version, AclSupportForProtocol252Version)
	features.SessionAffinity = isFeatureSupported(globals.Version, SessionAffinityVersion)
	features.IPv6DualStack = isFeatureSupported(globals.Version, IPv6DualStackVersion)
	features.SetPolicy = isFeatureSupported(globals.Version, SetPolicyVersion)
	features.VxlanPort = isFeatureSupported(globals.Version, VxlanPortVersion)
	features.L4Proxy = isFeatureSupported(globals.Version, L4ProxyPolicyVersion)
	features.L4WfpProxy = isFeatureSupported(globals.Version, L4WfpProxyPolicyVersion)
	features.TierAcl = isFeatureSupported(globals.Version, TierAclPolicyVersion)
	features.NetworkACL = isFeatureSupported(globals.Version, NetworkACLPolicyVersion)
	features.NestedIpSet = isFeatureSupported(globals.Version, NestedIpSetVersion)
	features.DisableHostPort = isFeatureSupported(globals.Version, DisableHostPortVersion)
	features.ModifyLoadbalancer = isFeatureSupported(globals.Version, ModifyLoadbalancerVersion)

	log.L.WithFields(logrus.Fields{
		"version":           globals.Version,
		"supportedFeatures": features,
	}).Info("HCN feature check")

	return features, nil
}

func isFeatureSupported(currentVersion Version, versionsSupported VersionRanges) bool {
	isFeatureSupported := false

	for _, versionRange := range versionsSupported {
		isFeatureSupported = isFeatureSupported || isFeatureInRange(currentVersion, versionRange)
	}

	return isFeatureSupported
}

func isFeatureInRange(currentVersion Version, versionRange VersionRange) bool {
	if currentVersion.Major < versionRange.MinVersion.Major {
		return false
	}
	if currentVersion.Major > versionRange.MaxVersion.Major {
		return false
	}
	if currentVersion.Major == versionRange.MinVersion.Major && currentVersion.Minor < versionRange.MinVersion.Minor {
		return false
	}
	if currentVersion.Major == versionRange.MaxVersion.Major && currentVersion.Minor > versionRange.MaxVersion.Minor {
		return false
	}
	return true
}
