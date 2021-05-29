package hcn

import (
	"github.com/sirupsen/logrus"
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

// GetSupportedFeatures returns the features supported by the Service.
func GetSupportedFeatures() SupportedFeatures {
	var features SupportedFeatures

	globals, err := GetGlobals()
	if err != nil {
		// Expected on pre-1803 builds, all features will be false/unsupported
		logrus.Debugf("Unable to obtain globals: %s", err)
		return features
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

	return features
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
		logrus.Infof("currentVersion.Major < versionRange.MinVersion.Major: %v, %v", currentVersion.Major, versionRange.MinVersion.Major)
		return false
	}
	if currentVersion.Major > versionRange.MaxVersion.Major {
		logrus.Infof("currentVersion.Major > versionRange.MaxVersion.Major: %v, %v", currentVersion.Major, versionRange.MaxVersion.Major)
		return false
	}
	if currentVersion.Major == versionRange.MinVersion.Major && currentVersion.Minor < versionRange.MinVersion.Minor {
		logrus.Infof("currentVersion.Minor < versionRange.MinVersion.Major: %v, %v", currentVersion.Minor, versionRange.MinVersion.Minor)
		return false
	}
	if currentVersion.Major == versionRange.MaxVersion.Major && currentVersion.Minor > versionRange.MaxVersion.Minor {
		logrus.Infof("currentVersion.Minor > versionRange.MaxVersion.Major: %v, %v", currentVersion.Minor, versionRange.MaxVersion.Minor)
		return false
	}
	return true
}
