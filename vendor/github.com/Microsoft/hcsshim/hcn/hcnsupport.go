package hcn

import (
	"github.com/sirupsen/logrus"
)

// SupportedFeatures are the features provided by the Service.
type SupportedFeatures struct {
	Acl          AclFeatures `json:"ACL"`
	Api          ApiSupport  `json:"API"`
	RemoteSubnet bool        `json:"RemoteSubnet"`
	HostRoute    bool        `json:"HostRoute"`
	DSR          bool        `json:"DSR"`
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

	return features
}

func isFeatureSupported(currentVersion Version, minVersionSupported Version) bool {
	if currentVersion.Major < minVersionSupported.Major {
		return false
	}
	if currentVersion.Major > minVersionSupported.Major {
		return true
	}
	if currentVersion.Minor < minVersionSupported.Minor {
		return false
	}
	return true
}
