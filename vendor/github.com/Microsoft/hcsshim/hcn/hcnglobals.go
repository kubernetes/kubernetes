package hcn

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

// Globals are all global properties of the HCN Service.
type Globals struct {
	Version Version `json:"Version"`
}

// Version is the HCN Service version.
type Version struct {
	Major int `json:"Major"`
	Minor int `json:"Minor"`
}

type VersionRange struct {
	MinVersion Version
	MaxVersion Version
}

type VersionRanges []VersionRange

var (
	// HNSVersion1803 added ACL functionality.
	HNSVersion1803 = VersionRanges{VersionRange{MinVersion: Version{Major: 7, Minor: 2}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
	// V2ApiSupport allows the use of V2 Api calls and V2 Schema.
	V2ApiSupport = VersionRanges{VersionRange{MinVersion: Version{Major: 9, Minor: 2}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
	// Remote Subnet allows for Remote Subnet policies on Overlay networks
	RemoteSubnetVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 9, Minor: 2}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
	// A Host Route policy allows for local container to local host communication Overlay networks
	HostRouteVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 9, Minor: 2}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
	// HNS 9.3 through 10.0 (not included), and 10.2+ allows for Direct Server Return for loadbalancing
	DSRVersion = VersionRanges{
		VersionRange{MinVersion: Version{Major: 9, Minor: 3}, MaxVersion: Version{Major: 9, Minor: math.MaxInt32}},
		VersionRange{MinVersion: Version{Major: 10, Minor: 2}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}},
	}
	// HNS 9.3 through 10.0 (not included) and, 10.4+ provide support for configuring endpoints with /32 prefixes
	Slash32EndpointPrefixesVersion = VersionRanges{
		VersionRange{MinVersion: Version{Major: 9, Minor: 3}, MaxVersion: Version{Major: 9, Minor: math.MaxInt32}},
		VersionRange{MinVersion: Version{Major: 10, Minor: 4}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}},
	}
	// HNS 9.3 through 10.0 (not included) and, 10.4+ allow for HNS ACL Policies to support protocol 252 for VXLAN
	AclSupportForProtocol252Version = VersionRanges{
		VersionRange{MinVersion: Version{Major: 11, Minor: 0}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}},
	}
	// HNS 12.0 allows for session affinity for loadbalancing
	SessionAffinityVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 12, Minor: 0}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
	// HNS 11.10+ supports Ipv6 dual stack.
	IPv6DualStackVersion = VersionRanges{
		VersionRange{MinVersion: Version{Major: 11, Minor: 10}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}},
	}
	// HNS 13.0 allows for Set Policy support
	SetPolicyVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 13, Minor: 0}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
	// HNS 10.3 allows for VXLAN ports
	VxlanPortVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 10, Minor: 3}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}

	//HNS 9.5 through 10.0(not included), 10.5 through 11.0(not included), 11.11 through 12.0(not included), 12.1 through 13.0(not included), 13.1+ allows for Network L4Proxy Policy support
	L4ProxyPolicyVersion = VersionRanges{
		VersionRange{MinVersion: Version{Major: 9, Minor: 5}, MaxVersion: Version{Major: 9, Minor: math.MaxInt32}},
		VersionRange{MinVersion: Version{Major: 10, Minor: 5}, MaxVersion: Version{Major: 10, Minor: math.MaxInt32}},
		VersionRange{MinVersion: Version{Major: 11, Minor: 11}, MaxVersion: Version{Major: 11, Minor: math.MaxInt32}},
		VersionRange{MinVersion: Version{Major: 12, Minor: 1}, MaxVersion: Version{Major: 12, Minor: math.MaxInt32}},
		VersionRange{MinVersion: Version{Major: 13, Minor: 1}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}},
	}

	//HNS 13.2 allows for L4WfpProxy Policy support
	L4WfpProxyPolicyVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 13, Minor: 2}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}

	//HNS 14.0 allows for TierAcl Policy support
	TierAclPolicyVersion = VersionRanges{VersionRange{MinVersion: Version{Major: 14, Minor: 0}, MaxVersion: Version{Major: math.MaxInt32, Minor: math.MaxInt32}}}
)

// GetGlobals returns the global properties of the HCN Service.
func GetGlobals() (*Globals, error) {
	var version Version
	err := hnsCall("GET", "/globals/version", "", &version)
	if err != nil {
		return nil, err
	}

	globals := &Globals{
		Version: version,
	}

	return globals, nil
}

type hnsResponse struct {
	Success bool
	Error   string
	Output  json.RawMessage
}

func hnsCall(method, path, request string, returnResponse interface{}) error {
	var responseBuffer *uint16
	logrus.Debugf("[%s]=>[%s] Request : %s", method, path, request)

	err := _hnsCall(method, path, request, &responseBuffer)
	if err != nil {
		return hcserror.New(err, "hnsCall ", "")
	}
	response := interop.ConvertAndFreeCoTaskMemString(responseBuffer)

	hnsresponse := &hnsResponse{}
	if err = json.Unmarshal([]byte(response), &hnsresponse); err != nil {
		return err
	}

	if !hnsresponse.Success {
		return fmt.Errorf("HNS failed with error : %s", hnsresponse.Error)
	}

	if len(hnsresponse.Output) == 0 {
		return nil
	}

	logrus.Debugf("Network Response : %s", hnsresponse.Output)
	err = json.Unmarshal(hnsresponse.Output, returnResponse)
	if err != nil {
		return err
	}

	return nil
}
