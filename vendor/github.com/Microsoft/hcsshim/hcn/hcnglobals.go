package hcn

import (
	"encoding/json"
	"fmt"

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

var (
	// HNSVersion1803 added ACL functionality.
	HNSVersion1803 = Version{Major: 7, Minor: 2}
	// V2ApiSupport allows the use of V2 Api calls and V2 Schema.
	V2ApiSupport = Version{Major: 9, Minor: 2}
	// Remote Subnet allows for Remote Subnet policies on Overlay networks
	RemoteSubnetVersion = Version{Major: 9, Minor: 2}
	// A Host Route policy allows for local container to local host communication Overlay networks
	HostRouteVersion = Version{Major: 9, Minor: 2}
	// HNS 10.2 allows for Direct Server Return for loadbalancing
	DSRVersion = Version{Major: 10, Minor: 2}
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
