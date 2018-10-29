/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package azure

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

const (
	metadataCacheTTL = time.Minute
	metadataCacheKey = "InstanceMetadata"
	metadataURL      = "http://169.254.169.254/metadata/instance"
)

// NetworkMetadata contains metadata about an instance's network
type NetworkMetadata struct {
	Interface []NetworkInterface `json:"interface"`
}

// NetworkInterface represents an instances network interface.
type NetworkInterface struct {
	IPV4 NetworkData `json:"ipv4"`
	IPV6 NetworkData `json:"ipv6"`
	MAC  string      `json:"macAddress"`
}

// NetworkData contains IP information for a network.
type NetworkData struct {
	IPAddress []IPAddress `json:"ipAddress"`
	Subnet    []Subnet    `json:"subnet"`
}

// IPAddress represents IP address information.
type IPAddress struct {
	PrivateIP string `json:"privateIPAddress"`
	PublicIP  string `json:"publicIPAddress"`
}

// Subnet represents subnet information.
type Subnet struct {
	Address string `json:"address"`
	Prefix  string `json:"prefix"`
}

// ComputeMetadata represents compute information
type ComputeMetadata struct {
	SKU            string `json:"sku,omitempty"`
	Name           string `json:"name,omitempty"`
	Zone           string `json:"zone,omitempty"`
	VMSize         string `json:"vmSize,omitempty"`
	OSType         string `json:"osType,omitempty"`
	Location       string `json:"location,omitempty"`
	FaultDomain    string `json:"platformFaultDomain,omitempty"`
	UpdateDomain   string `json:"platformUpdateDomain,omitempty"`
	ResourceGroup  string `json:"resourceGroupName,omitempty"`
	VMScaleSetName string `json:"vmScaleSetName,omitempty"`
}

// InstanceMetadata represents instance information.
type InstanceMetadata struct {
	Compute *ComputeMetadata `json:"compute,omitempty"`
	Network *NetworkMetadata `json:"network,omitempty"`
}

// InstanceMetadataService knows how to query the Azure instance metadata server.
type InstanceMetadataService struct {
	metadataURL string
	imsCache    *timedCache
}

// NewInstanceMetadataService creates an instance of the InstanceMetadataService accessor object.
func NewInstanceMetadataService(metadataURL string) (*InstanceMetadataService, error) {
	ims := &InstanceMetadataService{
		metadataURL: metadataURL,
	}

	imsCache, err := newTimedcache(metadataCacheTTL, ims.getInstanceMetadata)
	if err != nil {
		return nil, err
	}

	ims.imsCache = imsCache
	return ims, nil
}

func (ims *InstanceMetadataService) getInstanceMetadata(key string) (interface{}, error) {
	req, err := http.NewRequest("GET", ims.metadataURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Metadata", "True")
	req.Header.Add("User-Agent", "golang/kubernetes-cloud-provider")

	q := req.URL.Query()
	q.Add("format", "json")
	q.Add("api-version", "2017-12-01")
	req.URL.RawQuery = q.Encode()

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("failure of getting instance metadata with response %q", resp.Status)
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	obj := InstanceMetadata{}
	err = json.Unmarshal(data, &obj)
	if err != nil {
		return nil, err
	}

	return &obj, nil
}

// GetMetadata gets instance metadata from cache.
func (ims *InstanceMetadataService) GetMetadata() (*InstanceMetadata, error) {
	cache, err := ims.imsCache.Get(metadataCacheKey)
	if err != nil {
		return nil, err
	}

	// Cache shouldn't be nil, but added a check incase something wrong.
	if cache == nil {
		return nil, fmt.Errorf("failure of getting instance metadata")
	}

	if metadata, ok := cache.(*InstanceMetadata); ok {
		return metadata, nil
	}

	return nil, fmt.Errorf("failure of getting instance metadata")
}
