/*
Copyright 2016 The Kubernetes Authors.

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
	"io/ioutil"
	"net/http"
)

// This is just for tests injection
var metadataURL = "http://169.254.169.254/metadata"

// SetMetadataURLForTesting is used to modify the URL used for
// accessing the metadata server. Should only be used for testing!
func SetMetadataURLForTesting(url string) {
	metadataURL = url
}

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

// QueryMetadataJSON queries the metadata server and populates the passed in object
func QueryMetadataJSON(path string, obj interface{}) error {
	data, err := queryMetadataBytes(path, "json")
	if err != nil {
		return err
	}
	return json.Unmarshal(data, obj)
}

// QueryMetadataText queries the metadata server and returns the corresponding text
func QueryMetadataText(path string) (string, error) {
	data, err := queryMetadataBytes(path, "text")
	if err != nil {
		return "", err
	}
	return string(data), err
}

func queryMetadataBytes(path, format string) ([]byte, error) {
	client := &http.Client{}

	req, err := http.NewRequest("GET", metadataURL+path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Metadata", "True")

	q := req.URL.Query()
	q.Add("format", format)
	q.Add("api-version", "2017-04-02")
	req.URL.RawQuery = q.Encode()

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ioutil.ReadAll(resp.Body)
}
