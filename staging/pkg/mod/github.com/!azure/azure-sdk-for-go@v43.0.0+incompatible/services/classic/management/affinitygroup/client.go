// +build go1.7

package affinitygroup

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"encoding/base64"
	"encoding/xml"
	"fmt"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

const (
	azureCreateAffinityGroupURL = "/affinitygroups"
	azureGetAffinityGroupURL    = "/affinitygroups/%s"
	azureListAffinityGroupsURL  = "/affinitygroups"
	azureUpdateAffinityGroupURL = "/affinitygroups/%s"
	azureDeleteAffinityGroupURL = "/affinitygroups/%s"

	errParameterNotSpecified = "Parameter %s not specified."
)

// AffinityGroupClient simply contains a management.Client and has
// methods for doing all affinity group-related API calls to Azure.
type AffinityGroupClient struct {
	mgmtClient management.Client
}

// NewClient returns an AffinityGroupClient with the given management.Client.
func NewClient(mgmtClient management.Client) AffinityGroupClient {
	return AffinityGroupClient{mgmtClient}
}

// CreateAffinityGroup creates a new affinity group.
//
// https://msdn.microsoft.com/en-us/library/azure/gg715317.aspx
func (c AffinityGroupClient) CreateAffinityGroup(params CreateAffinityGroupParams) error {
	params.Label = encodeLabel(params.Label)

	req, err := xml.Marshal(params)
	if err != nil {
		return err
	}

	_, err = c.mgmtClient.SendAzurePostRequest(azureCreateAffinityGroupURL, req)
	return err
}

// GetAffinityGroup returns the system properties that are associated with the
// specified affinity group.
//
// https://msdn.microsoft.com/en-us/library/azure/ee460789.aspx
func (c AffinityGroupClient) GetAffinityGroup(name string) (AffinityGroup, error) {
	var affgroup AffinityGroup
	if name == "" {
		return affgroup, fmt.Errorf(errParameterNotSpecified, "name")
	}

	url := fmt.Sprintf(azureGetAffinityGroupURL, name)
	resp, err := c.mgmtClient.SendAzureGetRequest(url)
	if err != nil {
		return affgroup, err
	}

	err = xml.Unmarshal(resp, &affgroup)
	affgroup.Label = decodeLabel(affgroup.Label)
	return affgroup, err
}

// ListAffinityGroups lists the affinity groups off Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/ee460797.aspx
func (c AffinityGroupClient) ListAffinityGroups() (ListAffinityGroupsResponse, error) {
	var affinitygroups ListAffinityGroupsResponse

	resp, err := c.mgmtClient.SendAzureGetRequest(azureListAffinityGroupsURL)
	if err != nil {
		return affinitygroups, err
	}

	err = xml.Unmarshal(resp, &affinitygroups)

	for i, grp := range affinitygroups.AffinityGroups {
		affinitygroups.AffinityGroups[i].Label = decodeLabel(grp.Label)
	}

	return affinitygroups, err
}

// UpdateAffinityGroup updates the label or description for an the group.
//
// https://msdn.microsoft.com/en-us/library/azure/gg715316.aspx
func (c AffinityGroupClient) UpdateAffinityGroup(name string, params UpdateAffinityGroupParams) error {
	if name == "" {
		return fmt.Errorf(errParameterNotSpecified, "name")
	}

	params.Label = encodeLabel(params.Label)
	req, err := xml.Marshal(params)
	if err != nil {
		return err
	}

	url := fmt.Sprintf(azureUpdateAffinityGroupURL, name)
	_, err = c.mgmtClient.SendAzurePutRequest(url, "text/xml", req)
	return err
}

// DeleteAffinityGroup deletes the given affinity group.
//
// https://msdn.microsoft.com/en-us/library/azure/gg715314.aspx
func (c AffinityGroupClient) DeleteAffinityGroup(name string) error {
	if name == "" {
		return fmt.Errorf(errParameterNotSpecified, name)
	}

	url := fmt.Sprintf(azureDeleteAffinityGroupURL, name)
	_, err := c.mgmtClient.SendAzureDeleteRequest(url)
	return err
}

// encodeLabel is a helper function which encodes the given string
// to the base64 string which will be sent to Azure as a Label.
func encodeLabel(label string) string {
	return base64.StdEncoding.EncodeToString([]byte(label))
}

// decodeLabel is a helper function which decodes the base64 encoded
// label received from Azure into standard encoding.
func decodeLabel(label string) string {
	res, _ := base64.StdEncoding.DecodeString(label)
	return string(res)
}
