// +build go1.7

package virtualmachine

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
	"encoding/xml"
)

const (
	azureResourceExtensionsURL = "services/resourceextensions"
)

// GetResourceExtensions lists the resource extensions that are available to add
// to a virtual machine.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn495441.aspx
func (c VirtualMachineClient) GetResourceExtensions() (extensions []ResourceExtension, err error) {
	data, err := c.client.SendAzureGetRequest(azureResourceExtensionsURL)
	if err != nil {
		return extensions, err
	}

	var response ResourceExtensions
	err = xml.Unmarshal(data, &response)
	extensions = response.List
	return
}
