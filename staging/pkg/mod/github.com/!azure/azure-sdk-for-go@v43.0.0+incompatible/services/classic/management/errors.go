// +build go1.7

package management

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
	"fmt"
)

// AzureError represents an error returned by the management API. It has an error
// code (for example, ResourceNotFound) and a descriptive message.
type AzureError struct {
	Code    string
	Message string
}

//Error implements the error interface for the AzureError type.
func (e AzureError) Error() string {
	return fmt.Sprintf("Error response from Azure. Code: %s, Message: %s", e.Code, e.Message)
}

// IsResourceNotFoundError returns true if the provided error is an AzureError
// reporting that a given resource has not been found.
func IsResourceNotFoundError(err error) bool {
	azureErr, ok := err.(AzureError)
	return ok && azureErr.Code == "ResourceNotFound"
}

// getAzureError converts an error response body into an AzureError instance.
func getAzureError(responseBody []byte) error {
	var azErr AzureError
	err := xml.Unmarshal(responseBody, &azErr)
	if err != nil {
		return fmt.Errorf("Failed parsing contents to AzureError format: %v", err)
	}
	return azErr

}
