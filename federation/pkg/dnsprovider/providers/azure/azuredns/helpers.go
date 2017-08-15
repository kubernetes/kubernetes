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

package azuredns

import (
	"encoding/json"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
)

const (
	credentialsPath = "/.azure/credentials.json"
)

// ToJSON returns the passed item as a pretty-printed JSON string. If any JSON error occurs,
// it returns the empty string.
func ToJSON(v interface{}) (string, error) {
	j, err := json.MarshalIndent(v, "", "  ")
	return string(j), err
}

// NewServicePrincipalTokenFromCredentials creates a new ServicePrincipalToken using values of the
// passed credentials map.
// This implementation is "borrowed" from a later version of the azuresdk-for-go/arm/examples
func NewServicePrincipalTokenFromCredentials(config Config, scope string) (*adal.ServicePrincipalToken, error) {

	oauthConfig, err := adal.NewOAuthConfig(azure.PublicCloud.ActiveDirectoryEndpoint, config.Global.TenantID)
	if err != nil {
		panic(err)
	}

	return adal.NewServicePrincipalToken(*oauthConfig, config.Global.ClientID, config.Global.Secret, scope)
}
