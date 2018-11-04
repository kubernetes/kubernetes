/*
Copyright 2017 The Kubernetes Authors.

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

package modes

import "k8s.io/apimachinery/pkg/util/sets"

const (
	// ModeAlwaysAllow results in all requests being allowed.
	ModeAlwaysAllow string = "AlwaysAllow"
	// ModeAlwaysDeny results in all requests being blocked.
	ModeAlwaysDeny string = "AlwaysDeny"
	// ModeABAC enables policies to be configured using local files.
	ModeABAC string = "ABAC"
	// ModeWebhook enables authorization to be managed using a remote REST endpoint.
	ModeWebhook string = "Webhook"
	// ModeRBAC enables the creation and storage of policies using the Kubernetes API.
	ModeRBAC string = "RBAC"
	// ModeNode enables node authorization to authorize requests.
	ModeNode string = "Node"
)

// AuthorizationModeChoices is a list of possible authorization options.
var AuthorizationModeChoices = []string{ModeAlwaysAllow, ModeAlwaysDeny, ModeABAC, ModeWebhook, ModeRBAC, ModeNode}

// IsValidAuthorizationMode returns true if the given authorization mode is a valid one for the apiserver
func IsValidAuthorizationMode(authzMode string) bool {
	return sets.NewString(AuthorizationModeChoices...).Has(authzMode)
}
