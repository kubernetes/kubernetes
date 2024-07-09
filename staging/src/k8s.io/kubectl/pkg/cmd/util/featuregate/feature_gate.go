/*
Copyright 2024 The Kubernetes Authors.

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

package featuregate

import (
	"os"
	"strings"
)

type FeatureGate string

const (
	ApplySet                FeatureGate = "KUBECTL_APPLYSET"
	CmdPluginAsSubcommand   FeatureGate = "KUBECTL_ENABLE_CMD_SHADOW"
	OpenAPIV3Patch          FeatureGate = "KUBECTL_OPENAPIV3_PATCH"
	RemoteCommandWebsockets FeatureGate = "KUBECTL_REMOTE_COMMAND_WEBSOCKETS"
	PortForwardWebsockets   FeatureGate = "KUBECTL_PORT_FORWARD_WEBSOCKETS"
	DebugCustomProfile      FeatureGate = "KUBECTL_DEBUG_CUSTOM_PROFILE"
	WatchList               FeatureGate = "KUBECTL_WATCHLIST"
)

// IsEnabled returns true iff environment variable is set to true.
// All other cases, it returns false.
func (f FeatureGate) IsEnabled() bool {
	return strings.ToLower(os.Getenv(string(f))) == "true"
}

// IsDisabled returns true iff environment variable is set to false.
// All other cases, it returns true.
// This function is used for the cases where feature is enabled by default,
// but it may be needed to provide a way to ability to disable this feature.
func (f FeatureGate) IsDisabled() bool {
	return strings.ToLower(os.Getenv(string(f))) == "false"
}
