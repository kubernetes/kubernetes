/*
Copyright 2025 The Kubernetes Authors.

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

// Package imagepullprogress contains server-side logic for handling image pull progress requests.
package imagepullprogress

// ProtocolV1Name is the name of the subprotocol used for image pull progress.
const ProtocolV1Name = "imagepullprogress.k8s.io"

// SupportedProtocols are the supported image pull progress protocols.
var SupportedProtocols = []string{ProtocolV1Name}
