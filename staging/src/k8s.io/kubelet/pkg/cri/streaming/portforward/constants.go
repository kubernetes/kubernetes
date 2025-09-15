/*
Copyright 2015 The Kubernetes Authors.

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

// Package portforward contains server-side logic for handling port forwarding requests.
package portforward

// ProtocolV1Name is the name of the subprotocol used for port forwarding.
const ProtocolV1Name = "portforward.k8s.io"

// SupportedProtocols are the supported port forwarding protocols.
var SupportedProtocols = []string{ProtocolV1Name}
