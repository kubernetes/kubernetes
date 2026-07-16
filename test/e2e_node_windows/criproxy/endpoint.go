/*
Copyright The Kubernetes Authors.

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

package criproxy

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/rand"
)

const (
	// defaultNpipeEndpoint is the named-pipe endpoint template used for the CRI
	// proxy's gRPC server. Windows CRI endpoints are named pipes (npipe://),
	// not unix sockets, so the proxy listens on a uniquely-named pipe.
	defaultNpipeEndpoint = "npipe://./pipe/kubelet-remote-proxy-%v"
)

// GenerateEndpoint generates a new named-pipe endpoint for the proxy gRPC server.
func GenerateEndpoint() (string, error) {
	// use a random int as part of the pipe name to keep it unique
	return fmt.Sprintf(defaultNpipeEndpoint, rand.Int()), nil
}
