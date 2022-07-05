/*
Copyright 2021 The Kubernetes Authors.
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

package zoneclient

import (
	"context"

	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// APIVersion is the API version for provider list api.
	APIVersion = "2020-06-01"
)

// Interface is the client interface for ARM.
// Don't forget to run the following command to generate the mock client:
// mockgen -source=$GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/zoneclient/interface.go -package=mockzoneclient Interface > $GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/zoneclient/mockzoneclient/interface.go
type Interface interface {
	GetZones(ctx context.Context, subscriptionID string) (map[string][]string, *retry.Error)
}
