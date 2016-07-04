/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package concerto_cloud

import (
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// GetZone returns the Zone containing the current failure zone and locality
// region that the program is running in
func (concerto *ConcertoCloud) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{"concerto", "concerto"}, nil
}
