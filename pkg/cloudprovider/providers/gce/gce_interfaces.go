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

package gce

import compute "google.golang.org/api/compute/v1"

// CloudAddressService is an interface for managing addresses
type CloudAddressService interface {
	ReserveRegionAddress(*compute.Address, string) error
	GetRegionAddress(string, string) (*compute.Address, error)
	// TODO: Mock `DeleteRegionAddress(name, region string) endpoint
	// TODO: Mock Global endpoints
}
