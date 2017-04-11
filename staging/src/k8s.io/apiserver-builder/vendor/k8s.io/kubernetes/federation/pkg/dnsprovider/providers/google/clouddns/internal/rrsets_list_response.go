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

package internal

import (
	dns "google.golang.org/api/dns/v1"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

// Compile time check for interface adherence
var _ interfaces.ResourceRecordSetsListResponse = &ResourceRecordSetsListResponse{}

type ResourceRecordSetsListResponse struct {
	impl *dns.ResourceRecordSetsListResponse
}

func (response *ResourceRecordSetsListResponse) Rrsets() []interfaces.ResourceRecordSet {
	rrsets := make([]interfaces.ResourceRecordSet, len(response.impl.Rrsets))
	for i, rrset := range response.impl.Rrsets {
		rrsets[i] = &ResourceRecordSet{rrset}
	}
	return rrsets

}
