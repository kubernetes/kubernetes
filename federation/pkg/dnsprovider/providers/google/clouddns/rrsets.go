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

package clouddns

import (
	"context"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSets = ResourceRecordSets{}

type ResourceRecordSets struct {
	zone *Zone
	impl interfaces.ResourceRecordSetsService
}

// List returns a list of resource records in the given project and
// managed zone.
// !!CAUTION!! Your memory might explode if you have a huge number of
// records in your managed zone.
func (rrsets ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	var list []dnsprovider.ResourceRecordSet

	ctx := context.Background()

	call := rrsets.impl.List(rrsets.project(), rrsets.zone.impl.Name())
	err := call.Pages(ctx, func(page interfaces.ResourceRecordSetsListResponse) error {
		for _, rrset := range page.Rrsets() {
			list = append(list, ResourceRecordSet{rrset, &rrsets})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return list, nil
}

func (rrsets ResourceRecordSets) Get(name string) ([]dnsprovider.ResourceRecordSet, error) {
	var list []dnsprovider.ResourceRecordSet

	ctx := context.Background()

	call := rrsets.impl.Get(rrsets.project(), rrsets.zone.impl.Name(), name)
	err := call.Pages(ctx, func(page interfaces.ResourceRecordSetsListResponse) error {
		for _, rrset := range page.Rrsets() {
			list = append(list, ResourceRecordSet{rrset, &rrsets})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return list, nil
}

func (r ResourceRecordSets) StartChangeset() dnsprovider.ResourceRecordChangeset {
	return &ResourceRecordChangeset{
		rrsets: &r,
	}
}

func (r ResourceRecordSets) New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) dnsprovider.ResourceRecordSet {
	return ResourceRecordSet{r.impl.NewResourceRecordSet(name, rrdatas, ttl, rrstype), &r}
}

func (rrsets ResourceRecordSets) project() string {
	return rrsets.zone.project()
}

// Zone returns the parent zone
func (rrset ResourceRecordSets) Zone() dnsprovider.Zone {
	return rrset.zone
}
