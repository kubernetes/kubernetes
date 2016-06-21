/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adeherence
var _ dnsprovider.ResourceRecordSets = ResourceRecordSets{}

type ResourceRecordSets struct {
	zone *Zone
	impl interfaces.ResourceRecordSetsService
}

func (rrsets ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	response, err := rrsets.impl.List(rrsets.project(), rrsets.zone.impl.Name()).Do()
	if err != nil {
		return nil, err
	}
	list := make([]dnsprovider.ResourceRecordSet, len(response.Rrsets()))
	for i, rrset := range response.Rrsets() {
		list[i] = ResourceRecordSet{rrset, &rrsets}
	}
	return list, nil
}

func (rrsets ResourceRecordSets) Add(rrset dnsprovider.ResourceRecordSet) (dnsprovider.ResourceRecordSet, error) {
	service := rrsets.zone.zones.interface_.service.Changes()
	additions := []interfaces.ResourceRecordSet{rrset.(*ResourceRecordSet).impl}
	change := service.NewChange(additions, []interfaces.ResourceRecordSet{})
	newChange, err := service.Create(rrsets.project(), rrsets.zone.impl.Name(), change).Do()
	if err != nil {
		return nil, err
	}
	newAdditions := newChange.Additions()
	if len(newAdditions) != len(additions) {
		return nil, fmt.Errorf("Internal error when adding resource record set.  Call succeeded but number of records returned is incorrect.  Records sent=%d, records returned=%d, record set:%v", len(additions), len(newAdditions), rrset)
	}
	return ResourceRecordSet{newChange.Additions()[0], &rrsets}, nil
}

func (rrsets ResourceRecordSets) Remove(rrset dnsprovider.ResourceRecordSet) error {
	service := rrsets.zone.zones.interface_.service.Changes()
	deletions := []interfaces.ResourceRecordSet{rrset.(ResourceRecordSet).impl}
	change := service.NewChange([]interfaces.ResourceRecordSet{}, deletions)
	newChange, err := service.Create(rrsets.project(), rrsets.zone.impl.Name(), change).Do()
	if err != nil {
		return err
	}
	newDeletions := newChange.Deletions()
	if len(newDeletions) != len(deletions) {
		return fmt.Errorf("Internal error when deleting resource record set.  Call succeeded but number of records returned is incorrect.  Records sent=%d, records returned=%d, record set:%v", len(deletions), len(newDeletions), rrset)
	}
	return nil
}

func (rrsets ResourceRecordSets) New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) dnsprovider.ResourceRecordSet {
	return &ResourceRecordSet{rrsets.impl.NewResourceRecordSet(name, rrdatas, ttl, rrstype), &rrsets}
}

func (rrsets ResourceRecordSets) project() string {
	return rrsets.zone.project()
}
