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
	"fmt"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordChangeset = &ResourceRecordChangeset{}

type ResourceRecordChangeset struct {
	rrsets *ResourceRecordSets

	additions []dnsprovider.ResourceRecordSet
	removals  []dnsprovider.ResourceRecordSet
}

func (c *ResourceRecordChangeset) Add(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.additions = append(c.additions, rrset)
	return c
}

func (c *ResourceRecordChangeset) Remove(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.removals = append(c.removals, rrset)
	return c
}

func (c *ResourceRecordChangeset) Apply() error {
	rrsets := c.rrsets

	service := rrsets.zone.zones.interface_.service.Changes()
	var additions []interfaces.ResourceRecordSet
	for _, r := range c.additions {
		additions = append(additions, r.(ResourceRecordSet).impl)
	}
	var deletions []interfaces.ResourceRecordSet
	for _, r := range c.removals {
		deletions = append(deletions, r.(ResourceRecordSet).impl)
	}

	change := service.NewChange(additions, deletions)
	newChange, err := service.Create(rrsets.project(), rrsets.zone.impl.Name(), change).Do()
	if err != nil {
		return err
	}
	newAdditions := newChange.Additions()
	if len(newAdditions) != len(additions) {
		return fmt.Errorf("Internal error when adding resource record set.  Call succeeded but number of records returned is incorrect.  Records sent=%d, records returned=%d, additions:%v", len(additions), len(newAdditions), c.additions)
	}
	newDeletions := newChange.Deletions()
	if len(newDeletions) != len(deletions) {
		return fmt.Errorf("Internal error when deleting resource record set.  Call succeeded but number of records returned is incorrect.  Records sent=%d, records returned=%d, deletions:%v", len(deletions), len(newDeletions), c.removals)
	}

	return nil
}
