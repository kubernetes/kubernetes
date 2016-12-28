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
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSet = ResourceRecordSet{}

type ResourceRecordSet struct {
	impl   interfaces.ResourceRecordSet
	rrsets *ResourceRecordSets
}

func (rrset ResourceRecordSet) String() string {
	return fmt.Sprintf("<(clouddns) %q type=%s rrdatas=%q ttl=%v>", rrset.Name(), rrset.Type(), rrset.Rrdatas(), rrset.Ttl())
}

func (rrset ResourceRecordSet) Name() string {
	return rrset.impl.Name()
}

func (rrset ResourceRecordSet) Rrdatas() []string {
	return rrset.impl.Rrdatas()
}

func (rrset ResourceRecordSet) Ttl() int64 {
	return rrset.impl.Ttl()
}

func (rrset ResourceRecordSet) Type() rrstype.RrsType {
	return rrstype.RrsType(rrset.impl.Type())
}
