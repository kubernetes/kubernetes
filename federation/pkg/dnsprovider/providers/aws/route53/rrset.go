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

package route53

import (
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"

	"github.com/aws/aws-sdk-go/service/route53"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSet = ResourceRecordSet{}

type ResourceRecordSet struct {
	impl   *route53.ResourceRecordSet
	rrsets *ResourceRecordSets
}

func (rrset ResourceRecordSet) Name() string {
	return *rrset.impl.Name
}

func (rrset ResourceRecordSet) Rrdatas() []string {
	// Sigh - need to unpack the strings out of the route53 ResourceRecords
	result := make([]string, len(rrset.impl.ResourceRecords))
	for i, record := range rrset.impl.ResourceRecords {
		result[i] = *record.Value
	}
	return result
}

func (rrset ResourceRecordSet) Ttl() int64 {
	return *rrset.impl.TTL
}

func (rrset ResourceRecordSet) Type() rrstype.RrsType {
	return rrstype.RrsType(*rrset.impl.Type)
}
