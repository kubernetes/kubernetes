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

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/route53"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSet = ResourceRecordSet{}

type ResourceRecordSet struct {
	impl   *route53.ResourceRecordSet
	rrsets *ResourceRecordSets
}

func (rrset ResourceRecordSet) Name() string {
	return aws.StringValue(rrset.impl.Name)
}

func (rrset ResourceRecordSet) Rrdatas() []string {
	// Sigh - need to unpack the strings out of the route53 ResourceRecords
	result := make([]string, len(rrset.impl.ResourceRecords))
	for i, record := range rrset.impl.ResourceRecords {
		result[i] = aws.StringValue(record.Value)
	}
	return result
}

func (rrset ResourceRecordSet) Ttl() int64 {
	return aws.Int64Value(rrset.impl.TTL)
}

func (rrset ResourceRecordSet) Type() rrstype.RrsType {
	return rrstype.RrsType(aws.StringValue(rrset.impl.Type))
}

// Route53ResourceRecordSet returns the route53 ResourceRecordSet object for the ResourceRecordSet
// This is a "back door" that allows for limited access to the ResourceRecordSet,
// without having to requery it, so that we can expose AWS specific functionality.
// Using this method should be avoided where possible; instead prefer to add functionality
// to the cross-provider ResourceRecordSet interface.
func (rrset ResourceRecordSet) Route53ResourceRecordSet() *route53.ResourceRecordSet {
	return rrset.impl
}
