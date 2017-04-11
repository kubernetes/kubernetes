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

package coredns

import (
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSet = ResourceRecordSet{}

type ResourceRecordSet struct {
	name    string
	rrdatas []string
	ttl     int64
	rrsType rrstype.RrsType
	rrsets  *ResourceRecordSets
}

func (rrset ResourceRecordSet) Name() string {
	return rrset.name
}

func (rrset ResourceRecordSet) Rrdatas() []string {
	return rrset.rrdatas
}

func (rrset ResourceRecordSet) Ttl() int64 {
	return rrset.ttl
}

func (rrset ResourceRecordSet) Type() rrstype.RrsType {
	return rrset.rrsType
}
