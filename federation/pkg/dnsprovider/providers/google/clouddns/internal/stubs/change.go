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

package stubs

import "k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"

// Compile time check for interface adeherence
var _ interfaces.Change = &Change{}

type Change struct {
	Service    *ChangesService
	Additions_ []interfaces.ResourceRecordSet
	Deletions_ []interfaces.ResourceRecordSet
}

func (c *Change) Additions() (rrsets []interfaces.ResourceRecordSet) {
	return c.Additions_
}

func (c *Change) Deletions() (rrsets []interfaces.ResourceRecordSet) {
	return c.Deletions_
}
