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
var _ interfaces.ChangesService = ChangesService{}

type ChangesService struct{ impl *dns.ChangesService }

func (c ChangesService) Create(project string, managedZone string, change interfaces.Change) interfaces.ChangesCreateCall {
	return &ChangesCreateCall{c.impl.Create(project, managedZone, change.(*Change).impl)}
}

func (c ChangesService) NewChange(additions, deletions []interfaces.ResourceRecordSet) interfaces.Change {
	adds := make([]*dns.ResourceRecordSet, len(additions))
	deletes := make([]*dns.ResourceRecordSet, len(deletions))
	for i, a := range additions {
		adds[i] = a.(*ResourceRecordSet).impl
	}
	for i, d := range deletions {
		deletes[i] = d.(*ResourceRecordSet).impl
	}
	return &Change{&dns.Change{Additions: adds, Deletions: deletes}}
}
