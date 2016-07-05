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

// Compile time check for interface adeherence
var _ interfaces.ResourceRecordSet = ResourceRecordSet{}

type ResourceRecordSet struct{ impl *dns.ResourceRecordSet }

func (r ResourceRecordSet) Name() string      { return r.impl.Name }
func (r ResourceRecordSet) Rrdatas() []string { return r.impl.Rrdatas }
func (r ResourceRecordSet) Ttl() int64        { return r.impl.Ttl }
func (r ResourceRecordSet) Type() string      { return r.impl.Type }
