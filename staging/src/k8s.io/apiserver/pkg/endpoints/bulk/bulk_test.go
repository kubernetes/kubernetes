/*
Copyright 2017 The Kubernetes Authors.

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

package bulk

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	gvAv1 = schema.GroupVersion{"A", "v1"}
	gvBv1 = schema.GroupVersion{"B", "v1"}
	gvAv2 = schema.GroupVersion{"A", "v2"}
)

func TestBulkManagerFactoryNew(t *testing.T) {

	m := APIManagerFactory{}.New()
	agv1 := EnabledAPIGroupInfo{GroupVersion: gvAv1}

	var err error
	var ok bool

	err = m.RegisterAPIGroup(agv1, false)
	_, ok = m.APIGroups[gvAv1]
	if err != nil || !ok {
		t.Fatal("unable to register group", err)
	}

	err = m.RegisterAPIGroup(agv1, false)
	if err == nil {
		t.Fatal("expected error: group already registered")
	}

	agv2 := EnabledAPIGroupInfo{GroupVersion: gvAv2}
	err = m.RegisterAPIGroup(agv2, true)
	_, ok = m.APIGroups[gvAv2]
	if err != nil || !ok {
		t.Fatal("unable to register group", err)
	}
}

func TestBulkManagerFactoryDelegate(t *testing.T) {

	m1 := APIManagerFactory{}.New()
	agv1 := EnabledAPIGroupInfo{GroupVersion: gvAv1}
	agv2 := EnabledAPIGroupInfo{GroupVersion: gvAv2}

	if err := m1.RegisterAPIGroup(agv1, false); err != nil {
		t.Error("unable to register group")
	}
	if err := m1.RegisterAPIGroup(agv2, true); err != nil {
		t.Error("unable to register group")
	}

	m2 := APIManagerFactory{Delegate: m1}.New()

	if len(m2.APIGroups) != 2 {
		t.Fatal("delegated api groups are not installed")
	}
	if m2.PreferredVersion["A"] != "v2" {
		t.Fatal("delegated api groups are not installed")
	}

	agv3 := EnabledAPIGroupInfo{GroupVersion: gvBv1}
	if err := m2.RegisterAPIGroup(agv3, false); err != nil {
		t.Fatal("unable to register group")
	}
	if len(m2.APIGroups) != 3 {
		t.Fatal("unable to register group")
	}
}
