/*
Copyright 2019 The Kubernetes Authors.

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

package inventory

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/resid"
)

func makeRefs() (Refs, Refs) {
	a := resid.FromString("G1_V1_K1|ns1|nm1")
	b := resid.FromString("G2_V2_K2|ns2|nm2")
	c := resid.FromString("G3_V3_K3|ns3|nm3")
	current := NewRefs()
	current[a] = []resid.ResId{b, c}
	current[b] = []resid.ResId{}
	current[c] = []resid.ResId{}
	newRefs := NewRefs()
	newRefs[a] = []resid.ResId{b}
	newRefs[b] = []resid.ResId{}
	return current, newRefs
}

func TestInventory(t *testing.T) {
	inventory := NewInventory()
	curref, _ := makeRefs()

	inventory.UpdateCurrent(curref)
	if len(inventory.Current) != 3 {
		t.Fatalf("not getting the correct inventory %v", inventory)
	}
	curref, newref := makeRefs()
	inventory.UpdateCurrent(curref)
	if len(inventory.Current) != 3 {
		t.Fatalf("not getting the corrent inventory %v", inventory)
	}
	if len(inventory.Previous) != 3 {
		t.Fatalf("not getting the corrent inventory %v", inventory)
	}

	items := inventory.Prune()
	if len(items) != 0 {
		t.Fatalf("not getting the corrent items %v", items)
	}
	if len(inventory.Previous) != 0 {
		t.Fatalf("not getting the corrent inventory %v", inventory)
	}

	inventory.UpdateCurrent(newref)
	items = inventory.Prune()
	if len(items) != 1 {
		t.Fatalf("not getting the corrent items %v", items)
	}
	if len(inventory.Previous) != 0 {
		t.Fatalf("not getting the corrent inventory %v", inventory.Previous)
	}
}
