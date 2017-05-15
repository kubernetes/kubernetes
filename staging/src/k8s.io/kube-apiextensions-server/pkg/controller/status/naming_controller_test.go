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

package status

import (
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	listers "k8s.io/kube-apiextensions-server/pkg/client/listers/apiextensions/internalversion"
)

type crdBuilder struct {
	curr apiextensions.CustomResourceDefinition
}

func newCRD(name string) *crdBuilder {
	tokens := strings.SplitN(name, ".", 2)
	return &crdBuilder{
		curr: apiextensions.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Group: tokens[1],
				Names: apiextensions.CustomResourceDefinitionNames{
					Plural: tokens[0],
				},
			},
		},
	}
}

func (b *crdBuilder) SpecNames(plural, singular, kind, listKind string, shortNames ...string) *crdBuilder {
	b.curr.Spec.Names.Plural = plural
	b.curr.Spec.Names.Singular = singular
	b.curr.Spec.Names.Kind = kind
	b.curr.Spec.Names.ListKind = listKind
	b.curr.Spec.Names.ShortNames = shortNames

	return b
}

func (b *crdBuilder) StatusNames(plural, singular, kind, listKind string, shortNames ...string) *crdBuilder {
	b.curr.Status.AcceptedNames.Plural = plural
	b.curr.Status.AcceptedNames.Singular = singular
	b.curr.Status.AcceptedNames.Kind = kind
	b.curr.Status.AcceptedNames.ListKind = listKind
	b.curr.Status.AcceptedNames.ShortNames = shortNames

	return b
}

func names(plural, singular, kind, listKind string, shortNames ...string) apiextensions.CustomResourceDefinitionNames {
	ret := apiextensions.CustomResourceDefinitionNames{
		Plural:     plural,
		Singular:   singular,
		Kind:       kind,
		ListKind:   listKind,
		ShortNames: shortNames,
	}
	return ret
}

func (b *crdBuilder) NewOrDie() *apiextensions.CustomResourceDefinition {
	return &b.curr
}

var goodCondition = apiextensions.CustomResourceDefinitionCondition{
	Type:    apiextensions.NameConflict,
	Status:  apiextensions.ConditionFalse,
	Reason:  "NoConflicts",
	Message: "no conflicts found",
}

func badCondition(reason, message string) apiextensions.CustomResourceDefinitionCondition {
	return apiextensions.CustomResourceDefinitionCondition{
		Type:    apiextensions.NameConflict,
		Status:  apiextensions.ConditionTrue,
		Reason:  reason,
		Message: message,
	}
}

func TestSync(t *testing.T) {
	tests := []struct {
		name string

		in                *apiextensions.CustomResourceDefinition
		existing          []*apiextensions.CustomResourceDefinition
		expectedNames     apiextensions.CustomResourceDefinitionNames
		expectedCondition apiextensions.CustomResourceDefinitionCondition
	}{
		{
			name:     "first resource",
			in:       newCRD("alfa.bravo.com").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{},
			expectedNames: apiextensions.CustomResourceDefinitionNames{
				Plural: "alfa",
			},
			expectedCondition: goodCondition,
		},
		{
			name: "different groups",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("alfa.charlie.com").StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: goodCondition,
		},
		{
			name: "conflict plural to singular",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
			expectedNames:     names("", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: badCondition("Plural", `"alfa" is already in use`),
		},
		{
			name: "conflict singular to shortName",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "", "delta-singular").NewOrDie(),
			},
			expectedNames:     names("alfa", "", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: badCondition("Singular", `"delta-singular" is already in use`),
		},
		{
			name: "conflict on shortName to shortName",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "", "hotel-shortname-2").NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind"),
			expectedCondition: badCondition("ShortNames", `"hotel-shortname-2" is already in use`),
		},
		{
			name: "conflict on kind to listkind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "echo-kind").NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: badCondition("Kind", `"echo-kind" is already in use`),
		},
		{
			name: "conflict on listkind to kind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "foxtrot-listkind", "").NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "echo-kind", "", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: badCondition("ListKind", `"foxtrot-listkind" is already in use`),
		},
		{
			name: "no conflict on resource and kind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "echo-kind", "", "").NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: goodCondition,
		},
		{
			name: "merge on conflicts",
			in: newCRD("alfa.bravo.com").
				SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				StatusNames("zulu", "yankee-singular", "xray-kind", "whiskey-listkind", "victor-shortname-1", "uniform-shortname-2").
				NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "foxtrot-listkind", "", "delta-singular").NewOrDie(),
			},
			expectedNames:     names("alfa", "yankee-singular", "echo-kind", "whiskey-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: badCondition("ListKind", `"foxtrot-listkind" is already in use`),
		},
		{
			name: "merge on conflicts shortNames as one",
			in: newCRD("alfa.bravo.com").
				SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				StatusNames("zulu", "yankee-singular", "xray-kind", "whiskey-listkind", "victor-shortname-1", "uniform-shortname-2").
				NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "foxtrot-listkind", "", "delta-singular", "golf-shortname-1").NewOrDie(),
			},
			expectedNames:     names("alfa", "yankee-singular", "echo-kind", "whiskey-listkind", "victor-shortname-1", "uniform-shortname-2"),
			expectedCondition: badCondition("ListKind", `"foxtrot-listkind" is already in use`),
		},
		{
			name: "no conflicts on self",
			in: newCRD("alfa.bravo.com").
				SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("alfa.bravo.com").
					SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
					StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
					NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedCondition: goodCondition,
		},
		{
			name: "no conflicts on self, remove shortname",
			in: newCRD("alfa.bravo.com").
				SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1").
				StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("alfa.bravo.com").
					SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
					StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
					NewOrDie(),
			},
			expectedNames:     names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1"),
			expectedCondition: goodCondition,
		},
	}

	for _, tc := range tests {
		crdIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		for _, obj := range tc.existing {
			crdIndexer.Add(obj)
		}

		c := NamingConditionController{
			crdLister:        listers.NewCustomResourceDefinitionLister(crdIndexer),
			crdMutationCache: cache.NewIntegerResourceVersionMutationCache(crdIndexer, crdIndexer),
		}
		actualNames, actualCondition := c.calculateNames(tc.in)

		if e, a := tc.expectedNames, actualNames; !reflect.DeepEqual(e, a) {
			t.Errorf("%v expected %v, got %#v", tc.name, e, a)
		}
		if e, a := tc.expectedCondition, actualCondition; !apiextensions.IsCRDConditionEquivalent(&e, &a) {
			t.Errorf("%v expected %v, got %v", tc.name, e, a)
		}
	}
}
