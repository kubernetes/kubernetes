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
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
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

func (b *crdBuilder) Condition(c apiextensions.CustomResourceDefinitionCondition) *crdBuilder {
	b.curr.Status.Conditions = append(b.curr.Status.Conditions, c)

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

var acceptedCondition = apiextensions.CustomResourceDefinitionCondition{
	Type:    apiextensions.NamesAccepted,
	Status:  apiextensions.ConditionTrue,
	Reason:  "NoConflicts",
	Message: "no conflicts found",
}

var notAcceptedCondition = apiextensions.CustomResourceDefinitionCondition{
	Type:    apiextensions.NamesAccepted,
	Status:  apiextensions.ConditionFalse,
	Reason:  "NotAccepted",
	Message: "not all names are accepted",
}

var installingCondition = apiextensions.CustomResourceDefinitionCondition{
	Type:    apiextensions.Established,
	Status:  apiextensions.ConditionFalse,
	Reason:  "Installing",
	Message: "the initial names have been accepted",
}

var notEstablishedCondition = apiextensions.CustomResourceDefinitionCondition{
	Type:    apiextensions.Established,
	Status:  apiextensions.ConditionFalse,
	Reason:  "NotAccepted",
	Message: "not all names are accepted",
}

func nameConflictCondition(reason, message string) apiextensions.CustomResourceDefinitionCondition {
	return apiextensions.CustomResourceDefinitionCondition{
		Type:    apiextensions.NamesAccepted,
		Status:  apiextensions.ConditionFalse,
		Reason:  reason,
		Message: message,
	}
}

func TestSync(t *testing.T) {
	tests := []struct {
		name string

		in                            *apiextensions.CustomResourceDefinition
		existing                      []*apiextensions.CustomResourceDefinition
		expectedNames                 apiextensions.CustomResourceDefinitionNames
		expectedNameConflictCondition apiextensions.CustomResourceDefinitionCondition
		expectedEstablishedCondition  apiextensions.CustomResourceDefinitionCondition
	}{
		{
			name:     "first resource",
			in:       newCRD("alfa.bravo.com").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{},
			expectedNames: apiextensions.CustomResourceDefinitionNames{
				Plural: "alfa",
			},
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name: "different groups",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("alfa.charlie.com").StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name: "conflict plural to singular",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
			expectedNames:                 names("", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("PluralConflict", `"alfa" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict singular to shortName",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "", "delta-singular").NewOrDie(),
			},
			expectedNames:                 names("alfa", "", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("SingularConflict", `"delta-singular" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict on shortName to shortName",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "", "hotel-shortname-2").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind"),
			expectedNameConflictCondition: nameConflictCondition("ShortNamesConflict", `"hotel-shortname-2" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict on kind to listkind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "echo-kind").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("KindConflict", `"echo-kind" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict on listkind to kind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "foxtrot-listkind", "").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("ListKindConflict", `"foxtrot-listkind" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "no conflict on resource and kind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "echo-kind", "", "").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
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
			expectedNames:                 names("alfa", "yankee-singular", "echo-kind", "whiskey-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("ListKindConflict", `"foxtrot-listkind" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
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
			expectedNames:                 names("alfa", "yankee-singular", "echo-kind", "whiskey-listkind", "victor-shortname-1", "uniform-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("ListKindConflict", `"foxtrot-listkind" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
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
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
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
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1"),
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name:     "installing before with true condition",
			in:       newCRD("alfa.bravo.com").Condition(acceptedCondition).NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{},
			expectedNames: apiextensions.CustomResourceDefinitionNames{
				Plural: "alfa",
			},
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name:     "not installing before with false condition",
			in:       newCRD("alfa.bravo.com").Condition(notAcceptedCondition).NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{},
			expectedNames: apiextensions.CustomResourceDefinitionNames{
				Plural: "alfa",
			},
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name: "conflicting, installing before with true condition",
			in: newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				Condition(acceptedCondition).
				NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
			expectedNames:                 names("", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("PluralConflict", `"alfa" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflicting, not installing before with false condition",
			in: newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").
				Condition(notAcceptedCondition).
				NewOrDie(),
			existing: []*apiextensions.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
			expectedNames:                 names("", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("PluralConflict", `"alfa" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
	}

	for _, tc := range tests {
		crdIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		for _, obj := range tc.existing {
			crdIndexer.Add(obj)
		}

		c := NamingConditionController{
			crdLister:        listers.NewCustomResourceDefinitionLister(crdIndexer),
			crdMutationCache: cache.NewIntegerResourceVersionMutationCache(crdIndexer, crdIndexer, 60*time.Second, false),
		}
		actualNames, actualNameConflictCondition, establishedCondition := c.calculateNamesAndConditions(tc.in)

		if e, a := tc.expectedNames, actualNames; !reflect.DeepEqual(e, a) {
			t.Errorf("%v expected %v, got %#v", tc.name, e, a)
		}
		if e, a := tc.expectedNameConflictCondition, actualNameConflictCondition; !apiextensions.IsCRDConditionEquivalent(&e, &a) {
			t.Errorf("%v expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expectedEstablishedCondition, establishedCondition; !apiextensions.IsCRDConditionEquivalent(&e, &a) {
			t.Errorf("%v expected %v, got %v", tc.name, e, a)
		}
	}
}
