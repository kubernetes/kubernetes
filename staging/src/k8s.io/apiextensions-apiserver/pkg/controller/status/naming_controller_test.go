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

	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
)

type crdBuilder struct {
	curr apiextensionsv1.CustomResourceDefinition
}

func newCRD(name string) *crdBuilder {
	tokens := strings.SplitN(name, ".", 2)
	return &crdBuilder{
		curr: apiextensionsv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Group: tokens[1],
				Names: apiextensionsv1.CustomResourceDefinitionNames{
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

func (b *crdBuilder) Condition(c apiextensionsv1.CustomResourceDefinitionCondition) *crdBuilder {
	b.curr.Status.Conditions = append(b.curr.Status.Conditions, c)

	return b
}

func names(plural, singular, kind, listKind string, shortNames ...string) apiextensionsv1.CustomResourceDefinitionNames {
	ret := apiextensionsv1.CustomResourceDefinitionNames{
		Plural:     plural,
		Singular:   singular,
		Kind:       kind,
		ListKind:   listKind,
		ShortNames: shortNames,
	}
	return ret
}

func (b *crdBuilder) NewOrDie() *apiextensionsv1.CustomResourceDefinition {
	return &b.curr
}

var acceptedCondition = apiextensionsv1.CustomResourceDefinitionCondition{
	Type:    apiextensionsv1.NamesAccepted,
	Status:  apiextensionsv1.ConditionTrue,
	Reason:  "NoConflicts",
	Message: "no conflicts found",
}

var notAcceptedCondition = apiextensionsv1.CustomResourceDefinitionCondition{
	Type:    apiextensionsv1.NamesAccepted,
	Status:  apiextensionsv1.ConditionFalse,
	Reason:  "NotAccepted",
	Message: "not all names are accepted",
}

var installingCondition = apiextensionsv1.CustomResourceDefinitionCondition{
	Type:    apiextensionsv1.Established,
	Status:  apiextensionsv1.ConditionFalse,
	Reason:  "Installing",
	Message: "the initial names have been accepted",
}

var notEstablishedCondition = apiextensionsv1.CustomResourceDefinitionCondition{
	Type:    apiextensionsv1.Established,
	Status:  apiextensionsv1.ConditionFalse,
	Reason:  "NotAccepted",
	Message: "not all names are accepted",
}

func nameConflictCondition(reason, message string) apiextensionsv1.CustomResourceDefinitionCondition {
	return apiextensionsv1.CustomResourceDefinitionCondition{
		Type:    apiextensionsv1.NamesAccepted,
		Status:  apiextensionsv1.ConditionFalse,
		Reason:  reason,
		Message: message,
	}
}

func TestSync(t *testing.T) {
	tests := []struct {
		name string

		in                            *apiextensionsv1.CustomResourceDefinition
		existing                      []*apiextensionsv1.CustomResourceDefinition
		expectedNames                 apiextensionsv1.CustomResourceDefinitionNames
		expectedNameConflictCondition apiextensionsv1.CustomResourceDefinitionCondition
		expectedEstablishedCondition  apiextensionsv1.CustomResourceDefinitionCondition
	}{
		{
			name:     "first resource",
			in:       newCRD("alfa.bravo.com").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{},
			expectedNames: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "alfa",
			},
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name: "different groups",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("alfa.charlie.com").StatusNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name: "conflict plural to singular",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
			expectedNames:                 names("", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("PluralConflict", `"alfa" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict singular to shortName",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "", "delta-singular").NewOrDie(),
			},
			expectedNames:                 names("alfa", "", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("SingularConflict", `"delta-singular" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict on shortName to shortName",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "", "hotel-shortname-2").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind"),
			expectedNameConflictCondition: nameConflictCondition("ShortNamesConflict", `"hotel-shortname-2" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict on kind to listkind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "", "echo-kind").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("KindConflict", `"echo-kind" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "conflict on listkind to kind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "indias", "foxtrot-listkind", "").NewOrDie(),
			},
			expectedNames:                 names("alfa", "delta-singular", "echo-kind", "", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("ListKindConflict", `"foxtrot-listkind" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
		{
			name: "no conflict on resource and kind",
			in:   newCRD("alfa.bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{},
			expectedNames: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "alfa",
			},
			expectedNameConflictCondition: acceptedCondition,
			expectedEstablishedCondition:  installingCondition,
		},
		{
			name:     "not installing before with false condition",
			in:       newCRD("alfa.bravo.com").Condition(notAcceptedCondition).NewOrDie(),
			existing: []*apiextensionsv1.CustomResourceDefinition{},
			expectedNames: apiextensionsv1.CustomResourceDefinitionNames{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{
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
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newCRD("india.bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
			expectedNames:                 names("", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2"),
			expectedNameConflictCondition: nameConflictCondition("PluralConflict", `"alfa" is already in use`),
			expectedEstablishedCondition:  notEstablishedCondition,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			crdIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, obj := range tc.existing {
				if err := crdIndexer.Add(obj); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}

			c := NamingConditionController{
				crdLister:        listers.NewCustomResourceDefinitionLister(crdIndexer),
				crdMutationCache: cache.NewIntegerResourceVersionMutationCache(logger, crdIndexer, crdIndexer, 60*time.Second, false),
			}
			actualNames, actualNameConflictCondition, establishedCondition := c.calculateNamesAndConditions(tc.in)

			if e, a := tc.expectedNames, actualNames; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %#v", e, a)
			}
			if e, a := tc.expectedNameConflictCondition, actualNameConflictCondition; !apiextensionshelpers.IsCRDConditionEquivalent(&e, &a) {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := tc.expectedEstablishedCondition, establishedCondition; !apiextensionshelpers.IsCRDConditionEquivalent(&e, &a) {
				t.Errorf("expected %v, got %v", e, a)
			}
		})
	}
}
