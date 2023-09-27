/*
Copyright 2022 The kcp Authors.

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
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"testing"
	"time"
)

func newBoundCRD(resource, group string) *crdBuilder {
	return &crdBuilder{
		curr: apiextensionsv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{
				Name: uuid.New().String(),
				Annotations: map[string]string{
					"apis.kcp.io/bound-crd": "",
				},
			},
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Group: group,
				Names: apiextensionsv1.CustomResourceDefinitionNames{
					Plural: resource,
				},
			},
		},
	}
}

func TestSync_KCP_BoundCRDsDoNotConflict(t *testing.T) {
	tests := []struct {
		name     string
		existing []*apiextensionsv1.CustomResourceDefinition
	}{
		{
			name: "conflict on plural to singular",
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newBoundCRD("india", "bravo.com").StatusNames("india", "alfa", "", "").NewOrDie(),
			},
		},
		{
			name: "conflict on singular to shortName",
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newBoundCRD("india", "bravo.com").StatusNames("india", "indias", "", "", "delta-singular").NewOrDie(),
			},
		},
		{
			name: "conflict on shortName to shortName",
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newBoundCRD("india", "bravo.com").StatusNames("india", "indias", "", "", "hotel-shortname-2").NewOrDie(),
			},
		},
		{
			name: "conflict on kind to listkind",
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newBoundCRD("india", "bravo.com").StatusNames("india", "indias", "", "echo-kind").NewOrDie(),
			},
		},
		{
			name: "conflict on listkind to kind",
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newBoundCRD("india", "bravo.com").StatusNames("india", "indias", "foxtrot-listkind", "").NewOrDie(),
			},
		},
		{
			name: "no conflict on resource and kind",
			existing: []*apiextensionsv1.CustomResourceDefinition{
				newBoundCRD("india", "bravo.com").StatusNames("india", "echo-kind", "", "").NewOrDie(),
			},
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

		newCRD := newBoundCRD("alfa", "bravo.com").SpecNames("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2").NewOrDie()

		expectedNames := names("alfa", "delta-singular", "echo-kind", "foxtrot-listkind", "golf-shortname-1", "hotel-shortname-2")

		actualNames, actualNameConflictCondition, establishedCondition := c.calculateNamesAndConditions(newCRD)

		require.Equal(t, expectedNames, actualNames, "calculated names mismatch")
		require.True(t, apiextensionshelpers.IsCRDConditionEquivalent(&acceptedCondition, &actualNameConflictCondition), "unexpected name conflict condition")
		require.True(t, apiextensionshelpers.IsCRDConditionEquivalent(&installingCondition, &establishedCondition), "unexpected established condition")
	}
}
