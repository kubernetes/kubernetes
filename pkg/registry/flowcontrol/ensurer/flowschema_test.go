/*
Copyright 2021 The Kubernetes Authors.

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

package ensurer

import (
	"context"
	"reflect"
	"testing"

	flowcontrolv1beta2 "k8s.io/api/flowcontrol/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/client-go/kubernetes/fake"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta2"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1beta2"
	"k8s.io/client-go/tools/cache"
	flowcontrolapisv1beta2 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta2"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
)

func TestEnsureFlowSchema(t *testing.T) {
	tests := []struct {
		name      string
		strategy  func(flowcontrolclient.FlowSchemaInterface, flowcontrollisters.FlowSchemaLister) FlowSchemaEnsurer
		current   *flowcontrolv1beta2.FlowSchema
		bootstrap *flowcontrolv1beta2.FlowSchema
		expected  *flowcontrolv1beta2.FlowSchema
	}{
		// for suggested configurations
		{
			name:      "suggested flow schema does not exist - the object should always be re-created",
			strategy:  NewSuggestedFlowSchemaEnsurer,
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   nil,
			expected:  newFlowSchema("fs1", "pl1", 100).Object(),
		},
		{
			name:      "suggested flow schema exists, auto update is enabled, spec does not match - current object should be updated",
			strategy:  NewSuggestedFlowSchemaEnsurer,
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("true").Object(),
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "suggested flow schema exists, auto update is disabled, spec does not match - current object should not be updated",
			strategy:  NewSuggestedFlowSchemaEnsurer,
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			expected:  newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
		},

		// for mandatory configurations
		{
			name:      "mandatory flow schema does not exist - new object should be created",
			strategy:  NewMandatoryFlowSchemaEnsurer,
			bootstrap: newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
			current:   nil,
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "mandatory flow schema exists, annotation is missing - annotation should be added",
			strategy:  NewMandatoryFlowSchemaEnsurer,
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 100).Object(),
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "mandatory flow schema exists, auto update is disabled, spec does not match - current object should be updated",
			strategy:  NewMandatoryFlowSchemaEnsurer,
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset().FlowcontrolV1beta2().FlowSchemas()
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if test.current != nil {
				client.Create(context.TODO(), test.current, metav1.CreateOptions{})
				indexer.Add(test.current)
			}

			ensurer := test.strategy(client, flowcontrollisters.NewFlowSchemaLister(indexer))

			err := ensurer.Ensure([]*flowcontrolv1beta2.FlowSchema{test.bootstrap})
			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
			}

			fsGot, err := client.Get(context.TODO(), test.bootstrap.Name, metav1.GetOptions{})
			switch {
			case test.expected == nil:
				if !apierrors.IsNotFound(err) {
					t.Fatalf("Expected GET to return an %q error, but got: %v", metav1.StatusReasonNotFound, err)
				}
			case err != nil:
				t.Fatalf("Expected GET to return no error, but got: %v", err)
			}

			if !reflect.DeepEqual(test.expected, fsGot) {
				t.Errorf("FlowSchema does not match - diff: %s", cmp.Diff(test.expected, fsGot))
			}
		})
	}
}

func TestSuggestedFSEnsureStrategy_ShouldUpdate(t *testing.T) {
	tests := []struct {
		name              string
		current           *flowcontrolv1beta2.FlowSchema
		bootstrap         *flowcontrolv1beta2.FlowSchema
		newObjectExpected *flowcontrolv1beta2.FlowSchema
	}{
		{
			name:              "auto update is enabled, first generation, spec does not match - spec update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").WithGeneration(1).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 200).Object(),
			newObjectExpected: newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("true").WithGeneration(1).Object(),
		},
		{
			name:              "auto update is enabled, first generation, spec matches - no update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").WithGeneration(1).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 100).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is enabled, second generation, spec does not match - spec update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").WithGeneration(2).Object(),
			bootstrap:         newFlowSchema("fs1", "pl2", 200).Object(),
			newObjectExpected: newFlowSchema("fs1", "pl2", 200).WithAutoUpdateAnnotation("true").WithGeneration(2).Object(),
		},
		{
			name:              "auto update is enabled, second generation, spec matches - no update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").WithGeneration(2).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 100).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, first generation, spec does not match - no update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("false").WithGeneration(1).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 200).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, first generation, spec matches - no update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("false").WithGeneration(1).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 100).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, second generation, spec does not match - no update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("false").WithGeneration(2).Object(),
			bootstrap:         newFlowSchema("fs1", "pl2", 200).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, second generation, spec matches - no update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("false").WithGeneration(2).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 100).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "annotation is missing, first generation, spec does not match - both annotation and spec update expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithGeneration(1).Object(),
			bootstrap:         newFlowSchema("fs1", "pl2", 200).Object(),
			newObjectExpected: newFlowSchema("fs1", "pl2", 200).WithAutoUpdateAnnotation("true").WithGeneration(1).Object(),
		},
		{
			name:              "annotation is missing, first generation, spec matches - annotation update is expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithGeneration(1).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 100).Object(),
			newObjectExpected: newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").WithGeneration(1).Object(),
		},
		{
			name:              "annotation is missing, second generation, spec does not match - annotation update is expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithGeneration(2).Object(),
			bootstrap:         newFlowSchema("fs1", "pl2", 200).Object(),
			newObjectExpected: newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("false").WithGeneration(2).Object(),
		},
		{
			name:              "annotation is missing, second generation, spec matches - annotation update is expected",
			current:           newFlowSchema("fs1", "pl1", 100).WithGeneration(2).Object(),
			bootstrap:         newFlowSchema("fs1", "pl1", 100).Object(),
			newObjectExpected: newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("false").WithGeneration(2).Object(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			strategy := newSuggestedEnsureStrategy(&flowSchemaWrapper{})
			newObjectGot, updateGot, err := strategy.ShouldUpdate(test.current, test.bootstrap)
			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}

			if test.newObjectExpected == nil {
				if newObjectGot != nil {
					t.Errorf("Expected a nil object, but got: %#v", newObjectGot)
				}
				if updateGot {
					t.Errorf("Expected update=%t but got: %t", false, updateGot)
				}
				return
			}

			if !updateGot {
				t.Errorf("Expected update=%t but got: %t", true, updateGot)
			}
			if !reflect.DeepEqual(test.newObjectExpected, newObjectGot) {
				t.Errorf("Expected the object to be updated to match - diff: %s", cmp.Diff(test.newObjectExpected, newObjectGot))
			}
		})
	}
}

func TestFlowSchemaSpecChanged(t *testing.T) {
	fs1 := &flowcontrolv1beta2.FlowSchema{
		Spec: flowcontrolv1beta2.FlowSchemaSpec{},
	}
	fs2 := &flowcontrolv1beta2.FlowSchema{
		Spec: flowcontrolv1beta2.FlowSchemaSpec{
			MatchingPrecedence: 1,
		},
	}
	fs1Defaulted := &flowcontrolv1beta2.FlowSchema{
		Spec: flowcontrolv1beta2.FlowSchemaSpec{
			MatchingPrecedence: flowcontrolapisv1beta2.FlowSchemaDefaultMatchingPrecedence,
		},
	}
	testCases := []struct {
		name        string
		expected    *flowcontrolv1beta2.FlowSchema
		actual      *flowcontrolv1beta2.FlowSchema
		specChanged bool
	}{
		{
			name:        "identical flow-schemas should work",
			expected:    bootstrap.MandatoryFlowSchemaCatchAll,
			actual:      bootstrap.MandatoryFlowSchemaCatchAll,
			specChanged: false,
		},
		{
			name:        "defaulted flow-schemas should work",
			expected:    fs1,
			actual:      fs1Defaulted,
			specChanged: false,
		},
		{
			name:        "non-defaulted flow-schema has wrong spec",
			expected:    fs1,
			actual:      fs2,
			specChanged: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			w := flowSchemaSpecChanged(testCase.expected, testCase.actual)
			assert.Equal(t, testCase.specChanged, w)
		})
	}
}

func TestRemoveFlowSchema(t *testing.T) {
	tests := []struct {
		name           string
		current        *flowcontrolv1beta2.FlowSchema
		bootstrapName  string
		removeExpected bool
	}{
		{
			name:          "flow schema does not exist",
			bootstrapName: "fs1",
			current:       nil,
		},
		{
			name:           "flow schema exists, auto update is enabled",
			bootstrapName:  "fs1",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("true").Object(),
			removeExpected: true,
		},
		{
			name:           "flow schema exists, auto update is disabled",
			bootstrapName:  "fs1",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			removeExpected: false,
		},
		{
			name:           "flow schema exists, the auto-update annotation is malformed",
			bootstrapName:  "fs1",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("invalid").Object(),
			removeExpected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset().FlowcontrolV1beta2().FlowSchemas()
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if test.current != nil {
				client.Create(context.TODO(), test.current, metav1.CreateOptions{})
				indexer.Add(test.current)
			}

			remover := NewFlowSchemaRemover(client, flowcontrollisters.NewFlowSchemaLister(indexer))
			err := remover.RemoveAutoUpdateEnabledObjects([]string{test.bootstrapName})
			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
			}

			if test.current == nil {
				return
			}
			_, err = client.Get(context.TODO(), test.bootstrapName, metav1.GetOptions{})
			switch {
			case test.removeExpected:
				if !apierrors.IsNotFound(err) {
					t.Errorf("Expected error: %q, but got: %v", metav1.StatusReasonNotFound, err)
				}
			default:
				if err != nil {
					t.Errorf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestGetFlowSchemaRemoveCandidate(t *testing.T) {
	tests := []struct {
		name      string
		current   []*flowcontrolv1beta2.FlowSchema
		bootstrap []*flowcontrolv1beta2.FlowSchema
		expected  []string
	}{
		{
			name: "no object has been removed from the bootstrap configuration",
			bootstrap: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs2", "pl2", 200).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs3", "pl3", 300).WithAutoUpdateAnnotation("true").Object(),
			},
			current: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs2", "pl2", 200).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs3", "pl3", 300).WithAutoUpdateAnnotation("true").Object(),
			},
			expected: []string{},
		},
		{
			name:      "bootstrap is empty, all current objects with the annotation should be candidates",
			bootstrap: []*flowcontrolv1beta2.FlowSchema{},
			current: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs2", "pl2", 200).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs3", "pl3", 300).Object(),
			},
			expected: []string{"fs1", "fs2"},
		},
		{
			name: "object(s) have been removed from the bootstrap configuration",
			bootstrap: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
			},
			current: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs2", "pl2", 200).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs3", "pl3", 300).WithAutoUpdateAnnotation("true").Object(),
			},
			expected: []string{"fs2", "fs3"},
		},
		{
			name: "object(s) without the annotation key are ignored",
			bootstrap: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
			},
			current: []*flowcontrolv1beta2.FlowSchema{
				newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
				newFlowSchema("fs2", "pl2", 200).Object(),
				newFlowSchema("fs3", "pl3", 300).Object(),
			},
			expected: []string{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			for i := range test.current {
				indexer.Add(test.current[i])
			}

			lister := flowcontrollisters.NewFlowSchemaLister(indexer)
			removeListGot, err := GetFlowSchemaRemoveCandidates(lister, test.bootstrap)
			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
			}

			if !cmp.Equal(test.expected, removeListGot, cmpopts.SortSlices(func(a string, b string) bool {
				return a < b
			})) {
				t.Errorf("Remove candidate list does not match - diff: %s", cmp.Diff(test.expected, removeListGot))
			}
		})
	}
}

type fsBuilder struct {
	object *flowcontrolv1beta2.FlowSchema
}

func newFlowSchema(name, plName string, matchingPrecedence int32) *fsBuilder {
	return &fsBuilder{
		object: &flowcontrolv1beta2.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: flowcontrolv1beta2.FlowSchemaSpec{
				PriorityLevelConfiguration: flowcontrolv1beta2.PriorityLevelConfigurationReference{
					Name: plName,
				},
				MatchingPrecedence: matchingPrecedence,
			},
		},
	}
}

func (b *fsBuilder) Object() *flowcontrolv1beta2.FlowSchema {
	return b.object
}

func (b *fsBuilder) WithGeneration(value int64) *fsBuilder {
	b.object.SetGeneration(value)
	return b
}

func (b *fsBuilder) WithAutoUpdateAnnotation(value string) *fsBuilder {
	setAnnotation(b.object, value)
	return b
}

func setAnnotation(accessor metav1.Object, value string) {
	if accessor.GetAnnotations() == nil {
		accessor.SetAnnotations(map[string]string{})
	}

	accessor.GetAnnotations()[flowcontrolv1beta2.AutoUpdateAnnotationKey] = value
}
