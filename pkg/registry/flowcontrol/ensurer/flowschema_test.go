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

	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/client-go/kubernetes/fake"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1"
	toolscache "k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	flowcontrolapisv1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

func init() {
	klog.InitFlags(nil)
}

func TestEnsureFlowSchema(t *testing.T) {
	tests := []struct {
		name      string
		strategy  func() EnsureStrategy[*flowcontrolv1.FlowSchema]
		current   *flowcontrolv1.FlowSchema
		bootstrap *flowcontrolv1.FlowSchema
		expected  *flowcontrolv1.FlowSchema
	}{
		// for suggested configurations
		{
			name:      "suggested flow schema does not exist - the object should always be re-created",
			strategy:  NewSuggestedEnsureStrategy[*flowcontrolv1.FlowSchema],
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   nil,
			expected:  newFlowSchema("fs1", "pl1", 100).Object(),
		},
		{
			name:      "suggested flow schema exists, auto update is enabled, spec does not match - current object should be updated",
			strategy:  NewSuggestedEnsureStrategy[*flowcontrolv1.FlowSchema],
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("true").Object(),
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "suggested flow schema exists, auto update is disabled, spec does not match - current object should not be updated",
			strategy:  NewSuggestedEnsureStrategy[*flowcontrolv1.FlowSchema],
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			expected:  newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
		},

		// for mandatory configurations
		{
			name:      "mandatory flow schema does not exist - new object should be created",
			strategy:  NewMandatoryEnsureStrategy[*flowcontrolv1.FlowSchema],
			bootstrap: newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
			current:   nil,
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "mandatory flow schema exists, annotation is missing - annotation should be added",
			strategy:  NewMandatoryEnsureStrategy[*flowcontrolv1.FlowSchema],
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 100).Object(),
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "mandatory flow schema exists, auto update is disabled, spec does not match - current object should be updated",
			strategy:  NewMandatoryEnsureStrategy[*flowcontrolv1.FlowSchema],
			bootstrap: newFlowSchema("fs1", "pl1", 100).Object(),
			current:   newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			expected:  newFlowSchema("fs1", "pl1", 100).WithAutoUpdateAnnotation("true").Object(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset().FlowcontrolV1().FlowSchemas()
			indexer := toolscache.NewIndexer(toolscache.MetaNamespaceKeyFunc, toolscache.Indexers{})
			if test.current != nil {
				client.Create(context.TODO(), test.current, metav1.CreateOptions{})
				indexer.Add(test.current)
			}

			ops := NewFlowSchemaOps(client, flowcontrollisters.NewFlowSchemaLister(indexer))
			boots := []*flowcontrolv1.FlowSchema{test.bootstrap}
			strategy := test.strategy()
			err := EnsureConfigurations(context.Background(), ops, boots, strategy)
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
		current           *flowcontrolv1.FlowSchema
		bootstrap         *flowcontrolv1.FlowSchema
		newObjectExpected *flowcontrolv1.FlowSchema
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

	ops := NewFlowSchemaOps(nil, nil)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			strategy := NewSuggestedEnsureStrategy[*flowcontrolv1.FlowSchema]()
			updatableGot, updateGot, err := strategy.ReviseIfNeeded(ops, test.current, test.bootstrap)
			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
			if test.newObjectExpected == nil {
				if updatableGot != nil {
					t.Errorf("Expected a nil object, but got: %#v", updatableGot)
				}
				if updateGot {
					t.Errorf("Expected update=%t but got: %t", false, updateGot)
				}
				return
			}

			if !updateGot {
				t.Errorf("Expected update=%t but got: %t", true, updateGot)
			}
			if !reflect.DeepEqual(test.newObjectExpected, updatableGot) {
				t.Errorf("Expected the object to be updated to match - diff: %s", cmp.Diff(test.newObjectExpected, updatableGot))
			}
		})
	}
}

func TestFlowSchemaSpecChanged(t *testing.T) {
	fs1 := &flowcontrolv1.FlowSchema{
		Spec: flowcontrolv1.FlowSchemaSpec{},
	}
	fs2 := &flowcontrolv1.FlowSchema{
		Spec: flowcontrolv1.FlowSchemaSpec{
			MatchingPrecedence: 1,
		},
	}
	fs1Defaulted := &flowcontrolv1.FlowSchema{
		Spec: flowcontrolv1.FlowSchemaSpec{
			MatchingPrecedence: flowcontrolapisv1.FlowSchemaDefaultMatchingPrecedence,
		},
	}
	testCases := []struct {
		name        string
		expected    *flowcontrolv1.FlowSchema
		actual      *flowcontrolv1.FlowSchema
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
			w := !flowSchemaSpecEqual(testCase.expected, testCase.actual)
			assert.Equal(t, testCase.specChanged, w)
		})
	}
}

func TestRemoveFlowSchema(t *testing.T) {
	tests := []struct {
		name           string
		current        *flowcontrolv1.FlowSchema
		bootstrapName  string
		removeExpected bool
	}{
		{
			name:          "no flow schema objects exist",
			bootstrapName: "fs1",
			current:       nil,
		},
		{
			name:           "flow schema unwanted, auto update is enabled",
			bootstrapName:  "fs0",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("true").Object(),
			removeExpected: true,
		},
		{
			name:           "flow schema unwanted, auto update is disabled",
			bootstrapName:  "fs0",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			removeExpected: false,
		},
		{
			name:           "flow schema unwanted, the auto-update annotation is malformed",
			bootstrapName:  "fs0",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("invalid").Object(),
			removeExpected: false,
		},
		{
			name:           "flow schema wanted, auto update is enabled",
			bootstrapName:  "fs1",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("true").Object(),
			removeExpected: false,
		},
		{
			name:           "flow schema wanted, auto update is disabled",
			bootstrapName:  "fs1",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("false").Object(),
			removeExpected: false,
		},
		{
			name:           "flow schema wanted, the auto-update annotation is malformed",
			bootstrapName:  "fs1",
			current:        newFlowSchema("fs1", "pl1", 200).WithAutoUpdateAnnotation("invalid").Object(),
			removeExpected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset().FlowcontrolV1().FlowSchemas()
			indexer := toolscache.NewIndexer(toolscache.MetaNamespaceKeyFunc, toolscache.Indexers{})
			if test.current != nil {
				client.Create(context.TODO(), test.current, metav1.CreateOptions{})
				indexer.Add(test.current)
			}
			bootFS := newFlowSchema(test.bootstrapName, "pl", 100).Object()
			ops := NewFlowSchemaOps(client, flowcontrollisters.NewFlowSchemaLister(indexer))
			boots := []*flowcontrolv1.FlowSchema{bootFS}
			err := RemoveUnwantedObjects(context.Background(), ops, boots)

			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
			}

			if test.current == nil {
				return
			}
			_, err = client.Get(context.TODO(), test.current.Name, metav1.GetOptions{})
			switch {
			case test.removeExpected:
				if !apierrors.IsNotFound(err) {
					t.Errorf("Expected error from Get after Delete: %q, but got: %v", metav1.StatusReasonNotFound, err)
				}
			default:
				if err != nil {
					t.Errorf("Expected no error from Get after Delete, but got: %v", err)
				}
			}
		})
	}
}

type fsBuilder struct {
	object *flowcontrolv1.FlowSchema
}

func newFlowSchema(name, plName string, matchingPrecedence int32) *fsBuilder {
	return &fsBuilder{
		object: &flowcontrolv1.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: flowcontrolv1.FlowSchemaSpec{
				PriorityLevelConfiguration: flowcontrolv1.PriorityLevelConfigurationReference{
					Name: plName,
				},
				MatchingPrecedence: matchingPrecedence,
			},
		},
	}
}

func (b *fsBuilder) Object() *flowcontrolv1.FlowSchema {
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

	accessor.GetAnnotations()[flowcontrolv1.AutoUpdateAnnotationKey] = value
}
