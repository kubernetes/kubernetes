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
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1beta2"
	"k8s.io/client-go/tools/cache"
	flowcontrolapisv1beta2 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta2"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

func TestEnsurePriorityLevel(t *testing.T) {
	tests := []struct {
		name      string
		strategy  func() EnsureStrategy
		current   *flowcontrolv1beta2.PriorityLevelConfiguration
		bootstrap *flowcontrolv1beta2.PriorityLevelConfiguration
		expected  *flowcontrolv1beta2.PriorityLevelConfiguration
	}{
		// for suggested configurations
		{
			name:      "suggested priority level configuration does not exist - the object should always be re-created",
			strategy:  NewSuggestedEnsureStrategy,
			bootstrap: newPLConfiguration("pl1").WithLimited(10).Object(),
			current:   nil,
			expected:  newPLConfiguration("pl1").WithLimited(10).Object(),
		},
		{
			name:      "suggested priority level configuration exists, auto update is enabled, spec does not match - current object should be updated",
			strategy:  NewSuggestedEnsureStrategy,
			bootstrap: newPLConfiguration("pl1").WithLimited(20).Object(),
			current:   newPLConfiguration("pl1").WithAutoUpdateAnnotation("true").WithLimited(10).Object(),
			expected:  newPLConfiguration("pl1").WithAutoUpdateAnnotation("true").WithLimited(20).Object(),
		},
		{
			name:      "suggested priority level configuration exists, auto update is disabled, spec does not match - current object should not be updated",
			strategy:  NewSuggestedEnsureStrategy,
			bootstrap: newPLConfiguration("pl1").WithLimited(20).Object(),
			current:   newPLConfiguration("pl1").WithAutoUpdateAnnotation("false").WithLimited(10).Object(),
			expected:  newPLConfiguration("pl1").WithAutoUpdateAnnotation("false").WithLimited(10).Object(),
		},

		// for mandatory configurations
		{
			name:      "mandatory priority level configuration does not exist - new object should be created",
			strategy:  NewMandatoryEnsureStrategy,
			bootstrap: newPLConfiguration("pl1").WithLimited(10).WithAutoUpdateAnnotation("true").Object(),
			current:   nil,
			expected:  newPLConfiguration("pl1").WithLimited(10).WithAutoUpdateAnnotation("true").Object(),
		},
		{
			name:      "mandatory priority level configuration exists, annotation is missing - annotation is added",
			strategy:  NewMandatoryEnsureStrategy,
			bootstrap: newPLConfiguration("pl1").WithLimited(20).Object(),
			current:   newPLConfiguration("pl1").WithLimited(20).Object(),
			expected:  newPLConfiguration("pl1").WithAutoUpdateAnnotation("true").WithLimited(20).Object(),
		},
		{
			name:      "mandatory priority level configuration exists, auto update is disabled, spec does not match - current object should be updated",
			strategy:  NewMandatoryEnsureStrategy,
			bootstrap: newPLConfiguration("pl1").WithLimited(20).Object(),
			current:   newPLConfiguration("pl1").WithAutoUpdateAnnotation("false").WithLimited(10).Object(),
			expected:  newPLConfiguration("pl1").WithAutoUpdateAnnotation("true").WithLimited(20).Object(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset().FlowcontrolV1beta2().PriorityLevelConfigurations()
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if test.current != nil {
				client.Create(context.TODO(), test.current, metav1.CreateOptions{})
				indexer.Add(test.current)
			}

			boots := WrapBootstrapPriorityLevelConfigurations(client, flowcontrollisters.NewPriorityLevelConfigurationLister(indexer), []*flowcontrolv1beta2.PriorityLevelConfiguration{test.bootstrap})
			strategy := test.strategy()

			err := EnsureConfigurations(boots, strategy)
			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
			}

			plGot, err := client.Get(context.TODO(), test.bootstrap.Name, metav1.GetOptions{})
			switch {
			case test.expected == nil:
				if !apierrors.IsNotFound(err) {
					t.Fatalf("Expected GET to return an %q error, but got: %v", metav1.StatusReasonNotFound, err)
				}
			case err != nil:
				t.Fatalf("Expected GET to return no error, but got: %v", err)
			}

			if !reflect.DeepEqual(test.expected, plGot) {
				t.Errorf("PriorityLevelConfiguration does not match - diff: %s", cmp.Diff(test.expected, plGot))
			}
		})
	}
}

func TestSuggestedPLEnsureStrategy_ShouldUpdate(t *testing.T) {
	tests := []struct {
		name              string
		current           *flowcontrolv1beta2.PriorityLevelConfiguration
		bootstrap         *flowcontrolv1beta2.PriorityLevelConfiguration
		newObjectExpected *flowcontrolv1beta2.PriorityLevelConfiguration
	}{
		{
			name:              "auto update is enabled, first generation, spec does not match - spec update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(1).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(10).Object(),
			newObjectExpected: newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(1).WithLimited(10).Object(),
		},
		{
			name:              "auto update is enabled, first generation, spec matches - no update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(1).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithGeneration(1).WithLimited(5).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is enabled, second generation, spec does not match - spec update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(2).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(10).Object(),
			newObjectExpected: newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(2).WithLimited(10).Object(),
		},
		{
			name:              "auto update is enabled, second generation, spec matches - no update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(2).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(5).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, first generation, spec does not match - no update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("false").WithGeneration(1).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(10).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, first generation, spec matches - no update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("false").WithGeneration(1).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(5).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, second generation, spec does not match - no update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("false").WithGeneration(2).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(10).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "auto update is disabled, second generation, spec matches - no update expected",
			current:           newPLConfiguration("foo").WithAutoUpdateAnnotation("false").WithGeneration(2).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(5).Object(),
			newObjectExpected: nil,
		},
		{
			name:              "annotation is missing, first generation, spec does not match - both annotation and spec update expected",
			current:           newPLConfiguration("foo").WithGeneration(1).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(10).Object(),
			newObjectExpected: newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(1).WithLimited(10).Object(),
		},
		{
			name:              "annotation is missing, first generation, spec matches - annotation update is expected",
			current:           newPLConfiguration("foo").WithGeneration(1).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(5).Object(),
			newObjectExpected: newPLConfiguration("foo").WithAutoUpdateAnnotation("true").WithGeneration(1).WithLimited(5).Object(),
		},
		{
			name:              "annotation is missing, second generation, spec does not match - annotation update is expected",
			current:           newPLConfiguration("foo").WithGeneration(2).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(10).Object(),
			newObjectExpected: newPLConfiguration("foo").WithAutoUpdateAnnotation("false").WithGeneration(2).WithLimited(5).Object(),
		},
		{
			name:              "annotation is missing, second generation, spec matches - annotation update is expected",
			current:           newPLConfiguration("foo").WithGeneration(2).WithLimited(5).Object(),
			bootstrap:         newPLConfiguration("foo").WithLimited(5).Object(),
			newObjectExpected: newPLConfiguration("foo").WithAutoUpdateAnnotation("false").WithGeneration(2).WithLimited(5).Object(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			strategy := NewSuggestedEnsureStrategy()
			wah := &wantAndHavePriorityLevelConfiguration{
				want: test.bootstrap,
				have: test.current,
			}
			updatableGot, updateGot, err := strategy.ShouldUpdate(wah)
			if err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
			var newObjectGot *flowcontrolv1beta2.PriorityLevelConfiguration
			if updatableGot != nil {
				newObjectGot = updatableGot.(*updatablePriorityLevelConfiguration).PriorityLevelConfiguration
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

func TestPriorityLevelSpecChanged(t *testing.T) {
	pl1 := &flowcontrolv1beta2.PriorityLevelConfiguration{
		Spec: flowcontrolv1beta2.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1beta2.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1beta2.LimitedPriorityLevelConfiguration{
				LimitResponse: flowcontrolv1beta2.LimitResponse{
					Type: flowcontrolv1beta2.LimitResponseTypeReject,
				},
			},
		},
	}
	pl2 := &flowcontrolv1beta2.PriorityLevelConfiguration{
		Spec: flowcontrolv1beta2.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1beta2.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1beta2.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: 1,
			},
		},
	}
	pl1Defaulted := &flowcontrolv1beta2.PriorityLevelConfiguration{
		Spec: flowcontrolv1beta2.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1beta2.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1beta2.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: flowcontrolapisv1beta2.PriorityLevelConfigurationDefaultAssuredConcurrencyShares,
				LimitResponse: flowcontrolv1beta2.LimitResponse{
					Type: flowcontrolv1beta2.LimitResponseTypeReject,
				},
			},
		},
	}
	testCases := []struct {
		name        string
		expected    *flowcontrolv1beta2.PriorityLevelConfiguration
		actual      *flowcontrolv1beta2.PriorityLevelConfiguration
		specChanged bool
	}{
		{
			name:        "identical priority-level should work",
			expected:    bootstrap.MandatoryPriorityLevelConfigurationCatchAll,
			actual:      bootstrap.MandatoryPriorityLevelConfigurationCatchAll,
			specChanged: false,
		},
		{
			name:        "defaulted priority-level should work",
			expected:    pl1,
			actual:      pl1Defaulted,
			specChanged: false,
		},
		{
			name:        "non-defaulted priority-level has wrong spec",
			expected:    pl1,
			actual:      pl2,
			specChanged: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			w := priorityLevelSpecChanged(testCase.expected, testCase.actual)
			assert.Equal(t, testCase.specChanged, w)
		})
	}
}

func TestRemovePriorityLevelConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		current        *flowcontrolv1beta2.PriorityLevelConfiguration
		bootstrapName  string
		removeExpected bool
	}{
		{
			name:          "no priority level configuration objects exist",
			bootstrapName: "pl1",
			current:       nil,
		},
		{
			name:           "priority level configuration not wanted, auto update is enabled",
			bootstrapName:  "pl0",
			current:        newPLConfiguration("pl1").WithAutoUpdateAnnotation("true").Object(),
			removeExpected: true,
		},
		{
			name:           "priority level configuration not wanted, auto update is disabled",
			bootstrapName:  "pl0",
			current:        newPLConfiguration("pl1").WithAutoUpdateAnnotation("false").Object(),
			removeExpected: false,
		},
		{
			name:           "priority level configuration not wanted, the auto-update annotation is malformed",
			bootstrapName:  "pl0",
			current:        newPLConfiguration("pl1").WithAutoUpdateAnnotation("invalid").Object(),
			removeExpected: false,
		},
		{
			name:           "priority level configuration wanted, auto update is enabled",
			bootstrapName:  "pl1",
			current:        newPLConfiguration("pl1").WithAutoUpdateAnnotation("true").Object(),
			removeExpected: false,
		},
		{
			name:           "priority level configuration wanted, auto update is disabled",
			bootstrapName:  "pl1",
			current:        newPLConfiguration("pl1").WithAutoUpdateAnnotation("false").Object(),
			removeExpected: false,
		},
		{
			name:           "priority level configuration wanted, the auto-update annotation is malformed",
			bootstrapName:  "pl1",
			current:        newPLConfiguration("pl1").WithAutoUpdateAnnotation("invalid").Object(),
			removeExpected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := fake.NewSimpleClientset().FlowcontrolV1beta2().PriorityLevelConfigurations()
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if test.current != nil {
				client.Create(context.TODO(), test.current, metav1.CreateOptions{})
				indexer.Add(test.current)
			}

			boot := newPLConfiguration(test.bootstrapName).Object()
			boots := WrapBootstrapPriorityLevelConfigurations(client, flowcontrollisters.NewPriorityLevelConfigurationLister(indexer), []*flowcontrolv1beta2.PriorityLevelConfiguration{boot})
			err := RemoveUnwantedObjects(boots)
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

type plBuilder struct {
	object *flowcontrolv1beta2.PriorityLevelConfiguration
}

func newPLConfiguration(name string) *plBuilder {
	return &plBuilder{
		object: &flowcontrolv1beta2.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		},
	}
}

func (b *plBuilder) Object() *flowcontrolv1beta2.PriorityLevelConfiguration {
	return b.object
}

func (b *plBuilder) WithGeneration(value int64) *plBuilder {
	b.object.SetGeneration(value)
	return b
}

func (b *plBuilder) WithAutoUpdateAnnotation(value string) *plBuilder {
	setAnnotation(b.object, value)
	return b
}

func (b *plBuilder) WithLimited(assuredConcurrencyShares int32) *plBuilder {
	b.object.Spec.Type = flowcontrolv1beta2.PriorityLevelEnablementLimited
	b.object.Spec.Limited = &flowcontrolv1beta2.LimitedPriorityLevelConfiguration{
		AssuredConcurrencyShares: assuredConcurrencyShares,
		LimitResponse: flowcontrolv1beta2.LimitResponse{
			Type: flowcontrolv1beta2.LimitResponseTypeReject,
		},
	}
	return b
}

// must be called after WithLimited
func (b *plBuilder) WithQueuing(queues, handSize, queueLengthLimit int32) *plBuilder {
	limited := b.object.Spec.Limited
	if limited == nil {
		return b
	}

	limited.LimitResponse.Type = flowcontrolv1beta2.LimitResponseTypeQueue
	limited.LimitResponse.Queuing = &flowcontrolv1beta2.QueuingConfiguration{
		Queues:           queues,
		HandSize:         handSize,
		QueueLengthLimit: queueLengthLimit,
	}

	return b
}
