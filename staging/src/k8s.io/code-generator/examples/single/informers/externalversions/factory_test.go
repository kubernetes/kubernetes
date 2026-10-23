/*
Copyright The Kubernetes Authors.

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

package externalversions

import (
	"slices"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/client-go/tools/cache"
	singleapiv1 "k8s.io/code-generator/examples/single/api/v1"
	"k8s.io/code-generator/examples/single/clientset/versioned"
)

// TestTransforms verified that transform calls are applied as expected.
func TestTransforms(t *testing.T) {
	tests := []struct {
		name                    string
		useFactoryTransform     bool
		usePerInformerTransform bool
		wantTransformCalls      []string
	}{
		{
			name:               "no transforms",
			wantTransformCalls: nil,
		},
		{
			name:                    "no factory transform preserves per-informer transform",
			usePerInformerTransform: true,
			wantTransformCalls:      []string{"per-informer"},
		},
		{
			name:                "factory transform applied when no per-informer transform",
			useFactoryTransform: true,
			wantTransformCalls:  []string{"factory"},
		},
		{
			name:                    "factory transform overrides per-informer transform",
			useFactoryTransform:     true,
			usePerInformerTransform: true,
			wantTransformCalls:      []string{"factory"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var transformCalls []string
			makeTransform := func(name string) cache.TransformFunc {
				return func(obj interface{}) (interface{}, error) {
					transformCalls = append(transformCalls, name)
					return obj, nil
				}
			}

			var opts []SharedInformerOption
			if tt.useFactoryTransform {
				opts = append(opts, WithTransform(makeTransform("factory")))
			}
			factory := NewSharedInformerFactoryWithOptions(nil, 0, opts...)

			var wrapper *transformTrackingInformer
			newFunc := func(_ versioned.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
				inner := cache.NewSharedIndexInformer(nil, &singleapiv1.TestType{}, resyncPeriod, cache.Indexers{})
				wrapper = &transformTrackingInformer{SharedIndexInformer: inner}
				if tt.usePerInformerTransform {
					err := wrapper.SetTransform(makeTransform("per-informer"))
					if err != nil {
						t.Fatalf("failed to set transform: %v", err)
					}
				}
				return wrapper
			}
			factory.InformerFor(&singleapiv1.TestType{}, newFunc)

			if wrapper.lastTransform != nil {
				_, err := wrapper.lastTransform(nil)
				if err != nil {
					t.Fatalf("failed to invoke transform: %v", err)
				}
			}

			if !slices.Equal(transformCalls, tt.wantTransformCalls) {
				t.Errorf("transform calls: got %v, want %v", transformCalls, tt.wantTransformCalls)
			}
		})
	}
}

type transformTrackingInformer struct {
	cache.SharedIndexInformer
	lastTransform     cache.TransformFunc
	lastShardSelector sharding.Selector
}

func (s *transformTrackingInformer) SetTransform(handler cache.TransformFunc) error {
	s.lastTransform = handler
	return s.SharedIndexInformer.SetTransform(handler)
}

func (s *transformTrackingInformer) SetShardSelector(selector sharding.Selector) error {
	s.lastShardSelector = selector
	return s.SharedIndexInformer.SetShardSelector(selector)
}

func TestShardSelector(t *testing.T) {
	factorySel, err := sharding.NewShardRangeSelector("object.metadata.uid", 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	informerSel, err := sharding.NewShardRangeSelector("object.metadata.uid", 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                        string
		factorySelector             sharding.Selector
		perInformerSelector         sharding.Selector
		wantEffectiveSelectorString string
	}{
		{
			name:                        "no shard selector",
			wantEffectiveSelectorString: "",
		},
		{
			name:                        "no factory shard selector preserves per-informer shard selector",
			perInformerSelector:         informerSel,
			wantEffectiveSelectorString: informerSel.String(),
		},
		{
			name:                        "factory shard selector applied when no per-informer shard selector",
			factorySelector:             factorySel,
			wantEffectiveSelectorString: factorySel.String(),
		},
		{
			name:                        "factory shard selector overrides per-informer shard selector",
			factorySelector:             factorySel,
			perInformerSelector:         informerSel,
			wantEffectiveSelectorString: factorySel.String(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var opts []SharedInformerOption
			if tt.factorySelector != nil {
				opts = append(opts, WithShardSelector(tt.factorySelector))
			}
			factory := NewSharedInformerFactoryWithOptions(nil, 0, opts...)

			var wrapper *transformTrackingInformer
			newFunc := func(_ versioned.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
				inner := cache.NewSharedIndexInformer(nil, &singleapiv1.TestType{}, resyncPeriod, cache.Indexers{})
				wrapper = &transformTrackingInformer{SharedIndexInformer: inner}
				if tt.perInformerSelector != nil {
					if err := wrapper.SetShardSelector(tt.perInformerSelector); err != nil {
						t.Fatalf("failed to set shard selector: %v", err)
					}
				}
				return wrapper
			}
			factory.InformerFor(&singleapiv1.TestType{}, newFunc)

			got := ""
			if wrapper.lastShardSelector != nil {
				got = wrapper.lastShardSelector.String()
			}
			if got != tt.wantEffectiveSelectorString {
				t.Errorf("shard selector: got %q, want %q", got, tt.wantEffectiveSelectorString)
			}
		})
	}
}
