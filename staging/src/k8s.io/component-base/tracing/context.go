/*
Copyright 2024 The Kubernetes Authors.

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

package tracing

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// assert that objectCarrier implements the TextMapCarrier interface.
var _ propagation.TextMapCarrier = &objectCarrier{}

// objectCarrier is a TextMapCarrier that stores and retrieves keys to/from
// Object annotations.
// TODO: Should this use a prefix?  e.g. tracing.kubernetes.io/traceparent vs traceparent
type objectCarrier struct {
	object metav1.Object
}

func (s *objectCarrier) Get(key string) string {
	return s.object.GetAnnotations()[key]
}

func (s *objectCarrier) Set(key string, value string) {
	annotations := s.object.GetAnnotations()
	annotations[key] = value
	s.object.SetAnnotations(annotations)
}

func (s *objectCarrier) Keys() []string {
	annotations := s.object.GetAnnotations()
	keys := make([]string, len(annotations))
	i := 0
	for k := range annotations {
		keys[i] = k
	}
	return keys
}

// Extract reads context from the object into the returned Context using the
// global propagators.
// This should be called when reconciling an object before starting any spans.
func ExtractContext(ctx context.Context, object metav1.Object) context.Context {
	return otel.GetTextMapPropagator().Extract(ctx, &objectCarrier{object: object})
}

// Extract injects the Context into the object using the global propagators.
// This should be called before updating the desired state of an object, including:
// * Creating the object
// * Setting the deletion timestamp of an object
// * Updating the object's Spec
// The caller needs to ensure the modified object is written after calling this
// function.
func InjectContext(ctx context.Context, object metav1.Object) {
	otel.GetTextMapPropagator().Inject(ctx, &objectCarrier{object: object})
}
