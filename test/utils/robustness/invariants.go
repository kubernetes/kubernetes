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

package robustness

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	clientset "k8s.io/client-go/kubernetes"
)

// Invariant defines a predicate evaluated against the un-wrapped admin client.
type Invariant func(ctx context.Context, c clientset.Interface) error

// NamedInvariant pairs an invariant with a human-readable name for reporting.
type NamedInvariant struct {
	Name string
	Fn   Invariant
}

// ObjectSatisfies builds an Invariant from a typed getter and a check on the fetched object.
func ObjectSatisfies[T any](get func(ctx context.Context, c clientset.Interface) (T, error), check func(obj T) error) Invariant {
	return func(ctx context.Context, c clientset.Interface) error {
		obj, err := get(ctx, c)
		if err != nil {
			return err
		}
		return check(obj)
	}
}

// CountAtMost enforces an upper bound on an object count. NotFound is treated as zero.
func CountAtMost(limit int, what string, count func(ctx context.Context, c clientset.Interface) (int, error)) Invariant {
	return func(ctx context.Context, c clientset.Interface) error {
		n, err := count(ctx, c)
		if err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		}
		if n > limit {
			return fmt.Errorf("safety violation: detected %d %s objects, maximum allowed is %d", n, what, limit)
		}
		return nil
	}
}

// AllOf combines multiple invariants into a single Invariant.
func AllOf(invariants ...Invariant) Invariant {
	return func(ctx context.Context, c clientset.Interface) error {
		for _, inv := range invariants {
			if err := inv(ctx, c); err != nil {
				return err
			}
		}
		return nil
	}
}
