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

// Invariant defines a predicate that must hold true under evaluation. It is
// always evaluated against the un-wrapped admin client, so fault injection
// never distorts what the invariant observes.
type Invariant func(ctx context.Context, c clientset.Interface) error

// NamedInvariant pairs an invariant with a human-readable name for reporting.
type NamedInvariant struct {
	Name string
	Fn   Invariant
}

// ObjectSatisfies builds an Invariant from a typed getter and a check on the
// fetched object, separating "how to fetch" from "what must hold":
//
//	robustness.ObjectSatisfies(
//	    func(ctx context.Context, c clientset.Interface) (*appsv1.DaemonSet, error) {
//	        return c.AppsV1().DaemonSets("default").Get(ctx, "ds-1", metav1.GetOptions{})
//	    },
//	    func(ds *appsv1.DaemonSet) error {
//	        if ds.Status.CurrentNumberScheduled != 1 {
//	            return fmt.Errorf("want 1 scheduled, got %d", ds.Status.CurrentNumberScheduled)
//	        }
//	        return nil
//	    })
func ObjectSatisfies[T any](get func(ctx context.Context, c clientset.Interface) (T, error), check func(obj T) error) Invariant {
	return func(ctx context.Context, c clientset.Interface) error {
		obj, err := get(ctx, c)
		if err != nil {
			return err
		}
		return check(obj)
	}
}

// CountAtMost builds a safety Invariant enforcing an upper bound on an object
// count (e.g. "never more than one Pod per node"). NotFound from the counter is
// treated as zero, since an absent collection cannot violate an upper bound.
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
