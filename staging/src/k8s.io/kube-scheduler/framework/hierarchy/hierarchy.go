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

package hierarchy

import (
	"errors"
	"fmt"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/kube-scheduler/framework"
)

// ErrMaxTreeDepthExceeded indicates that a hierarchy traversal reached or exceeded WorkloadMaxTreeDepth.
var ErrMaxTreeDepthExceeded = errors.New("hierarchy exceeded maximum tree depth, possibly caused by cycle or deep hierarchy")

// WalkUp traverses upward from startKey by repeatedly resolving parent keys until no parent is found.
// At each node (including startKey), visitFn is called if provided.
// If visitFn returns (stop = true, nil), traversal terminates immediately and returns the current key.
// If visitFn returns an error, traversal terminates immediately and returns that error.
// If getParent returns (parent == nil, nil), traversal terminates and returns the current key as the root.
// If getParent returns an error (e.g., apierrors.IsNotFound when an entity or parent is missing from storage),
// traversal terminates immediately and returns that error.
// If the traversal does not terminate within WorkloadMaxTreeDepth steps, ErrMaxTreeDepthExceeded is returned.
func WalkUp(
	startKey framework.EntityKey,
	getParent func(key framework.EntityKey) (parent *framework.EntityKey, err error),
	visitFn func(key framework.EntityKey, depth int) (stop bool, err error),
) (framework.EntityKey, error) {
	currentKey := startKey
	for depth := range schedulingv1alpha3.WorkloadMaxTreeDepth {
		if visitFn != nil {
			stop, err := visitFn(currentKey, depth)
			if err != nil {
				return framework.EntityKey{}, err
			}
			if stop {
				return currentKey, nil
			}
		}
		parentKey, err := getParent(currentKey)
		if err != nil {
			return framework.EntityKey{}, err
		}
		if parentKey == nil {
			return currentKey, nil
		}
		currentKey = *parentKey
	}
	return framework.EntityKey{}, fmt.Errorf("hierarchy exceeded maximum tree depth at %s, possibly caused by cycle or deep hierarchy: %w", currentKey.String(), ErrMaxTreeDepthExceeded)
}

// WalkDown traverses downward from root by recursively visiting children up to WorkloadMaxTreeDepth levels.
// At each node, visitFn is called if provided.
// If visitFn returns stop = true, traversal terminates immediately across all branches.
// If visitFn returns skipChildren = true, children of the current node are not visited.
// If visitFn returns an error, traversal terminates immediately and returns that error.
// If depth reaches or exceeds WorkloadMaxTreeDepth and is not skipped by visitFn, ErrMaxTreeDepthExceeded is returned.
func WalkDown[T any](
	root T,
	getChildren func(node T) ([]T, error),
	visitFn func(node T, depth int) (stop bool, skipChildren bool, err error),
) error {
	_, err := dfs(root, 0, getChildren, visitFn)
	return err
}

// dfs recursively visits nodes depth-first up to WorkloadMaxTreeDepth levels.
// It returns (stopped = true, nil) to signal callers that traversal stopped early via visitFn,
// avoids expanding children if skipChildren is requested, and returns ErrMaxTreeDepthExceeded
// if the hierarchy exceeds the max depth limit.
func dfs[T any](
	curr T,
	depth int,
	getChildren func(node T) ([]T, error),
	visitFn func(node T, depth int) (stop bool, skipChildren bool, err error),
) (bool, error) {
	if depth >= schedulingv1alpha3.WorkloadMaxTreeDepth {
		return false, ErrMaxTreeDepthExceeded
	}
	if visitFn != nil {
		stop, skipChildren, err := visitFn(curr, depth)
		if err != nil {
			return false, err
		}
		if stop {
			return true, nil
		}
		if skipChildren {
			return false, nil
		}
	}
	children, err := getChildren(curr)
	if err != nil {
		return false, err
	}
	for _, child := range children {
		stopped, err := dfs(child, depth+1, getChildren, visitFn)
		if err != nil {
			return false, err
		}
		if stopped {
			return true, nil
		}
	}
	return false, nil
}
