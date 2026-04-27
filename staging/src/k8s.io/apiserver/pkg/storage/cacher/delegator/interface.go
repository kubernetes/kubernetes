/*
Copyright 2025 The Kubernetes Authors.

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

package delegator

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	etcdfeature "k8s.io/apiserver/pkg/storage/feature"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

func ShouldDelegateListMeta(opts *metav1.ListOptions, cache Helper) (Result, error) {
	return ShouldDelegateList(
		storage.ListOptions{
			ResourceVersionMatch: opts.ResourceVersionMatch,
			ResourceVersion:      opts.ResourceVersion,
			Predicate: storage.SelectionPredicate{
				Continue: opts.Continue,
				Limit:    opts.Limit,
			},
			Recursive: true,
		}, cache)
}

func ShouldDelegateList(opts storage.ListOptions, cache Helper) (Result, error) {
	// see https://kubernetes.io/docs/reference/using-api/api-concepts/#semantics-for-get-and-list
	switch opts.ResourceVersionMatch {
	case metav1.ResourceVersionMatchExact:
		return cache.ShouldDelegateExactRV(opts.ResourceVersion, opts.Recursive)
	case metav1.ResourceVersionMatchNotOlderThan:
		return Result{ShouldDelegate: false}, nil
	case "":
		// Continue
		if len(opts.Predicate.Continue) > 0 {
			return cache.ShouldDelegateContinue(opts.Predicate.Continue, opts.Recursive)
		}
		// Legacy exact match
		if opts.Predicate.Limit > 0 && len(opts.ResourceVersion) > 0 && opts.ResourceVersion != "0" {
			return cache.ShouldDelegateExactRV(opts.ResourceVersion, opts.Recursive)
		}
		// Consistent Read
		if opts.ResourceVersion == "" {
			return cache.ShouldDelegateConsistentRead()
		}
		return Result{ShouldDelegate: false}, nil
	default:
		return Result{ShouldDelegate: true}, nil
	}
}

type Helper interface {
	ShouldDelegateExactRV(rv string, recursive bool) (Result, error)
	ShouldDelegateContinue(continueToken string, recursive bool) (Result, error)
	ShouldDelegateConsistentRead() (Result, error)
}

// Result of delegator decision.
type Result struct {
	// Whether a request cannot be served by cache and should be delegated to etcd.
	ShouldDelegate bool
	// Whether a request is a consistent read, used by delegator to decide if it should call GetCurrentResourceVersion to get RV.
	// Included in interface as only cacher has keyPrefix needed to parse continue token.
	ConsistentRead bool
}

type CacheWithoutSnapshots struct{}

var _ Helper = CacheWithoutSnapshots{}

func (c CacheWithoutSnapshots) ShouldDelegateContinue(continueToken string, recursive bool) (Result, error) {
	return Result{
		ShouldDelegate: true,
		// Continue with negative RV is considered a consistent read, however token cannot be parsed without keyPrefix unavailable in staging/src/k8s.io/apiserver/pkg/util/flow_control/request/list_work_estimator.go.
		ConsistentRead: false,
	}, nil
}

func (c CacheWithoutSnapshots) ShouldDelegateExactRV(rv string, recursive bool) (Result, error) {
	return Result{
		ShouldDelegate: true,
		ConsistentRead: false,
	}, nil
}

func (c CacheWithoutSnapshots) ShouldDelegateConsistentRead() (Result, error) {
	return Result{
		ShouldDelegate: !ConsistentReadSupported(),
		ConsistentRead: true,
	}, nil
}

// ConsistentReadSupported returns whether cache can be used to serve reads with RV not yet observed by cache, including both consistent reads.
// Function is located here to avoid import cycles between staging/src/k8s.io/apiserver/pkg/storage/cacher/delegator.go and staging/src/k8s.io/apiserver/pkg/util/flow_control/request/list_work_estimator.go.
func ConsistentReadSupported() bool {
	consistentListFromCacheEnabled := utilfeature.DefaultFeatureGate.Enabled(features.ConsistentListFromCache)
	requestWatchProgressSupported := etcdfeature.DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress)
	return consistentListFromCacheEnabled && requestWatchProgressSupported
}
