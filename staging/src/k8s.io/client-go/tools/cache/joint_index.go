/*
Copyright 2022 The Kubernetes Authors.

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

package cache

import (
	"fmt"

	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	// UnionIndexType performs a union operation between the results.
	UnionIndexType JointIndexType = "Union"
	// IntersectionIndexType performs an intersection operation between the results.
	IntersectionIndexType JointIndexType = "Intersection"
	// SymmetricDifferenceIndexType performs a symmetric difference operation between the results.
	SymmetricDifferenceIndexType JointIndexType = "SymmetricDifference"
)

// JointIndexType the build-in joint indexing type.
type JointIndexType string

// IndexConditions contains each IndexCondition.
type IndexConditions []IndexCondition

// JointIndexFunc makes a custom joint index by given index func.
type JointIndexFunc func(conds IndexConditions) (sets.String, error)

// JointIndexOptions applies to indexed result.
type JointIndexOptions struct {
	// IndexType is ignored if IndexFunc exists.
	IndexType JointIndexType
	// IndexFunc makes a custom joint index.
	IndexFunc JointIndexFunc
}

// IndexCondition contains a IndexName, a IndexedValue, and an operator.
type IndexCondition struct {
	IndexName    string
	IndexedValue string

	// Operator initial index action that support Equals, DoubleEquals and NotEquals.
	Operator selection.Operator

	// indexedResult holds the result of the initial index.
	indexedResult sets.String
}

// JointIndexer for doing convenient joint index against indexer by
// given IndexConditions and JointIndexOptions.
type JointIndexer interface {
	// ByIndexes returns the stored objects by given IndexConditions and JointIndexOptions
	ByIndexes(conds IndexConditions, opts *JointIndexOptions) ([]interface{}, error)
}

func (cond1 IndexCondition) Union(cond2 IndexCondition) IndexCondition {
	return IndexCondition{indexedResult: cond1.indexedResult.Union(cond2.indexedResult)}
}

func (cond1 IndexCondition) Intersection(cond2 IndexCondition) IndexCondition {
	return IndexCondition{indexedResult: cond1.indexedResult.Intersection(cond2.indexedResult)}
}

func (cond1 IndexCondition) Difference(cond2 IndexCondition) IndexCondition {
	return IndexCondition{indexedResult: cond1.indexedResult.Difference(cond2.indexedResult)}
}

func (cond1 IndexCondition) SymmetricDifference(cond2 IndexCondition) IndexCondition {
	// todo:(weilaaa) remove it as soon as SymmetricDifference method be added into sets.String.
	return IndexCondition{indexedResult: cond1.indexedResult.Difference(cond2.indexedResult).Union(cond2.indexedResult.Difference(cond1.indexedResult))}
}

func (cond1 IndexCondition) Complete() sets.String {
	return cond1.indexedResult
}

// apply JointIndexOptions to preliminary indexing results. If IndexType
// is empty, it will be set to UnionIndexType by default.
func (conds IndexConditions) apply(opts *JointIndexOptions) (sets.String, error) {
	if opts == nil {
		return sets.String{}, fmt.Errorf("joint index opts must be set")
	}

	if opts.IndexFunc != nil {
		return opts.IndexFunc(conds)
	}

	if len(opts.IndexType) == 0 {
		opts.IndexType = UnionIndexType
	}

	return conds.genericJointIndex(opts.IndexType)
}

// genericJointIndex provides built-in joint indexing methods, including pairwise
// union, pairwise intersection and pairwise symmetric difference.
func (conds IndexConditions) genericJointIndex(indexType JointIndexType) (sets.String, error) {
	if len(conds) < 1 {
		return sets.String{}, fmt.Errorf("index conditions must not be empty")
	}

	var indexFn func(set1, set2 sets.String) sets.String

	switch indexType {
	case UnionIndexType:
		indexFn = func(set1, set2 sets.String) sets.String { return set1.Union(set2) }
	case IntersectionIndexType:
		indexFn = func(set1, set2 sets.String) sets.String { return set1.Intersection(set2) }
	case SymmetricDifferenceIndexType:
		// todo:(weilaaa) remove it as soon as SymmetricDifference method be added into sets.String.
		indexFn = func(set1, set2 sets.String) sets.String { return set1.Difference(set2).Union(set2.Difference(set1)) }
	default:
		return sets.String{}, fmt.Errorf("joint index type \"%v\" is not supported", indexType)
	}

	for i := 0; i < len(conds)-1; i++ {
		conds[i+1].indexedResult = indexFn(conds[i].indexedResult, conds[i+1].indexedResult)
	}

	return conds[len(conds)-1].indexedResult, nil
}
