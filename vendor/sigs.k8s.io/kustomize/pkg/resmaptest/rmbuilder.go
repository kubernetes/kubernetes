// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resmaptest_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
)

// Builds ResMaps for tests, with test-aware error handling.
type rmBuilder struct {
	t  *testing.T
	m  resmap.ResMap
	rf *resource.Factory
}

func NewSeededRmBuilder(t *testing.T, rf *resource.Factory, m resmap.ResMap) *rmBuilder {
	return &rmBuilder{t: t, rf: rf, m: m}
}

func NewRmBuilder(t *testing.T, rf *resource.Factory) *rmBuilder {
	return NewSeededRmBuilder(t, rf, resmap.New())
}

func (rm *rmBuilder) Add(m map[string]interface{}) *rmBuilder {
	return rm.AddR(rm.rf.FromMap(m))
}

func (rm *rmBuilder) AddR(r *resource.Resource) *rmBuilder {
	err := rm.m.Append(r)
	if err != nil {
		rm.t.Fatalf("test setup failure: %v", err)
	}
	return rm
}

func (rm *rmBuilder) AddWithId(id resid.ResId, m map[string]interface{}) *rmBuilder {
	err := rm.m.Append(rm.rf.FromMap(m))
	if err != nil {
		rm.t.Fatalf("test setup failure: %v", err)
	}
	return rm
}

func (rm *rmBuilder) AddWithName(n string, m map[string]interface{}) *rmBuilder {
	err := rm.m.Append(rm.rf.FromMapWithName(n, m))
	if err != nil {
		rm.t.Fatalf("test setup failure: %v", err)
	}
	return rm
}

func (rm *rmBuilder) AddWithNs(ns string, m map[string]interface{}) *rmBuilder {
	err := rm.m.Append(rm.rf.FromMapWithNamespace(ns, m))
	if err != nil {
		rm.t.Fatalf("test setup failure: %v", err)
	}
	return rm
}

func (rm *rmBuilder) ReplaceResource(m map[string]interface{}) *rmBuilder {
	r := rm.rf.FromMap(m)
	_, err := rm.m.Replace(r)
	if err != nil {
		rm.t.Fatalf("test setup failure: %v", err)
	}
	return rm
}

func (rm *rmBuilder) ResMap() resmap.ResMap {
	return rm.m
}
