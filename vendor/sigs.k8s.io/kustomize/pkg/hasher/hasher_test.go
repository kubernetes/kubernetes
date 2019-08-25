// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package hasher_test

import (
	"testing"

	. "sigs.k8s.io/kustomize/pkg/hasher"
)

func TestSortArrayAndComputeHash(t *testing.T) {
	array1 := []string{"a", "b", "c", "d"}
	array2 := []string{"c", "b", "d", "a"}
	h1, err := SortArrayAndComputeHash(array1)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}
	if h1 == "" {
		t.Errorf("failed to hash %v", array1)
	}
	h2, err := SortArrayAndComputeHash(array2)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}
	if h2 == "" {
		t.Errorf("failed to hash %v", array2)
	}
	if h1 != h2 {
		t.Errorf("hash is not consistent with reordered list: %s %s", h1, h2)
	}
}

func TestHash(t *testing.T) {
	// hash the empty string to be sure that sha256 is being used
	expect := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	sum := Hash("")
	if expect != sum {
		t.Errorf("expected hash %q but got %q", expect, sum)
	}
}
