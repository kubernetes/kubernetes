// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package inventory

const (
	// Annotation that contains the inventory content.
	ContentAnnotation = "kustomize.config.k8s.io/Inventory"

	// Annotation for inventory content hash.
	HashAnnotation = "kustomize.config.k8s.io/InventoryHash"
)
