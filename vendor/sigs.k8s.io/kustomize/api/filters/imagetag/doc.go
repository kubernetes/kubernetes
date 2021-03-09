// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package imagetag contains two kio.Filter implementations to cover the
// functionality of the kustomize imagetag transformer.
//
// Filter updates fields based on a FieldSpec and an ImageTag.
//
// LegacyFilter doesn't use a FieldSpec, and instead only updates image
// references if the field is name image and it is underneath a field called
// either containers or initContainers.
package imagetag
