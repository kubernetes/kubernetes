// Copyright 2026 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package openapi

import _ "embed"

//go:generate go run ./cmd/openapi-bundle -input kubernetesapi/v1_21_2/swagger.pb.gz -output kubernetesapi/data/kubernetes-openapi-union-v1.21.2.bundle-v1.json.gz -kubernetes-version v1.21.2

const (
	// DefaultOpenAPI is the Kubernetes version represented by the built-in
	// schema. It remains v1.21.2 during the initial artifact-format migration.
	DefaultOpenAPI = "v1.21.2"

	// BuiltinSchemaInfo is the value printed by `kustomize openapi info`.
	BuiltinSchemaInfo = "{title:Kubernetes,version:" + DefaultOpenAPI + "}"
)

//go:embed kubernetesapi/data/kubernetes-openapi-union-v1.21.2.bundle-v1.json.gz
var builtinKubernetesOpenAPIBundle []byte

//go:embed kustomizationapi/swagger.json
var builtinKustomizationOpenAPI []byte

func hasBuiltinOpenAPIVersion(version string) bool {
	return version == DefaultOpenAPI
}
