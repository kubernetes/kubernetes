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

package example

import _ "embed"

//go:embed builtin/pod.yaml
var podYAML string

//go:embed builtin/deployment.yaml
var deploymentYAML string

//go:embed builtin/service.yaml
var serviceYAML string

//go:embed builtin/persistentvolumeclaim.yaml
var pvcYAML string

//go:embed builtin/secret.yaml
var secretYAML string

//go:embed builtin/crd.yaml
var crdYAML string

// examplesByKind maps lower-cased Kind or canonical resource to embedded YAML templates.
var examplesByKind = map[string]string{
	"pod":                       podYAML,
	"pods":                      podYAML,
	"po":                        podYAML,
	"deployment":                deploymentYAML,
	"deployments":               deploymentYAML,
	"deploy":                    deploymentYAML,
	"service":                   serviceYAML,
	"services":                  serviceYAML,
	"svc":                       serviceYAML,
	"persistentvolumeclaim":     pvcYAML,
	"persistentvolumeclaims":    pvcYAML,
	"pvc":                       pvcYAML,
	"secret":                    secretYAML,
	"secrets":                   secretYAML,
	"customresourcedefinition":  crdYAML,
	"customresourcedefinitions": crdYAML,
	"crd":                       crdYAML,
}
