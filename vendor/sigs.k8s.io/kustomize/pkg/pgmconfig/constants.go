/*
Copyright 2017 The Kubernetes Authors.

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

// Package constants holds global constants for the kustomize tool.
package pgmconfig

// KustomizationFileNames is a list of filenames
// that kustomize recognizes.
// To avoid ambiguity, a directory cannot contain
// more than one match to this list.
var KustomizationFileNames = []string{
	"kustomization.yaml",
	"kustomization.yml",
	"Kustomization",
}

const (
	// Program name, for help, finding the XDG_CONFIG_DIR, etc.
	ProgramName = "kustomize"
	// Domain from which kustomize code is imported, for locating
	// plugin source code under $GOPATH.
	DomainName = "sigs.k8s.io"
)
