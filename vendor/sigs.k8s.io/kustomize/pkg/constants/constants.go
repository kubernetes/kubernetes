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
package constants

// KustomizationFileSuffix is expected suffix for KustomizationFileName.
const KustomizationFileSuffix = ".yaml"

// SecondaryKustomizationFileSuffix is the second expected suffix when KustomizationFileSuffix is not found
const SecondaryKustomizationFileSuffix = ".yml"

// KustomizationFileName is the Well-Known File Name for a kustomize configuration file.
const KustomizationFileName = "kustomization" + KustomizationFileSuffix

// SecondaryKustomizationFileName is the secondary File Name for a kustomize configuration file when
// KustomizationFileName is not found
const SecondaryKustomizationFileName = "kustomization" + SecondaryKustomizationFileSuffix
