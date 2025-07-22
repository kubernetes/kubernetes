/*
Copyright 2021 The Kubernetes Authors.

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

package testing_manifests

import (
	"embed"

	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
)

//go:embed *.yaml
var e2eNodeTestingManifestsFS embed.FS

func GetE2ENodeTestingManifestsFS() e2etestfiles.EmbeddedFileSource {
	return e2etestfiles.EmbeddedFileSource{
		EmbeddedFS: e2eNodeTestingManifestsFS,
		Root:       "test/e2e_node/testing-manifests",
	}
}
