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

package fixtures

import (
	"embed"

	"k8s.io/kubernetes/test/e2e/framework/testfiles"
)

//go:embed doc-yaml/user-guide/liveness doc-yaml/user-guide/secrets doc-yaml/user-guide/downward-api doc-yaml/user-guide/update-demo
var testFixturesFS embed.FS

func GetTestFixturesFS() testfiles.EmbeddedFileSource {
	return testfiles.EmbeddedFileSource{
		EmbeddedFS: testFixturesFS,
		Root:       "test/fixtures",
	}
}
