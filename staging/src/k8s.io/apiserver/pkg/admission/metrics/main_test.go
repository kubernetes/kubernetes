/*
Copyright The Kubernetes Authors.

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

package metrics

import (
	"os"
	"testing"

	compbasemetrics "k8s.io/component-base/metrics"
)

func TestMain(m *testing.M) {
	// The tests check deprecated metrics, which get hidden depending on the source version.
	// This is done once for the whole process because the global registry is populated
	// at init time and SetShowHidden cannot be undone or applied per test.
	compbasemetrics.SetShowHidden()
	os.Exit(m.Run())
}
