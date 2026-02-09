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

package compatversion

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

var (
	testedMetrics = []string{"version_info"}
)

func TestRecordCompatVersionInfo(t *testing.T) {
	RecordCompatVersionInfo(context.Background(), "name", "1.28", "1.27", "1.26")
	want := `# HELP version_info [ALPHA] Provides the compatibility version info of the component. The component label is the name of the component, usually kube, but is relevant for aggregated-apiservers.
    # TYPE version_info gauge
    version_info{binary="1.28",component="name",emulation="1.27",min_compat="1.26"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), testedMetrics...); err != nil {
		t.Fatal(err)
	}
}
