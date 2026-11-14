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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestLastConfigInfo(t *testing.T) {
	RegisterMetrics()
	configHashProvider.SetHashes()
	defer configHashProvider.SetHashes()

	const name = "apiserver_storage_compression_last_config_info"
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(""), name); err != nil {
		t.Fatalf("no series may exist before a configuration is recorded: %v", err)
	}

	RecordLastConfigInfo("apiserver-1", "sha256:abc")
	RecordLastConfigInfo("apiserver-1", "sha256:def")
	want := `
# HELP apiserver_storage_compression_last_config_info [ALPHA] Information about the last loaded storage compression configuration, with its hash as a label, split by apiserver identity.
# TYPE apiserver_storage_compression_last_config_info gauge
apiserver_storage_compression_last_config_info{apiserver_id_hash="` + getHash("apiserver-1") + `",hash="sha256:def"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), name); err != nil {
		t.Fatalf("only the last recorded configuration may be exported: %v", err)
	}
}
