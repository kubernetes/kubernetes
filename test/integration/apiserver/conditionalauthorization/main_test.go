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

package conditionalauthorization

import (
	"testing"

	"go.uber.org/goleak"

	"k8s.io/kubernetes/test/integration/framework"
)

// TestMain adds package-scoped goleak ignores for HTTP transport goroutines
// that survive kube-apiserver teardown. The CRD conversion webhook client
// cache (apiextensions-apiserver's CRConverterFactory → webhook.ClientManager)
// holds strong references to *rest.RESTClient — and therefore to
// *http.Transport — past server.TearDownFn. Client-go's tlsTransportCache
// uses weak references (ClientsAllowTLSCacheGC=beta), but the strong ref from
// the converter factory prevents finalization within a typical test deadline,
// leaving idle persistConn.{readLoop,writeLoop} goroutines that
// framework.goleakFindRetry then polls up to 600 s for.
func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run,
		goleak.IgnoreTopFunction("net/http.(*persistConn).readLoop"),
		goleak.IgnoreTopFunction("net/http.(*persistConn).writeLoop"),
	)
}
