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

package cost

import (
	"flag"
	"testing"

	"github.com/go-logr/logr"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/integration/framework"
)

func init() {
	// Each iteration resets 100 pods while the benchmark timer is stopped, so
	// default to 1x iteration unless overridden via -benchtime on the CLI.
	testing.Init()
	_ = flag.Set("test.benchtime", "1x")
}

func TestMain(m *testing.M) {
	klog.SetLogger(logr.Discard())
	framework.EtcdMain(m.Run)
}
