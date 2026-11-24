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

package options

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value/compression"
)

type fakeResourceRules map[schema.GroupResource]compression.Rule

func (f fakeResourceRules) RuleFor(gr schema.GroupResource) compression.Rule { return f[gr] }

// TestCRDRESTOptionsCarryCompressionRule checks that a custom resource's
// REST options carry its compression Rule.
func TestCRDRESTOptionsCarryCompressionRule(t *testing.T) {
	widgets := schema.GroupResource{Group: "example.com", Resource: "widgets"}
	deflate := compression.Rule{Algorithm: compression.AlgorithmDeflate, MinSize: 1024}
	etcdOptions := genericoptions.NewEtcdOptions(storagebackend.NewDefaultConfig("", nil))

	getter := NewCRDRESTOptionsGetter(*etcdOptions, nil, fakeResourceRules{widgets: deflate}, nil)
	restOptions, err := getter.GetRESTOptions(widgets, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := restOptions.StorageConfig.CompressionRule; got != deflate {
		t.Errorf("got Rule %+v, want %+v", got, deflate)
	}
}
