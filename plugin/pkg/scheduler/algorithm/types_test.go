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

package algorithm

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// EmptyMetadataProducer should returns a no-op MetadataProducer type.
func TestEmptyMetadataProducer(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1"}}

	nodeNameToInfo := map[string]*schedulercache.NodeInfo{
		"3": schedulercache.NewNodeInfo(),
		"2": schedulercache.NewNodeInfo(),
		"1": schedulercache.NewNodeInfo(pod),
	}
	err := EmptyMetadataProducer(pod, nodeNameToInfo)
	if err != nil {
		t.Errorf("failed to produce empty metadata")
		return
	}
}
