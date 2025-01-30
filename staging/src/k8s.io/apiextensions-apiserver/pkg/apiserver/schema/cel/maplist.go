/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/model"
	"k8s.io/apiserver/pkg/cel/common"
)

// makeMapList returns a queryable interface over the provided x-kubernetes-list-type=map
// keyedItems. If the provided schema is _not_ an array with x-kubernetes-list-type=map, returns an
// empty mapList.
func makeMapList(sts *schema.Structural, items []interface{}) (rv common.MapList) {
	return common.MakeMapList(&model.Structural{Structural: sts}, items)
}
