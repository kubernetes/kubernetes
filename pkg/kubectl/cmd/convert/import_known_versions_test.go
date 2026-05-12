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

package convert

import (
	"sort"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

func TestKnownVersions(t *testing.T) {
	legacytypes := legacyscheme.Scheme.AllKnownTypes()
	alreadyErroredGroups := map[string]bool{}
	var gvks []schema.GroupVersionKind
	for gvk := range scheme.Scheme.AllKnownTypes() {
		gvks = append(gvks, gvk)
	}
	sort.Slice(gvks, func(i, j int) bool {
		if isWatchEvent1, isWatchEvent2 := gvks[i].Kind == "WatchEvent", gvks[j].Kind == "WatchEvent"; isWatchEvent1 != isWatchEvent2 {
			return isWatchEvent2
		}
		if isList1, isList2 := strings.HasSuffix(gvks[i].Kind, "List"), strings.HasSuffix(gvks[j].Kind, "List"); isList1 != isList2 {
			return isList2
		}
		if isOptions1, isOptions2 := strings.HasSuffix(gvks[i].Kind, "Options"), strings.HasSuffix(gvks[j].Kind, "Options"); isOptions1 != isOptions2 {
			return isOptions2
		}
		if isInternal1, isInternal2 := gvks[i].Group == runtime.APIVersionInternal, gvks[j].Group == runtime.APIVersionInternal; isInternal1 != isInternal2 {
			return isInternal2
		}
		return gvks[i].String() < gvks[j].String()
	})
	for _, gvk := range gvks {
		if alreadyErroredGroups[gvk.Group] {
			continue
		}
		if _, legacyregistered := legacytypes[gvk]; !legacyregistered {
			t.Errorf("%v is not registered in legacyscheme. Add group %q (all internal and external versions) to convert/import_known_versions.go", gvk, gvk.Group)
			alreadyErroredGroups[gvk.Group] = true
		}
	}
}
