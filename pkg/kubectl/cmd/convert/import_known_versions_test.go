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

// TestKnownVersions 确认 kubectl convert 使用的 legacy scheme 覆盖 client-go 已知的
// Kubernetes 内置类型。
//
// 功能分析：
//  1. 从 legacyscheme.Scheme 读取 convert 实际可用的类型集合。
//  2. 从 client-go scheme 读取客户端通常知道的 Kubernetes API 类型集合。
//  3. 对 client-go 中的每个 GVK 检查 legacy scheme 是否也已注册；如果某个 API 组缺失，
//     convert/import_known_versions.go 就需要补充对应 install 包。
//  4. 排序时把 WatchEvent、List、Options 和 internal 类型放在后面，使失败输出更可能先
//     指向用户真正关心的资源类型，而不是辅助类型。
//
// 注意点：
//  1. alreadyErroredGroups 用来让同一个 API group 只报一次错，避免一个 group 漏注册时刷出
//     大量重复类型错误。
//  2. 该测试保护的是注册完整性，不直接验证转换语义；具体转换结果由 convert_test.go 覆盖。
//  3. 新增 API 组后如果这里失败，通常不是测试需要放宽，而是 legacy scheme 的 blank import
//     没有同步更新。
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
