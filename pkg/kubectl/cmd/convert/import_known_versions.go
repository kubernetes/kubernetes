/*
Copyright 2016 The Kubernetes Authors.

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

// These imports are the API groups the client will support.
// TODO: Remove these manual install once we don't need legacy scheme in convert
//
// 功能分析：
//  1. 这些 blank imports 通过各 API 组的 install 包把 internal/external 类型、默认值、
//     转换函数和 scheme 信息注册到 Kubernetes legacy scheme。
//  2. kubectl convert 在本地转换 manifest 时依赖这个 scheme 判断对象 GVK，并调用
//     ConvertToVersion 把对象转成目标 API 版本。
//  3. 该文件本身没有可调用函数，但它的 init side effect 是 convert 命令能够识别内置
//     Kubernetes API 组的关键。
//
// 注意点：
//  1. 新增内置 API 组或让 convert 支持新的历史版本时，需要在这里添加对应 install 包，
//     并确保 internal 和 external 版本都注册完整。
//  2. deprecated API 组放在最后，避免 prioritized versions 选择时优先落到旧版本。
//  3. import_known_versions_test.go 会对 client-go scheme 与 legacy scheme 的已知类型做
//     对比，漏加 API 组通常会在该测试中暴露。
import (
	_ "k8s.io/kubernetes/pkg/apis/admission/install"
	_ "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	_ "k8s.io/kubernetes/pkg/apis/apiserverinternal/install"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/authentication/install"
	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/coordination/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/discovery/install"
	_ "k8s.io/kubernetes/pkg/apis/events/install"
	_ "k8s.io/kubernetes/pkg/apis/flowcontrol/install"
	_ "k8s.io/kubernetes/pkg/apis/imagepolicy/install"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	_ "k8s.io/kubernetes/pkg/apis/node/install"
	_ "k8s.io/kubernetes/pkg/apis/policy/install"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
	_ "k8s.io/kubernetes/pkg/apis/storagemigration/install"

	// Put the deprecated apis last to ensure that the latest apis can be used first.
	// Related issue: https://github.com/kubernetes/kubernetes/issues/112682
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
)
