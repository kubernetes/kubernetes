/*
Copyright 2014 The Kubernetes Authors.

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

// Package kubectl provides Kubernetes-repository-specific helpers used by the
// kubectl command line tool.
//
// 功能分析：
//  1. 当前 Kubernetes 仓库中的 kubectl 大部分实现已经位于
//     staging/src/k8s.io/kubectl，那里是可独立发布的 kubectl 模块源码。
//  2. 本 pkg/kubectl 包只保留仍然需要依赖 k8s.io/kubernetes 内部代码的少量逻辑，
//     典型例子是 cmd/convert，它需要使用 pkg/api/legacyscheme 和 pkg/apis/*/install
//     完成历史 API 版本转换。
//  3. cmd/kubectl/main 仍应保持薄入口，只负责进程初始化和调用 staging kubectl 命令树。
//
// 注意点：
//  1. 新增 kubectl 普通子命令时，优先放到 staging/src/k8s.io/kubectl/pkg/cmd，不应放到
//     pkg/kubectl。
//  2. 只有当实现确实需要依赖 k8s.io/kubernetes/pkg 下的内部 API、legacy scheme 或其他
//     不能发布到 staging 的代码时，才适合放在这里。
//  3. 这里的代码仍应保持可测试，避免把逻辑塞回 cmd/kubectl/main。
package kubectl
