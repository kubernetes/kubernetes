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

package convert

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

type testcase struct {
	name          string
	file          string
	outputVersion string
	fields        []checkField
}

type checkField struct {
	expected string
}

// TestConvertObject 验证 kubectl convert 能把典型内置资源转换为用户指定的目标 API 版本。
//
// 功能分析：
//  1. 使用 test/fixtures/pkg/kubectl/cmd/convert 下的历史 manifest 作为输入，覆盖
//     Deployment、Ingress 以及多对象文件等转换场景。
//  2. 通过 NewCmdConvert 构造真实 Cobra 命令，并设置 --filename、--output-version、
//     --local=true 和 -o yaml，尽量贴近用户运行 kubectl convert 的路径。
//  3. 用 fake REST client 拦截任何网络请求；如果本地转换路径意外访问 apiserver，测试会
//     立即失败。
//  4. 最终只断言输出中包含期望 apiVersion，聚焦 convert 命令最核心的版本转换结果。
//
// 注意点：
//  1. 该测试依赖 import_known_versions.go 的 blank imports。如果某个 API 组没有注册到
//     legacy scheme，转换过程可能无法找到目标版本。
//  2. 这里选择局部字段断言而不是完整 YAML golden，是为了避免 printer 的格式细节变化让
//     测试过度脆弱。
//  3. 增加新 fixture 时，应优先覆盖真实迁移路径，例如 deprecated API 到当前 API 的转换。
func TestConvertObject(t *testing.T) {
	testcases := []testcase{
		{
			name:          "apps deployment to extensions deployment",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/appsdeployment.yaml",
			outputVersion: "extensions/v1beta1",
			fields: []checkField{
				{
					expected: "apiVersion: extensions/v1beta1",
				},
			},
		},
		{
			name:          "extensions deployment to apps deployment",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/extensionsdeployment.yaml",
			outputVersion: "apps/v1beta2",
			fields: []checkField{
				{
					expected: "apiVersion: apps/v1beta2",
				},
			},
		},
		{
			name:          "v1beta1 Ingress to extensions Ingress",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/v1beta1ingress.yaml",
			outputVersion: "extensions/v1beta1",
			fields: []checkField{
				{
					expected: "apiVersion: extensions/v1beta1",
				},
			},
		},
		{
			name:          "converting multiple including service to neworking.k8s.io/v1",
			file:          "../../../../test/fixtures/pkg/kubectl/cmd/convert/serviceandingress.yaml",
			outputVersion: "networking.k8s.io/v1",
			fields: []checkField{
				{
					expected: "apiVersion: networking.k8s.io/v1",
				},
			},
		},
	}

	for _, tc := range testcases {
		for _, field := range tc.fields {
			t.Run(fmt.Sprintf("%s %s", tc.name, field), func(t *testing.T) {
				tf := cmdtesting.NewTestFactory().WithNamespace("test")
				defer tf.Cleanup()

				tf.UnstructuredClient = &fake.RESTClient{
					Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}),
				}

				buf := bytes.NewBuffer([]byte{})
				cmd := NewCmdConvert(tf, genericiooptions.IOStreams{Out: buf, ErrOut: buf})
				cmd.Flags().Set("filename", tc.file)
				cmd.Flags().Set("output-version", tc.outputVersion)
				cmd.Flags().Set("local", "true")
				cmd.Flags().Set("output", "yaml")
				cmd.Run(cmd, []string{})
				if !strings.Contains(buf.String(), field.expected) {
					t.Errorf("unexpected output when converting %s to %q, expected: %q, but got %q", tc.file, tc.outputVersion, field.expected, buf.String())
				}
			})
		}
	}
}
