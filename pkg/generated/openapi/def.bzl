# Copyright 2017 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@io_bazel_rules_go//go:def.bzl", "go_library")
load("@io_kubernetes_build//defs:go.bzl", "go_genrule")

def openapi_library(name, tags, srcs, go_prefix, vendor_prefix = "", openapi_targets = [], vendor_targets = []):
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
    ] + ["//%s:go_default_library" % target for target in openapi_targets] + ["//staging/src/%s:go_default_library" % target for target in vendor_targets]
    go_library(
        name = name,
        srcs = srcs + [":zz_generated.openapi"],
        importpath = go_prefix + "pkg/generated/openapi",
        tags = tags,
        deps = deps,
    )
    go_genrule(
        name = "zz_generated.openapi",
        srcs = ["//" + vendor_prefix + "hack/boilerplate:boilerplate.go.txt"],
        outs = ["zz_generated.openapi.go"],
        # In order for vendored dependencies to be imported correctly,
        # the generator must run from the repo root inside the generated GOPATH.
        # All of bazel's $(location)s are relative to the original working directory, however,
        # so we must save it first.
        cmd = " ".join([
            "cd $$GOPATH/src/" + go_prefix + ";",
            "$$GO_GENRULE_EXECROOT/$(location //vendor/k8s.io/kube-openapi/cmd/openapi-gen)",
            "--v 1",
            "--logtostderr",
            "--go-header-file $$GO_GENRULE_EXECROOT/$(location //" + vendor_prefix + "hack/boilerplate:boilerplate.go.txt)",
            "--output-file-base zz_generated.openapi",
            "--output-package " + go_prefix + "pkg/generated/openapi",
            "--report-filename tmp_api_violations.report",
            "--input-dirs " + ",".join([go_prefix + target for target in openapi_targets] + [go_prefix + "vendor/" + target for target in vendor_targets]),
            "&& cp $$GOPATH/src/" + go_prefix + "pkg/generated/openapi/zz_generated.openapi.go $$GO_GENRULE_EXECROOT/$(location :zz_generated.openapi.go)",
            "&& rm tmp_api_violations.report",
        ]),
        go_deps = deps,
        tools = ["//vendor/k8s.io/kube-openapi/cmd/openapi-gen"],
    )
