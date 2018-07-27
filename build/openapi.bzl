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

load("//build:code_generation.bzl", "bazel_go_library", "go_pkg", "go_prefix")
load("@io_k8s_code_generation//:tags.bzl", "tags_values_pkgs")
load("@io_kubernetes_build//defs:go.bzl", "go_genrule")

# A project wanting to generate openapi code for vendored
# k8s.io/kubernetes will need to set the following variables in
# //build/openapi.bzl in their project and customize the go prefix:
#
# openapi_vendor_prefix = "vendor/k8s.io/kubernetes/"

openapi_vendor_prefix = ""

def openapi_deps():
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
    ]
    deps.extend([bazel_go_library(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]])
    return deps

# Calls openapi-gen to produce the zz_generated.openapi.go file, which should be provided in outs.
# output_pkg should be set to the full go package name for this generated file.
def gen_openapi(outs, output_pkg):
    go_genrule(
        name = "zz_generated.openapi",
        srcs = ["//" + openapi_vendor_prefix + "hack/boilerplate:boilerplate.generatego.txt"],
        outs = outs,
        # In order for vendored dependencies to be imported correctly,
        # the generator must run from the repo root inside the generated GOPATH.
        # All of bazel's $(location)s are relative to the original working directory, however,
        # so we must save it first.
        cmd = " ".join([
            "ORIG_WD=$$(pwd);",
            "cd $$GOPATH/src/" + go_prefix + ";",
            "$$ORIG_WD/$(location //vendor/k8s.io/kube-openapi/cmd/openapi-gen)",
            "--v 1",
            "--logtostderr",
            "--go-header-file $$ORIG_WD/$(location //" + openapi_vendor_prefix + "hack/boilerplate:boilerplate.generatego.txt)",
            "--output-file-base zz_generated.openapi",
            "--output-package " + output_pkg,
            "--report-filename tmp_api_violations.report",
            "--input-dirs " + ",".join([go_pkg(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]]),
            "&& cp $$GOPATH/src/" + output_pkg + "/zz_generated.openapi.go $$ORIG_WD/$(location :zz_generated.openapi.go)",
        ]),
        go_deps = openapi_deps(),
        tools = ["//vendor/k8s.io/kube-openapi/cmd/openapi-gen"],
    )
