# Copyright 2018 The Kubernetes Authors.
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

load("//build:kazel_generated.bzl", "go_prefix", "tags_values_pkgs")
load("//build:openapi.bzl", "openapi_vendor_prefix")
load("@io_k8s_repo_infra//defs:go.bzl", "go_genrule")
load("@bazel_skylib//lib:paths.bzl", "paths")

def bazel_go_library(pkg):
    """Returns the Bazel label for the Go library for the provided package.
    This is intended to be used with the //build:kazel_generated.bzl tag dictionaries; for example:
    load("//build:kazel_generated.bzl", "tags_values_pkgs")
    some_rule(
        ...
        deps = [bazel_go_library(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]],
        ...
    )
    """
    return "//%s:go_default_library" % pkg

def go_pkg(pkg):
    """Returns the full Go package name for the provided workspace-relative package.
    This is suitable to pass to tools depending on the Go build library.
    If any packages are in staging/src, they are remapped to their intended path in vendor/.
    This is intended to be used with the //build:kazel_generated.bzl tag dictionaries.
    For example:
    load("//build:kazel_generated.bzl", "tags_values_pkgs")
    genrule(
        ...
        cmd = "do something --pkgs=%s" % ",".join([go_pkg(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]]),
        ...
    )
    """
    for prefix in ["staging/src", "vendor"]:
        if pkg.startswith(prefix):
            return paths.relativize(pkg, prefix)
    return paths.join(go_prefix, pkg)

def openapi_deps():
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
    ]
    deps.extend([bazel_go_library(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]])
    return deps

def applies(pkg, prefixes, default):
    if prefixes == None or len(prefixes) == 0:
        return default
    for prefix in prefixes:
        if pkg == prefix or pkg.startswith(prefix + "/"):
            return True
    return False

def gen_openapi(outs, output_pkg, include_pkgs=[], exclude_pkgs=[]):
    """Calls openapi-gen to produce the zz_generated.openapi.go file,
    which should be provided in outs.
    output_pkg should be set to the full go package name for this generated file.
    """
    go_genrule(
        name = "zz_generated.openapi",
        srcs = ["//" + openapi_vendor_prefix + "hack/boilerplate:boilerplate.generatego.txt"],
        outs = outs,
        # In order for vendored dependencies to be imported correctly,
        # the generator must run from the repo root inside the generated GOPATH.
        # All of bazel's $(location)s are relative to the original working directory, however.
        cmd = " ".join([
            "$(location //vendor/k8s.io/kube-openapi/cmd/openapi-gen)",
            "--v 1",
            "--logtostderr",
            "--go-header-file $(location //" + openapi_vendor_prefix + "hack/boilerplate:boilerplate.generatego.txt)",
            "--output-file-base zz_generated.openapi",
            "--output-package " + output_pkg,
            "--report-filename tmp_api_violations.report",
            "--input-dirs " + ",".join([go_pkg(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"] if applies(pkg, include_pkgs, True) and not applies(pkg, exclude_pkgs, False)]),
            "&& cp $$GOPATH/src/" + output_pkg + "/zz_generated.openapi.go $(location :zz_generated.openapi.go)",
            "&& rm tmp_api_violations.report",
        ]),
        go_deps = openapi_deps(),
        tools = ["//vendor/k8s.io/kube-openapi/cmd/openapi-gen"],
        message = "GenOpenAPI",
    )
