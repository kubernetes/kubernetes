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

load("@io_bazel_rules_go//go:def.bzl",
     "go_test",
     "go_binary",
     "go_prefix",
     _go_library="go_library",
)

def go_library(
    name,
    srcs=None,
    deps=None,
    **kw):
  _go_library(name=name, srcs=srcs, deps=deps, **kw)
  srcs = srcs or []
  deps = deps or []
  # All go files in this package
  native.filegroup(
    name = name+".gofiles",
    srcs = native.glob(["*.go"]),
    visibility = ["//visibility:public"],
  )
  # All non-generated go files in this package
  native.filegroup(
    name = name+".sourcefiles",
    srcs = native.glob(["*.go"], exclude=["zz_generated.*.go"]),
    visibility = ["//visibility:public"],
  )
