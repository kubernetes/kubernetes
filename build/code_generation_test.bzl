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

load(":code_generation.bzl", "bazel_go_library", "go_pkg")
load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")

def _bazel_go_library_test_impl(ctx):
    env = unittest.begin(ctx)
    test_cases = [
        ("pkg/kubectl/util", "//pkg/kubectl/util:go_default_library"),
        ("vendor/some/third/party", "//vendor/some/third/party:go_default_library"),
        ("staging/src/k8s.io/apimachinery/api", "//staging/src/k8s.io/apimachinery/api:go_default_library"),
    ]
    for input, expected in test_cases:
        asserts.equals(env, expected, bazel_go_library(input))
    return unittest.end(env)

bazel_go_library_test = unittest.make(_bazel_go_library_test_impl)

def _go_pkg_test_impl(ctx):
    env = unittest.begin(ctx)
    test_cases = [
        ("pkg/kubectl/util", "k8s.io/kubernetes/pkg/kubectl/util"),
        ("vendor/some/third/party", "some/third/party"),
        ("staging/src/k8s.io/apimachinery/api", "k8s.io/apimachinery/api"),
    ]
    for input, expected in test_cases:
        asserts.equals(env, expected, go_pkg(input))
    return unittest.end(env)

go_pkg_test = unittest.make(_go_pkg_test_impl)

def code_generation_test_suite(name):
    unittest.suite(
        name,
        bazel_go_library_test,
        go_pkg_test,
    )
