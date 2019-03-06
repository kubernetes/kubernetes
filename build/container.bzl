# Copyright 2019 The Kubernetes Authors.
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

load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image")
load("//build:platforms.bzl", "go_platform_constraint")

# multi_arch_container produces a private internal container_image, multiple
# arch-specific tagged container_bundles (named NAME-ARCH) and aliases
# from NAME and NAME.tar to the appropriately NAME-ARCH container_bundle target
# for the currently-configured architecture.
# Args:
#   name: name used for the alias; the internal container_image and
#     container_bundles are based on this name
#   architectures: list of architectures (in GOARCH naming parlance) to
#     configure
#   base: base image to use for the containers. The format string {ARCH} will
#     be replaced with the configured GOARCH.
#   docker_tags: list of docker tags to apply to the image. The format string
#     {ARCH} will be replaced with the configured GOARCH; any stamping variables
#     should be escaped, e.g. {{STABLE_MY_VAR}}.
#   tags: will be applied to all rules
#   visiblity: will be applied only to the container_bundles; the internal
#     container_image is private
#   All other args will be applied to the internal container_image.
def multi_arch_container(
        name,
        architectures,
        base,
        docker_tags,
        tags = None,
        visibility = None,
        **kwargs):
    container_image(
        name = "%s-internal" % name,
        base = select({
            go_platform_constraint(os = "linux", arch = arch): base.format(ARCH = arch)
            for arch in architectures
        }),
        tags = tags,
        visibility = ["//visibility:private"],
        **kwargs
    )

    for arch in architectures:
        container_bundle(
            name = "%s-%s" % (name, arch),
            images = {
                docker_tag.format(ARCH = arch): ":%s-internal" % name
                for docker_tag in docker_tags
            },
            tags = tags,
            visibility = visibility,
        )
    for suffix in ["", ".tar"]:
        native.alias(
            name = "%s%s" % (name, suffix),
            actual = select({
                go_platform_constraint(os = "linux", arch = arch): "%s-%s%s" % (name, arch, suffix)
                for arch in architectures
            }),
        )
