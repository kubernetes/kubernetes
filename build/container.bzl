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
load("@io_bazel_rules_docker//contrib:push-all.bzl", "docker_push")
load("//build:platforms.bzl", "go_platform_constraint")

# multi_arch_container produces a private internal container_image, multiple
# arch-specific tagged container_bundles (named NAME-ARCH), an alias
# from NAME to the appropriately NAME-ARCH container_bundle target, and a
# genrule for NAME.tar copying the appropriate NAME-ARCH container bundle
# tarball output for the currently-configured architecture.
# Additionally, if docker_push_tags is provided, uses multi_arch_container_push
# to create container_bundles named push-NAME-ARCH with the provided push tags,
# along with a push-NAME docker_push target.
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
#   docker_push_tags: list of docker tags to apply to the image for pushing.
#     The format string {ARCH} will be replaced with the configured GOARCH;
#     any stamping variables should be escaped, e.g. {{STABLE_MY_VAR}}.
#   tags: will be applied to all targets
#   visiblity: will be applied only to the container_bundles; the internal
#     container_image is private
#   All other args will be applied to the internal container_image.
def multi_arch_container(
        name,
        architectures,
        base,
        docker_tags,
        docker_push_tags = None,
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
    native.alias(
        name = name,
        actual = select({
            go_platform_constraint(os = "linux", arch = arch): "%s-%s" % (name, arch)
            for arch in architectures
        }),
    )
    native.genrule(
        name = "gen_%s.tar" % name,
        outs = ["%s.tar" % name],
        srcs = select({
            go_platform_constraint(os = "linux", arch = arch): ["%s-%s.tar" % (name, arch)]
            for arch in architectures
        }),
        cmd = "cp $< $@",
        output_to_bindir = True,
    )

    if docker_push_tags:
        multi_arch_container_push(
            name = name,
            architectures = architectures,
            docker_tags_images = {docker_push_tag: ":%s-internal" % name for docker_push_tag in docker_push_tags},
            tags = tags,
        )

# multi_arch_container_push creates container_bundles named push-NAME-ARCH for
# the provided architectures, populating them with the images directory.
# It additionally creates a push-NAME docker_push rule which can be run to
# push the images to a Docker repository.
# Args:
#   name: name used for targets created by this macro; the internal
#     container_bundles are based on this name
#   architectures: list of architectures (in GOARCH naming parlance) to
#     configure
#   docker_tags_images: dictionary mapping docker tag to the corresponding
#     container_image target. The format string {ARCH} will be replaced
#     in tags with the configured GOARCH; any stamping variables should be
#     escaped, e.g. {{STABLE_MY_VAR}}.
#   tags: applied to container_bundle targets
def multi_arch_container_push(
        name,
        architectures,
        docker_tags_images,
        tags = None):
    for arch in architectures:
        container_bundle(
            name = "push-%s-%s" % (name, arch),
            images = {tag.format(ARCH = arch): image for tag, image in docker_tags_images.items()},
            tags = tags,
            visibility = ["//visibility:private"],
        )
    docker_push(
        name = "push-%s" % name,
        bundle = select({
            go_platform_constraint(os = "linux", arch = arch): "push-%s-%s" % (name, arch)
            for arch in architectures
        }),
    )
