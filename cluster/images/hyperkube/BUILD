load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image")
load("//build:platforms.bzl", "SERVER_PLATFORMS", "for_platforms")

container_image(
    name = "hyperkube-internal",
    base = select(for_platforms(
        for_server = "@debian-hyperkube-base-{ARCH}//image",
        only_os = "linux",
    )),
    files = [
        "//cmd/hyperkube",
    ],
    stamp = True,
)

[container_bundle(
    name = "hyperkube-%s" % arch,
    images = {"k8s.gcr.io/hyperkube-%s:{STABLE_DOCKER_TAG}" % arch: "hyperkube-internal"},
    tags = ["manual"],
    visibility = ["//visibility:public"],
) for arch in SERVER_PLATFORMS["linux"]]

[alias(
    name = "hyperkube%s" % suffix,
    actual = select(for_platforms(for_server = "hyperkube-{ARCH}%s" % suffix)),
) for suffix in [
    "",
    ".tar",
]]

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
