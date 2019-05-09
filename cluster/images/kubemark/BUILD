package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//container:container.bzl", "container_image", "container_push")

container_image(
    name = "image",
    base = "@official_busybox//image",
    entrypoint = ["/kubemark"],
    files = ["//cmd/kubemark"],
    stamp = True,
)

container_push(
    name = "push",
    format = "Docker",
    image = ":image",
    registry = "$(REGISTRY)",
    repository = "kubemark",
    stamp = True,
    tag = "$(IMAGE_TAG)",
    tags = ["manual"],
)

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
