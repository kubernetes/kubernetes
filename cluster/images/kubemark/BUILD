package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//docker:docker.bzl", "docker_build", "docker_push")

docker_build(
    name = "image",
    base = "@official_busybox//image",
    entrypoint = ["/kubemark"],
    files = ["//cmd/kubemark"],
)

docker_push(
    name = "push",
    image = ":image",
    registry = "gcr.io",
    repository = "$(PROJECT)/kubemark",
    stamp = True,
    tag = "latest",
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
