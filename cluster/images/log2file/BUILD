load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

container_image(
    name = "log2file-internal",
    cmd = ["/log2file"],
    files = [
        ":log2file",
    ],
    stamp = True,
)

container_bundle(
    name = "log2file",
    images = {"staging-k8s.gcr.io/log2file-amd64:{STABLE_DOCKER_TAG}": "log2file-internal"},
    stamp = True,
)

go_library(
    name = "go_default_library",
    srcs = ["log2file.go"],
    importpath = "k8s.io/kubernetes/cluster/images/log2file",
    visibility = ["//visibility:private"],
)

go_binary(
    name = "log2file",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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
