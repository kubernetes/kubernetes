load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image")

container_image(
    name = "hyperkube-internal",
    base = "@debian-hyperkube-base-amd64//image",
    files = [
        "//cmd/hyperkube",
    ],
    stamp = True,
)

container_bundle(
    name = "hyperkube",
    images = {"k8s.gcr.io/hyperkube-amd64:{STABLE_DOCKER_TAG}": "hyperkube-internal"},
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
