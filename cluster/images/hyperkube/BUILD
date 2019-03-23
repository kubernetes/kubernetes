load("//build:container.bzl", "multi_arch_container")
load("//build:platforms.bzl", "SERVER_PLATFORMS")

multi_arch_container(
    name = "hyperkube",
    architectures = SERVER_PLATFORMS["linux"],
    base = "@debian-hyperkube-base-{ARCH}//image",
    # {ARCH} is replaced by the macro, but STABLE_ vars are replaced by the
    # build stamping, so we need to escape them
    docker_push_tags = ["{{STABLE_DOCKER_PUSH_REGISTRY}}/hyperkube-{ARCH}:{{STABLE_DOCKER_TAG}}"],
    docker_tags = ["{{STABLE_DOCKER_REGISTRY}}/hyperkube-{ARCH}:{{STABLE_DOCKER_TAG}}"],
    files = [
        "//cmd/hyperkube",
    ],
    stamp = True,
    tags = ["manual"],
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
