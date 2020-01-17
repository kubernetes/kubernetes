load("@io_bazel_rules_docker//container:container.bzl", "container_layer")
load("//build:container.bzl", "multi_arch_container")
load("//build:platforms.bzl", "SERVER_PLATFORMS")

container_layer(
    name = "scripts",
    directory = "/",
    files = [
        "hyperkube",
    ],
)

container_layer(
    name = "bins",
    directory = "/usr/local/bin",
    files = [
        "//cmd/kube-apiserver",
        "//cmd/kube-controller-manager",
        "//cmd/kube-proxy",
        "//cmd/kube-scheduler",
        "//cmd/kubectl",
        "//cmd/kubelet",
    ],
)

multi_arch_container(
    name = "image",
    architectures = SERVER_PLATFORMS["linux"],
    base = "@debian-hyperkube-base-{ARCH}//image",
    cmd = [
        "/hyperkube",
    ],
    # {ARCH} is replaced by the macro, but STABLE_ vars are replaced by the
    # build stamping, so we need to escape them
    docker_push_tags = ["{{STABLE_DOCKER_PUSH_REGISTRY}}/hyperkube-{ARCH}:{{STABLE_DOCKER_TAG}}"],
    docker_tags = ["{{STABLE_DOCKER_REGISTRY}}/hyperkube-{ARCH}:{{STABLE_DOCKER_TAG}}"],
    layers = [
        ":bins",
        ":scripts",
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
