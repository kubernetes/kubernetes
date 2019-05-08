load("@io_bazel_rules_docker//container:container.bzl", "container_layer")
load("//build:platforms.bzl", "SERVER_PLATFORMS")
load("//build:container.bzl", "multi_arch_container")

container_layer(
    name = "cluster-srcs",
    data_path = "/",
    directory = "/kubernetes",
    files = ["//cluster:all-srcs"],
)

container_layer(
    name = "bins",
    directory = "/usr/local/bin",
    files = [
        "//cmd/kubectl",
        "//test/e2e:e2e.test_binary",
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
)

multi_arch_container(
    name = "conformance",
    architectures = SERVER_PLATFORMS["linux"],
    base = "@debian-hyperkube-base-{ARCH}//image",
    cmd = [
        "/bin/bash",
        "-c",
        "/run_e2e.sh",
    ],
    # {ARCH} is replaced by the macro, but STABLE_ vars are replaced by the
    # build stamping, so we need to escape them
    docker_push_tags = ["{{STABLE_DOCKER_PUSH_REGISTRY}}/conformance-{ARCH}:{{STABLE_DOCKER_TAG}}"],
    docker_tags = ["{{STABLE_DOCKER_REGISTRY}}/conformance-{ARCH}:{{STABLE_DOCKER_TAG}}"],
    env = {
        "E2E_FOCUS": "\[Conformance\]",
        "E2E_SKIP": "",
        "E2E_PARALLEL": "1",
        "E2E_PROVIDER": "local",
        "RESULTS_DIR": "/tmp/results",
        "KUBECONFIG": "",
    },
    files = [
        ":run_e2e.sh",
    ],
    layers = [
        ":cluster-srcs",
        ":bins",
    ],
    stamp = True,
    tags = ["manual"],
    visibility = ["//visibility:public"],
    workdir = "/usr/local/bin",
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
