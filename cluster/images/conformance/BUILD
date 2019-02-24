load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image", "container_layer")
load("//build:platforms.bzl", "SERVER_PLATFORMS", "for_platforms")

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

container_image(
    name = "conformance-internal",
    base = select(for_platforms(
        for_server = "@debian-hyperkube-base-{ARCH}//image",
        only_os = "linux",
    )),
    cmd = [
        "/bin/bash",
        "-c",
        "/run_e2e.sh",
    ],
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
    workdir = "/usr/local/bin",
)

[container_bundle(
    name = "conformance-%s" % arch,
    images = {"k8s.gcr.io/conformance-%s:{STABLE_DOCKER_TAG}" % arch: "conformance-internal"},
    tags = ["manual"],
    visibility = ["//visibility:public"],
) for arch in SERVER_PLATFORMS["linux"]]

[alias(
    name = "conformance%s" % suffix,
    actual = select(for_platforms(for_server = "conformance-{ARCH}%s" % suffix)),
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
