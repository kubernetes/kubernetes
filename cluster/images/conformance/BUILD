load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image", "container_layer")

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
        "//test/e2e:e2e.test",
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
)

container_image(
    name = "conformance-internal",
    base = "@debian-hyperkube-base-amd64//image",
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

container_bundle(
    name = "conformance",
    images = {"k8s.gcr.io/conformance-amd64:{STABLE_DOCKER_TAG}": "conformance-internal"},
    stamp = True,
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
