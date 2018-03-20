package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//docker:docker.bzl", "docker_build", "docker_bundle")

filegroup(
    name = "test-targets",
    srcs = [
        "//test/e2e:e2e.test",
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
)

docker_build(
    name = "dind-node-build",
    base = "@dind-test-base//image",
    debs = [
        "//build/debs:kubelet",
        "//build/debs:kubectl",
        "//build/debs:kubeadm",
        "//build/debs:kubernetes-cni.deb",
    ],
    entrypoint = ["/init-wrapper.sh"],
    files = [
        "init-wrapper.sh",
        "start.sh",
        "//build:master-components-amd64",
        "//build/dind-test/util:k8sversion",
    ],
)

docker_bundle(
    name = "dind-node-bundle",
    images = {"gcr.io/google-containers/dind-node:0.1": ":dind-node-build"},
)

docker_build(
    name = "dind-cluster-build",
    base = "@dind-test-base//image",
    data_path = ".",
    debs = [
        "//build/debs:kubectl",
    ],
    entrypoint = ["/init-wrapper.sh"],
    files = [
        "dind-test.sh",
        "init-wrapper.sh",
        "start.sh",
        ":test-targets",
        ":dind-node-bundle.tar",
    ],
)

docker_bundle(
    name = "dind-cluster-bundle",
    images = {"gcr.io/google-containers/dind-cluster:0.1": "dind-cluster-build"},
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//build/dind-test/util:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
