load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image")

container_image(
    name = "hyperkube-internal",
    base = "@debian-hyperkube-base-amd64//image",
    files = [
        "//cmd/hyperkube",
    ],
    stamp = True,
    symlinks = {
        "/%s" % path: "/hyperkube"
        for path in [
            "/apiserver",
            "/controller-manager",
            "/kubectl",
            "/kubelet",
            "/proxy",
            "/scheduler",
            "/usr/local/bin/kube-apiserver",
            "/usr/local/bin/kube-controller-manager",
            "/usr/local/bin/kubectl",
            "/usr/local/bin/kubelet",
            "/usr/local/bin/kube-proxy",
            "/usr/local/bin/kube-scheduler",
        ]
    },
)

container_bundle(
    name = "hyperkube",
    images = {"k8s.gcr.io/hyperkube-amd64:{STABLE_DOCKER_TAG}": "hyperkube-internal"},
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
