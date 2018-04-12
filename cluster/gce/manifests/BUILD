package(default_visibility = ["//visibility:public"])

load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")
load("@io_kubernetes_build//defs:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "gce-master-manifests",
    srcs = [":manifests"],
    mode = "0644",
)

filegroup(
    name = "manifests",
    srcs = [
        "abac-authz-policy.jsonl",
        "cluster-autoscaler.manifest",
        "e2e-image-puller.manifest",
        "etcd.manifest",
        "glbc.manifest",
        "kms-plugin-container.manifest",
        "kube-addon-manager.yaml",
        "kube-apiserver.manifest",
        "kube-controller-manager.manifest",
        "kube-proxy.manifest",
        "kube-scheduler.manifest",
        "rescheduler.manifest",
    ],
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
