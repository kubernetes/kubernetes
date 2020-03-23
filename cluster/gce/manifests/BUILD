package(default_visibility = ["//visibility:public"])

load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "gce-master-manifests",
    srcs = [":manifests"],
    mode = "0644",
)

# if you update this, also update function kube::release::package_kube_manifests_tarball() in build/lib/release.sh
filegroup(
    name = "manifests",
    srcs = [
        "abac-authz-policy.jsonl",
        "cluster-autoscaler.manifest",
        "etcd.manifest",
        "etcd-empty-dir-cleanup.yaml",
        "glbc.manifest",
        "konnectivity-server.yaml",
        "kube-addon-manager.yaml",
        "kube-apiserver.manifest",
        "kube-controller-manager.manifest",
        "kube-proxy.manifest",
        "kube-scheduler.manifest",
    ] + glob(["internal-*"]),
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
