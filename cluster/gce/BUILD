package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")

pkg_tar(
    name = "gci-trusty-manifests",
    files = [
        "gci/configure-helper.sh",
        "gci/health-monitor.sh",
        "//cluster/gce/gci/mounter",
    ],
    mode = "0755",
    strip_prefix = ".",
    # pkg_tar doesn't support renaming the files we add, so instead create symlinks.
    symlinks = {
        "gci-configure-helper.sh": "gci/configure-helper.sh",
        "health-monitor.sh": "gci/health-monitor.sh",
        "gci-mounter": "gci/mounter/mounter",
        "trusty-configure-helper.sh": "trusty/configure-helper.sh",
    },
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
        "//cluster/gce/addons:all-srcs",
        "//cluster/gce/gci/mounter:all-srcs",
    ],
    tags = ["automanaged"],
)

# Having the COS code from the GCE cluster deploy hosted with the release is
# useful for GKE.  This list should match the list in
# kubernetes/release/lib/releaselib.sh.
release_filegroup(
    name = "gcs-release-artifacts",
    srcs = [
        "gci/configure.sh",
        "gci/master.yaml",
        "gci/node.yaml",
    ],
)

pkg_tar(
    name = "gce-master-manifests",
    files = [
        "manifests/abac-authz-policy.jsonl",
        "manifests/cluster-autoscaler.manifest",
        "manifests/e2e-image-puller.manifest",
        "manifests/etcd.manifest",
        "manifests/glbc.manifest",
        "manifests/kube-addon-manager.yaml",
        "manifests/kube-apiserver.manifest",
        "manifests/kube-controller-manager.manifest",
        "manifests/kube-proxy.manifest",
        "manifests/kube-scheduler.manifest",
        "manifests/rescheduler.manifest",
    ],
    mode = "0644",
)
