package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

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
)

# TODO(#3579): This is a temporary hack. It gathers up the yaml,
# yaml.in, json files in cluster/addons (minus any demos) and overlays
# them into kube-addons, where we expect them.
# These files are expected in a salt/kube-addons subdirectory.
pkg_tar(
    name = "_salt_kube-addons",
    package_dir = "salt/kube-addons",
    strip_prefix = "/cluster/addons",
    visibility = ["//visibility:private"],
    deps = [
        "//cluster/addons",
    ],
)

pkg_tar(
    name = "salt",
    files = glob(
        ["**"],
        exclude = ["BUILD"],
    ),
    mode = "0644",
    modes = {
        "install.sh": "0755",
    },
    package_dir = "kubernetes/saltbase",
    strip_prefix = ".",
    deps = [
        ":_salt_kube-addons",
    ],
)

# The following are used in the kubernetes salt tarball.
pkg_tar(
    name = "salt-manifests",
    files = [
        "salt/fluentd-gcp/fluentd-gcp.yaml",
        "salt/kube-proxy/kube-proxy.manifest",
        "salt/kube-registry-proxy/kube-registry-proxy.yaml",
    ],
    mode = "0644",
)

pkg_tar(
    name = "_kube-admission-controls",
    files = glob(["salt/kube-admission-controls/limit-range/**"]),
    mode = "0644",
    # Maintain limit-range/ subdirectory in tarball
    strip_prefix = "./salt/kube-admission-controls/",
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "gci-trusty-salt-manifests",
    files = [
        "salt/cluster-autoscaler/cluster-autoscaler.manifest",
        "salt/e2e-image-puller/e2e-image-puller.manifest",
        "salt/etcd/etcd.manifest",
        "salt/kube-addons/kube-addon-manager.yaml",
        "salt/kube-apiserver/abac-authz-policy.jsonl",
        "salt/kube-apiserver/kube-apiserver.manifest",
        "salt/kube-controller-manager/kube-controller-manager.manifest",
        "salt/kube-scheduler/kube-scheduler.manifest",
        "salt/l7-gcp/glbc.manifest",
        "salt/rescheduler/rescheduler.manifest",
    ],
    mode = "0644",
    deps = [
        "_kube-admission-controls",
    ],
)
