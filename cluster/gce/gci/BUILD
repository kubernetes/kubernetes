load("@io_bazel_rules_go//go:def.bzl", "go_test")
load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")
load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")

go_test(
    name = "go_default_test",
    srcs = [
        "apiserver_manifest_test.go",
        "configure_helper_test.go",
    ],
    data = [
        ":scripts-test-data",
        "//cluster/gce/manifests",
    ],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
    ],
)

# Having the COS code from the GCE cluster deploy hosted with the release is
# useful for GKE.  This list should match the list in
# kubernetes/release/lib/releaselib.sh.
release_filegroup(
    name = "gcs-release-artifacts",
    srcs = [
        "configure.sh",
        "master.yaml",
        "node.yaml",
        "shutdown.sh",
    ],
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "gci-trusty-manifests",
    srcs = glob(["gke-internal-configure-helper.sh"]),
    files = {
        "//cluster/gce/gci/mounter": "gci-mounter",
        "configure-helper.sh": "gci-configure-helper.sh",
        "health-monitor.sh": "health-monitor.sh",
    },
    mode = "0755",
    strip_prefix = ".",
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
    srcs = [
        ":package-srcs",
        "//cluster/gce/gci/mounter:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "scripts-test-data",
    srcs = [
        "configure-helper.sh",
    ],
)
