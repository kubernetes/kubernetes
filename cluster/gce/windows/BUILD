load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")

# Having the Windows code from the GCE cluster deploy hosted with the release is
# useful for GKE.  This list should match the list in k8s.io/release/lib/releaselib.sh.

# IMPORTANT PLEASE NOTE:
# Any time the file structure in the `windows` directory changes, `windows/BUILD`
# and `k8s.io/release/lib/releaselib.sh` must be manually updated with the changes.
# We HIGHLY recommend not changing the file structure, because consumers of
# Kubernetes releases depend on the release structure remaining stable.
release_filegroup(
    name = "gcs-release-artifacts",
    srcs = [
        "common.psm1",
        "configure.ps1",
        "k8s-node-setup.psm1",
        "testonly/install-ssh.psm1",
        "testonly/user-profile.psm1",
    ],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
