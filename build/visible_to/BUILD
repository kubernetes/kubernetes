# Package groups defined for use in kubernetes visibility rules.
#
# See associated README.md for explanation.
#
# Style suggestions:
#
#  - Sort package group definitions by name.
#
#  - Prefer obvious package group names.
#
#    E.g "pkg_kubectl_cmd_util_CONSUMERS" names a group
#    of packages allowed to depend on (consume) the
#    //pkg/kubectl/cmd/util package.
#
#
#  - A group name ending in _BAD wants to be deleted.
#
#    Such a group wants to contract, rather than expand.
#    It likely exists to permit a legacy unintentional
#    dependency that requires more work to remove.
#
#  - Prefer defining new groups to expanding groups.
#
#    The former permits tight targeting, the latter can
#    allow unnecessary visibility and thus bad deps.
#

package_group(
    name = "COMMON_generators",
    packages = [
        "//cmd/gendocs",
        "//cmd/genman",
        "//cmd/genyaml",
    ],
)

package_group(
    name = "COMMON_release",
    packages = [
        "//build",
        "//build/debs",
        "//build/rpms",
    ],
)

package_group(
    name = "COMMON_testing",
    packages = [
        "//hack",
        "//hack/lib",
        "//hack/make-rules",
        "//test/e2e",
        "//test/e2e/framework",
        "//test/e2e/kubectl",
        "//test/e2e/workload",
        "//test/integration/etcd",
        "//test/integration/framework",
        "//test/integration/kubectl",
    ],
)

package_group(
    name = "FEDERATION_BAD",
    packages = [
        "//federation/cmd/genfeddocs",
        "//federation/cmd/kubefed/app",
        "//federation/pkg/kubefed",
        "//federation/pkg/kubefed/init",
        "//federation/pkg/kubefed/testing",
        "//federation/pkg/kubefed/util",
    ],
)

package_group(
    name = "cluster",
    packages = [
        "//cluster/...",
    ],
)

package_group(
    name = "KUBEADM_BAD",
    packages = [
        "//cmd/kubeadm/app",
        "//cmd/kubeadm/app/cmd",
    ],
)

# Added by ./hack/verify-bazel.sh; should be excluded from
# that script since it makes no sense here.
filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

# Added by ./hack/verify-bazel.sh; should be excluded from
# that script since it makes no sense here.
filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//build/visible_to:COMMON_release"],
)
