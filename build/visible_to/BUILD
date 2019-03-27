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
    name = "COMMON_testing",
    packages = [
        "//hack",
        "//hack/lib",
        "//hack/make-rules",
        "//test/cmd",
        "//test/e2e/...",
        "//test/integration/...",
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
        "//cmd/kubeadm/app/cmd",
    ],
)

package_group(
    name = "cmd_kubectl_CONSUMERS",
    packages = [
        "//cmd",
    ],
)

package_group(
    name = "cmd_kubectl_app_CONSUMERS",
    packages = [
        "//cmd/kubectl",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_CONSUMERS_BAD",
    packages = [
        "//cmd/clicheck",
        "//cmd/hyperkube",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":pkg_kubectl_cmd_CONSUMERS_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl",
        "//pkg/kubectl/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_auth_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/rollout",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_config_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_create_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/edit",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_get_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_rollout_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_set_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/rollout",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_edit_testdata_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd/edit",
    ],
)

package_group(
    name = "pkg_kubectl_metricsutil_CONSUMERS",
    includes = [
        ":COMMON_generators",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/top",
    ],
)

package_group(
    name = "pkg_kubectl_testing_CONSUMERS",
    packages = [
        "//pkg/kubectl",
        "//pkg/printers/internalversion",
    ],
)

package_group(
    name = "pkg_kubectl_util_CONSUMERS",
    packages = [
        "//pkg/kubectl",
        "//pkg/kubectl/cmd/...",
        "//pkg/kubectl/generate",
        "//pkg/kubectl/generate/versioned",
        "//pkg/kubectl/proxy",
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
    visibility = ["//visibility:public"],
)
