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
        "//cmd/genslateyaml",
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
    name = "pkg_kubectl_CONSUMERS_BAD",
    includes = [
        ":FEDERATION_BAD",
        ":KUBEADM_BAD",
    ],
    packages = [
        "//cmd/clicheck",
        "//cmd/hyperkube",
        "//pkg",
    ],
)

package_group(
    name = "pkg_kubectl_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":pkg_kubectl_CONSUMERS_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/cmd/util/editor",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_CONSUMERS_BAD",
    includes = [
        ":FEDERATION_BAD",
    ],
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
    name = "pkg_kubectl_cmd_templates_CONSUMERS_BAD",
    packages = [
        "//federation/pkg/kubefed/init",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_templates_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":COMMON_testing",
        ":FEDERATION_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/templates",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/cmd/util/sanity",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_testdata_edit_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_testing_CONSUMERS_BAD",
    packages = [
        "//federation/pkg/kubefed",
        "//federation/pkg/kubefed/init",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_testing_CONSUMERS",
    includes = [
        ":pkg_kubectl_cmd_testing_CONSUMERS_BAD",
    ],
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/set",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_CONSUMERS_BAD",
    includes = [
        ":FEDERATION_BAD",
        ":KUBEADM_BAD",
    ],
    packages = [
        "//cmd/clicheck",
        "//cmd/hyperkube",
        "//cmd/kube-proxy/app",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":COMMON_testing",
        ":pkg_kubectl_cmd_util_CONSUMERS_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/cmd/util/editor",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_editor_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/util",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_jsonmerge_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/util",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_sanity_CONSUMERS",
    packages = [
        "//cmd/clicheck",
        "//pkg/kubectl/cmd/util",
    ],
)

package_group(
    name = "pkg_kubectl_metricsutil_CONSUMERS_BAD",
    includes = [
        ":FEDERATION_BAD",
    ],
    packages = [
        "//cmd/clicheck",
        "//cmd/hyperkube",
    ],
)

package_group(
    name = "pkg_kubectl_metricsutil_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":pkg_kubectl_metricsutil_CONSUMERS_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl",
        "//pkg/kubectl/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_resource_CONSUMERS_BAD",
    packages = [
        "//federation/pkg/kubefed",
    ],
)

package_group(
    name = "pkg_kubectl_resource_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":COMMON_testing",
        ":pkg_kubectl_resource_CONSUMERS_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/cmd/util/editor",
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
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/proxy",
    ],
)

package_group(
    name = "pkg_kubectl_validation_CONSUMERS",
    packages = [
        "//pkg/kubectl",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/resource",
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
