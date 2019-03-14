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
    name = "pkg_kubectl_CONSUMERS",
    includes = [
        ":COMMON_generators",
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
    name = "pkg_kubectl_util_templates_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":COMMON_testing",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/annotate",
        "//pkg/kubectl/cmd/apiresources",
        "//pkg/kubectl/cmd/apply",
        "//pkg/kubectl/cmd/attach",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/autoscale",
        "//pkg/kubectl/cmd/certificates",
        "//pkg/kubectl/cmd/clusterinfo",
        "//pkg/kubectl/cmd/completion",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/cp",
        "//pkg/kubectl/cmd/create",
        "//pkg/kubectl/cmd/delete",
        "//pkg/kubectl/cmd/describe",
        "//pkg/kubectl/cmd/diff",
        "//pkg/kubectl/cmd/drain",
        "//pkg/kubectl/cmd/edit",
        "//pkg/kubectl/cmd/exec",
        "//pkg/kubectl/cmd/explain",
        "//pkg/kubectl/cmd/expose",
        "//pkg/kubectl/cmd/get",
        "//pkg/kubectl/cmd/help",
        "//pkg/kubectl/cmd/kustomize",
        "//pkg/kubectl/cmd/label",
        "//pkg/kubectl/cmd/logs",
        "//pkg/kubectl/cmd/options",
        "//pkg/kubectl/cmd/patch",
        "//pkg/kubectl/cmd/plugin",
        "//pkg/kubectl/cmd/portforward",
        "//pkg/kubectl/cmd/proxy",
        "//pkg/kubectl/cmd/replace",
        "//pkg/kubectl/cmd/rollingupdate",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/run",
        "//pkg/kubectl/cmd/scale",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/taint",
        "//pkg/kubectl/cmd/top",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/cmd/util/sanity",
        "//pkg/kubectl/cmd/version",
        "//pkg/kubectl/cmd/wait",
        "//pkg/kubectl/util",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_edit_testdata_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd/edit",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_testing_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/annotate",
        "//pkg/kubectl/cmd/apply",
        "//pkg/kubectl/cmd/attach",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/clusterinfo",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/cp",
        "//pkg/kubectl/cmd/create",
        "//pkg/kubectl/cmd/delete",
        "//pkg/kubectl/cmd/describe",
        "//pkg/kubectl/cmd/drain",
        "//pkg/kubectl/cmd/edit",
        "//pkg/kubectl/cmd/exec",
        "//pkg/kubectl/cmd/expose",
        "//pkg/kubectl/cmd/get",
        "//pkg/kubectl/cmd/label",
        "//pkg/kubectl/cmd/logs",
        "//pkg/kubectl/cmd/patch",
        "//pkg/kubectl/cmd/portforward",
        "//pkg/kubectl/cmd/replace",
        "//pkg/kubectl/cmd/rollingupdate",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/run",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/taint",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/top",
        "//pkg/kubectl/explain",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":COMMON_testing",
        ":KUBEADM_BAD",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/annotate",
        "//pkg/kubectl/cmd/apiresources",
        "//pkg/kubectl/cmd/apply",
        "//pkg/kubectl/cmd/attach",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/autoscale",
        "//pkg/kubectl/cmd/certificates",
        "//pkg/kubectl/cmd/clusterinfo",
        "//pkg/kubectl/cmd/completion",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/cp",
        "//pkg/kubectl/cmd/create",
        "//pkg/kubectl/cmd/delete",
        "//pkg/kubectl/cmd/describe",
        "//pkg/kubectl/cmd/diff",
        "//pkg/kubectl/cmd/drain",
        "//pkg/kubectl/cmd/edit",
        "//pkg/kubectl/cmd/exec",
        "//pkg/kubectl/cmd/explain",
        "//pkg/kubectl/cmd/expose",
        "//pkg/kubectl/cmd/get",
        "//pkg/kubectl/cmd/help",
        "//pkg/kubectl/cmd/label",
        "//pkg/kubectl/cmd/logs",
        "//pkg/kubectl/cmd/patch",
        "//pkg/kubectl/cmd/plugin",
        "//pkg/kubectl/cmd/portforward",
        "//pkg/kubectl/cmd/proxy",
        "//pkg/kubectl/cmd/replace",
        "//pkg/kubectl/cmd/rollingupdate",
        "//pkg/kubectl/cmd/rollout",
        "//pkg/kubectl/cmd/run",
        "//pkg/kubectl/cmd/scale",
        "//pkg/kubectl/cmd/set",
        "//pkg/kubectl/cmd/taint",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/top",
        "//pkg/kubectl/cmd/util",
        "//pkg/kubectl/cmd/util/editor",
        "//pkg/kubectl/cmd/util/sanity",
        "//pkg/kubectl/cmd/version",
        "//pkg/kubectl/cmd/wait",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_editor_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/apply",
        "//pkg/kubectl/cmd/create",
        "//pkg/kubectl/cmd/edit",
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
    name = "pkg_kubectl_resource_CONSUMERS",
    includes = [
        ":COMMON_generators",
        ":COMMON_testing",
    ],
    packages = [
        "//cmd/kubectl",
        "//cmd/kubectl/app",
        "//pkg/kubectl",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/config",
        "//pkg/kubectl/cmd/create",
        "//pkg/kubectl/cmd/get",
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
        "//pkg/kubectl/cmd/...",
        "//pkg/kubectl/generate",
        "//pkg/kubectl/generate/versioned",
        "//pkg/kubectl/proxy",
    ],
)

package_group(
    name = "pkg_kubectl_validation_CONSUMERS",
    packages = [
        "//pkg/kubectl",
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/apply",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/replace",
        "//pkg/kubectl/cmd/rollingupdate",
        "//pkg/kubectl/cmd/testing",
        "//pkg/kubectl/cmd/util",
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
