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
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/config",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
        "//staging/src/k8s.io/kubectl/pkg/cmd/set",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util/editor",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_CONSUMERS_BAD",
    packages = [
        "//cmd/clicheck",
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
        "//staging/src/k8s.io/kubectl/pkg",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_auth_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_config_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_create_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/edit",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_get_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_rollout_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_set_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
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
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/get",
        "//pkg/kubectl/util",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/annotate",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apiresources",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apply",
        "//staging/src/k8s.io/kubectl/pkg/cmd/attach",
        "//staging/src/k8s.io/kubectl/pkg/cmd/autoscale",
        "//staging/src/k8s.io/kubectl/pkg/cmd/certificates",
        "//staging/src/k8s.io/kubectl/pkg/cmd/clusterinfo",
        "//staging/src/k8s.io/kubectl/pkg/cmd/completion",
        "//staging/src/k8s.io/kubectl/pkg/cmd/config",
        "//staging/src/k8s.io/kubectl/pkg/cmd/cp",
        "//staging/src/k8s.io/kubectl/pkg/cmd/create",
        "//staging/src/k8s.io/kubectl/pkg/cmd/delete",
        "//staging/src/k8s.io/kubectl/pkg/cmd/describe",
        "//staging/src/k8s.io/kubectl/pkg/cmd/diff",
        "//staging/src/k8s.io/kubectl/pkg/cmd/drain",
        "//staging/src/k8s.io/kubectl/pkg/cmd/edit",
        "//staging/src/k8s.io/kubectl/pkg/cmd/exec",
        "//staging/src/k8s.io/kubectl/pkg/cmd/explain",
        "//staging/src/k8s.io/kubectl/pkg/cmd/expose",
        "//staging/src/k8s.io/kubectl/pkg/cmd/help",
        "//staging/src/k8s.io/kubectl/pkg/cmd/kustomize",
        "//staging/src/k8s.io/kubectl/pkg/cmd/label",
        "//staging/src/k8s.io/kubectl/pkg/cmd/logs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/options",
        "//staging/src/k8s.io/kubectl/pkg/cmd/patch",
        "//staging/src/k8s.io/kubectl/pkg/cmd/plugin",
        "//staging/src/k8s.io/kubectl/pkg/cmd/portforward",
        "//staging/src/k8s.io/kubectl/pkg/cmd/proxy",
        "//staging/src/k8s.io/kubectl/pkg/cmd/replace",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
        "//staging/src/k8s.io/kubectl/pkg/cmd/run",
        "//staging/src/k8s.io/kubectl/pkg/cmd/scale",
        "//staging/src/k8s.io/kubectl/pkg/cmd/set",
        "//staging/src/k8s.io/kubectl/pkg/cmd/taint",
        "//staging/src/k8s.io/kubectl/pkg/cmd/top",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util/sanity",
        "//staging/src/k8s.io/kubectl/pkg/cmd/version",
        "//staging/src/k8s.io/kubectl/pkg/cmd/wait",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_edit_testdata_CONSUMERS",
    packages = [
        "//staging/src/k8s.io/kubectl/pkg/cmd/edit",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_testing_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/cp",
        "//pkg/kubectl/explain",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/annotate",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apply",
        "//staging/src/k8s.io/kubectl/pkg/cmd/attach",
        "//staging/src/k8s.io/kubectl/pkg/cmd/clusterinfo",
        "//staging/src/k8s.io/kubectl/pkg/cmd/cp",
        "//staging/src/k8s.io/kubectl/pkg/cmd/create",
        "//staging/src/k8s.io/kubectl/pkg/cmd/delete",
        "//staging/src/k8s.io/kubectl/pkg/cmd/describe",
        "//staging/src/k8s.io/kubectl/pkg/cmd/drain",
        "//staging/src/k8s.io/kubectl/pkg/cmd/edit",
        "//staging/src/k8s.io/kubectl/pkg/cmd/exec",
        "//staging/src/k8s.io/kubectl/pkg/cmd/expose",
        "//staging/src/k8s.io/kubectl/pkg/cmd/get",
        "//staging/src/k8s.io/kubectl/pkg/cmd/label",
        "//staging/src/k8s.io/kubectl/pkg/cmd/logs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/patch",
        "//staging/src/k8s.io/kubectl/pkg/cmd/portforward",
        "//staging/src/k8s.io/kubectl/pkg/cmd/replace",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
        "//staging/src/k8s.io/kubectl/pkg/cmd/run",
        "//staging/src/k8s.io/kubectl/pkg/cmd/set",
        "//staging/src/k8s.io/kubectl/pkg/cmd/taint",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing",
        "//staging/src/k8s.io/kubectl/pkg/cmd/top",
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
        "//pkg/kubectl/cmd/auth",
        "//pkg/kubectl/cmd/convert",
        "//pkg/kubectl/cmd/get",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/annotate",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apiresources",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apply",
        "//staging/src/k8s.io/kubectl/pkg/cmd/attach",
        "//staging/src/k8s.io/kubectl/pkg/cmd/autoscale",
        "//staging/src/k8s.io/kubectl/pkg/cmd/certificates",
        "//staging/src/k8s.io/kubectl/pkg/cmd/clusterinfo",
        "//staging/src/k8s.io/kubectl/pkg/cmd/completion",
        "//staging/src/k8s.io/kubectl/pkg/cmd/config",
        "//staging/src/k8s.io/kubectl/pkg/cmd/cp",
        "//staging/src/k8s.io/kubectl/pkg/cmd/create",
        "//staging/src/k8s.io/kubectl/pkg/cmd/delete",
        "//staging/src/k8s.io/kubectl/pkg/cmd/describe",
        "//staging/src/k8s.io/kubectl/pkg/cmd/diff",
        "//staging/src/k8s.io/kubectl/pkg/cmd/drain",
        "//staging/src/k8s.io/kubectl/pkg/cmd/edit",
        "//staging/src/k8s.io/kubectl/pkg/cmd/exec",
        "//staging/src/k8s.io/kubectl/pkg/cmd/explain",
        "//staging/src/k8s.io/kubectl/pkg/cmd/expose",
        "//staging/src/k8s.io/kubectl/pkg/cmd/help",
        "//staging/src/k8s.io/kubectl/pkg/cmd/label",
        "//staging/src/k8s.io/kubectl/pkg/cmd/logs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/patch",
        "//staging/src/k8s.io/kubectl/pkg/cmd/plugin",
        "//staging/src/k8s.io/kubectl/pkg/cmd/portforward",
        "//staging/src/k8s.io/kubectl/pkg/cmd/proxy",
        "//staging/src/k8s.io/kubectl/pkg/cmd/replace",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
        "//staging/src/k8s.io/kubectl/pkg/cmd/run",
        "//staging/src/k8s.io/kubectl/pkg/cmd/scale",
        "//staging/src/k8s.io/kubectl/pkg/cmd/set",
        "//staging/src/k8s.io/kubectl/pkg/cmd/taint",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing",
        "//staging/src/k8s.io/kubectl/pkg/cmd/top",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util/editor",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util/sanity",
        "//staging/src/k8s.io/kubectl/pkg/cmd/version",
        "//staging/src/k8s.io/kubectl/pkg/cmd/wait",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_editor_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apply",
        "//staging/src/k8s.io/kubectl/pkg/cmd/create",
        "//staging/src/k8s.io/kubectl/pkg/cmd/edit",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_jsonmerge_CONSUMERS",
    packages = [
        "//pkg/kubectl/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
    ],
)

package_group(
    name = "pkg_kubectl_cmd_util_sanity_CONSUMERS",
    packages = [
        "//cmd/clicheck",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
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
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/top",
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
        "//pkg/kubectl/cmd/get",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/config",
        "//staging/src/k8s.io/kubectl/pkg/cmd/create",
        "//staging/src/k8s.io/kubectl/pkg/cmd/rollout",
        "//staging/src/k8s.io/kubectl/pkg/cmd/set",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util/editor",
    ],
)

package_group(
    name = "pkg_kubectl_testing_CONSUMERS",
    packages = [
        "//pkg/kubectl",
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
        "//pkg/kubectl/cmd/convert",
        "//staging/src/k8s.io/kubectl/pkg/cmd",
        "//staging/src/k8s.io/kubectl/pkg/cmd/apply",
        "//staging/src/k8s.io/kubectl/pkg/cmd/replace",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util",
    ],
)

package_group(
    name = "vendor_githubcom_prometheus_CONSUMERS",
    packages = [
        "//cluster/images/etcd-version-monitor",
        "//pkg/scheduler/framework/v1alpha1",
        "//pkg/volume/util/operationexecutor",
        "//staging/src/k8s.io/apiserver/pkg/admission/metrics",
        "//staging/src/k8s.io/component-base/metrics/...",
        "//test/e2e_node",
        "//test/integration/apiserver/flowcontrol",
        "//vendor/...",
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
