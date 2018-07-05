# Scripts runnable from make, e.g.
#
#    cd $GOPATH/src/k8s.io/kubernetes
#    make test-e2e-node
#
# The sh_binary rules below exist only to validate
# dependencies; if a shell dependency is accidentally
# deleted, a presubmit BUILD will fail.
#
# If the scripts sourced their dependencies from
# $RUNFILES (rather than $BASH_SOURCE/../.. or
# whatever), then bazel build hack/...  would install
# runnable, hermetically sealed shell "binaries".
# E.g. the following command would work:
#
#   ./bazel-bin/hack/make-rules/test-e2e-node
#
# TODO(#47064): Should be a sh_test instead of sh_binary

sh_library(
    name = "legacy-script",
    srcs = [
        "apply.sh",
        "apps.sh",
        "authorization.sh",
        "batch.sh",
        "certificate.sh",
        "core.sh",
        "crd.sh",
        "create.sh",
        "diff.sh",
        "discovery.sh",
        "generic-resources.sh",
        "get.sh",
        "initializers.sh",
        "kubeconfig.sh",
        "legacy-script.sh",
        "node-management.sh",
        "old-print.sh",
        "plugins.sh",
        "proxy.sh",
        "rbac.sh",
        "request-timeout.sh",
        "run.sh",
        "save-config.sh",
        "storage.sh",
        "template-output.sh",
        "version.sh",
    ],
    data = ["//pkg/kubectl/validation:testdata/v1/validPod.yaml"],
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
