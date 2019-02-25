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
sh_binary(
    name = "test-cmd",
    srcs = ["test-cmd.sh"],
    deps = [
        "//hack/lib",
        "//test/cmd:legacy-script",
    ],
)

sh_binary(
    name = "test-e2e-node",
    srcs = ["test-e2e-node.sh"],
    deps = [
        "//hack/lib",
    ],
)

sh_binary(
    name = "test-integration",
    srcs = ["test-cmd.sh"],
    deps = [
        "//hack/lib",
    ],
)

sh_binary(
    name = "build",
    srcs = ["build.sh"],
    deps = [
        "//hack/lib",
    ],
)

sh_binary(
    name = "cross",
    srcs = ["cross.sh"],
    deps = [
        "//hack/lib",
    ],
)

sh_binary(
    name = "test",
    srcs = ["test.sh"],
    deps = [
        "//hack/lib",
    ],
)

sh_binary(
    name = "vet",
    srcs = ["vet.sh"],
    deps = [
        "//hack/lib",
    ],
)

sh_binary(
    name = "verify",
    srcs = ["verify.sh"],
    deps = [
        "//hack/lib",
    ],
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
        "//hack/make-rules/helpers/go2make:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
