load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "authority.go",
        "policies.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/certificates/authority",
    visibility = [
        # This cmd/kubelet dependency is used for testing. We should migrate
        # that test to an integration test that uses the certificates
        # controller, and remove the dependency.
        "//cmd/kubelet/app:__pkg__",
        "//pkg/controller/certificates:__subpackages__",
    ],
    deps = ["//staging/src/k8s.io/api/certificates/v1beta1:go_default_library"],
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

go_test(
    name = "go_default_test",
    srcs = [
        "authority_test.go",
        "policies_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp/cmpopts:go_default_library",
    ],
)
