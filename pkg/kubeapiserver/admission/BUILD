package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["initializer_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/apiserver/pkg/admission:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["initializer.go"],
    importpath = "k8s.io/kubernetes/pkg/kubeapiserver/admission",
    deps = [
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion:go_default_library",
        "//pkg/quota/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/webhook:go_default_library",
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
        "//pkg/kubeapiserver/admission/util:all-srcs",
    ],
    tags = ["automanaged"],
)
