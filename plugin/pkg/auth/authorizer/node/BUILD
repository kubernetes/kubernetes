package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["node_authorizer_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/auth/nodeidentifier:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "graph.go",
        "graph_populator.go",
        "node_authorizer.go",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/persistentvolume:go_default_library",
        "//pkg/api/pod:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/auth/nodeidentifier:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/core/internalversion:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//third_party/forked/gonum/graph:go_default_library",
        "//third_party/forked/gonum/graph/simple:go_default_library",
        "//third_party/forked/gonum/graph/traverse:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
