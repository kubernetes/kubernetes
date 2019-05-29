package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "graph_test.go",
        "intset_test.go",
        "node_authorizer_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/auth/nodeidentifier:go_default_library",
        "//pkg/features:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "graph.go",
        "graph_populator.go",
        "intset.go",
        "node_authorizer.go",
    ],
    importpath = "k8s.io/kubernetes/plugin/pkg/auth/authorizer/node",
    deps = [
        "//pkg/api/v1/persistentvolume:go_default_library",
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/apis/coordination:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//pkg/auth/nodeidentifier:go_default_library",
        "//pkg/features:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/informers/storage/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate:go_default_library",
        "//third_party/forked/gonum/graph:go_default_library",
        "//third_party/forked/gonum/graph/simple:go_default_library",
        "//third_party/forked/gonum/graph/traverse:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
