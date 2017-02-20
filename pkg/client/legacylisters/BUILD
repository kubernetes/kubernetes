package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["listers_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/client-go/tools/cache",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "listers.go",
        "listers_core.go",
        "listers_extensions.go",
        "listers_rbac.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/apis/apps/v1beta1",
        "//vendor:k8s.io/apis/pkg/apis/certificates/v1beta1",
        "//vendor:k8s.io/apis/pkg/apis/extensions",
        "//vendor:k8s.io/apis/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/apis/pkg/apis/policy/v1beta1",
        "//vendor:k8s.io/apis/pkg/apis/rbac",
        "//vendor:k8s.io/apis/pkg/apis/storage",
        "//vendor:k8s.io/apis/pkg/apis/storage/v1beta1",
        "//vendor:k8s.io/client-go/tools/cache",
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
