package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["testapi.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/install:go_default_library",
        "//pkg/apis/componentconfig/install:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/recognizer",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/install",
        "//vendor:k8s.io/apis/pkg/apis/apps",
        "//vendor:k8s.io/apis/pkg/apis/apps/install",
        "//vendor:k8s.io/apis/pkg/apis/authentication/install",
        "//vendor:k8s.io/apis/pkg/apis/authorization",
        "//vendor:k8s.io/apis/pkg/apis/authorization/install",
        "//vendor:k8s.io/apis/pkg/apis/autoscaling",
        "//vendor:k8s.io/apis/pkg/apis/autoscaling/install",
        "//vendor:k8s.io/apis/pkg/apis/batch",
        "//vendor:k8s.io/apis/pkg/apis/batch/install",
        "//vendor:k8s.io/apis/pkg/apis/certificates",
        "//vendor:k8s.io/apis/pkg/apis/certificates/install",
        "//vendor:k8s.io/apis/pkg/apis/extensions",
        "//vendor:k8s.io/apis/pkg/apis/extensions/install",
        "//vendor:k8s.io/apis/pkg/apis/imagepolicy",
        "//vendor:k8s.io/apis/pkg/apis/imagepolicy/install",
        "//vendor:k8s.io/apis/pkg/apis/policy",
        "//vendor:k8s.io/apis/pkg/apis/policy/install",
        "//vendor:k8s.io/apis/pkg/apis/rbac",
        "//vendor:k8s.io/apis/pkg/apis/rbac/install",
        "//vendor:k8s.io/apis/pkg/apis/storage",
        "//vendor:k8s.io/apis/pkg/apis/storage/install",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["testapi_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
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
