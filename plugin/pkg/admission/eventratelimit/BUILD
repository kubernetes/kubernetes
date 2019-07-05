package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "admission_test.go",
        "cache_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//plugin/pkg/admission/eventratelimit/apis/eventratelimit:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//vendor/github.com/hashicorp/golang-lru:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "cache.go",
        "clock.go",
        "config.go",
        "doc.go",
        "limitenforcer.go",
    ],
    importpath = "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//plugin/pkg/admission/eventratelimit/apis/eventratelimit:go_default_library",
        "//plugin/pkg/admission/eventratelimit/apis/eventratelimit/install:go_default_library",
        "//plugin/pkg/admission/eventratelimit/apis/eventratelimit/v1alpha1:go_default_library",
        "//plugin/pkg/admission/eventratelimit/apis/eventratelimit/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//vendor/github.com/hashicorp/golang-lru:go_default_library",
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
        "//plugin/pkg/admission/eventratelimit/apis/eventratelimit:all-srcs",
    ],
    tags = ["automanaged"],
)
