package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "configuration_manager_test.go",
        "initializer_manager_test.go",
        "mutating_webhook_manager_test.go",
        "validating_webhook_manager_test.go",
    ],
    importpath = "k8s.io/apiserver/pkg/admission/configuration",
    library = ":go_default_library",
    deps = [
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "configuration_manager.go",
        "initializer_manager.go",
        "mutating_webhook_manager.go",
        "validating_webhook_manager.go",
    ],
    importpath = "k8s.io/apiserver/pkg/admission/configuration",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
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
