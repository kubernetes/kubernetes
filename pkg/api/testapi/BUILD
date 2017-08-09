package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["testapi.go"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/install:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/install:go_default_library",
        "//pkg/apis/admission:go_default_library",
        "//pkg/apis/admission/install:go_default_library",
        "//pkg/apis/admissionregistration:go_default_library",
        "//pkg/apis/admissionregistration/install:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/apps/install:go_default_library",
        "//pkg/apis/authentication/install:go_default_library",
        "//pkg/apis/authorization:go_default_library",
        "//pkg/apis/authorization/install:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/autoscaling/install:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/install:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/certificates/install:go_default_library",
        "//pkg/apis/componentconfig/install:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//pkg/apis/imagepolicy:go_default_library",
        "//pkg/apis/imagepolicy/install:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/apis/networking/install:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/policy/install:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/install:go_default_library",
        "//pkg/apis/scheduling:go_default_library",
        "//pkg/apis/scheduling/install:go_default_library",
        "//pkg/apis/settings:go_default_library",
        "//pkg/apis/settings/install:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//pkg/apis/storage/install:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/recognizer:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["testapi_test.go"],
    library = ":go_default_library",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
