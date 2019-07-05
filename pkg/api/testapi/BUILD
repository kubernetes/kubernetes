package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["testapi.go"],
    importpath = "k8s.io/kubernetes/pkg/api/testapi",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/admission:go_default_library",
        "//pkg/apis/admission/install:go_default_library",
        "//pkg/apis/admissionregistration:go_default_library",
        "//pkg/apis/admissionregistration/install:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/apps/install:go_default_library",
        "//pkg/apis/auditregistration:go_default_library",
        "//pkg/apis/auditregistration/install:go_default_library",
        "//pkg/apis/authentication/install:go_default_library",
        "//pkg/apis/authorization:go_default_library",
        "//pkg/apis/authorization/install:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/autoscaling/install:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/install:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/certificates/install:go_default_library",
        "//pkg/apis/coordination:go_default_library",
        "//pkg/apis/coordination/install:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/install:go_default_library",
        "//pkg/apis/events:go_default_library",
        "//pkg/apis/events/install:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//pkg/apis/imagepolicy:go_default_library",
        "//pkg/apis/imagepolicy/install:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/apis/networking/install:go_default_library",
        "//pkg/apis/node:go_default_library",
        "//pkg/apis/node/install:go_default_library",
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
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/recognizer:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["testapi_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
