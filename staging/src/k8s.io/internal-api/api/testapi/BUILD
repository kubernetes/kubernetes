package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["testapi.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/internal-api/api/testapi",
    importpath = "k8s.io/internal-api/api/testapi",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/recognizer:go_default_library",
        "//staging/src/k8s.io/internal-api/api/legacyscheme:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/admission:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/admission/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/admissionregistration:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/admissionregistration/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/apps:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/apps/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/auditregistration:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/auditregistration/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/authentication/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/authorization:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/authorization/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/autoscaling:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/autoscaling/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/batch:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/batch/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/certificates:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/certificates/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/coordination:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/coordination/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/discovery:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/discovery/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/events:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/events/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/extensions:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/extensions/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/imagepolicy:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/imagepolicy/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/networking:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/networking/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/node:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/node/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/policy:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/policy/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/rbac:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/rbac/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/scheduling:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/scheduling/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/settings:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/settings/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/storage:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/storage/install:go_default_library",
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
