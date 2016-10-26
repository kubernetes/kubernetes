package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "zz_generated.listers.apps.go",
        "zz_generated.listers.authentication.go",
        "zz_generated.listers.authorization.go",
        "zz_generated.listers.autoscaling.go",
        "zz_generated.listers.batch.go",
        "zz_generated.listers.certificates.go",
        "zz_generated.listers.core.go",
        "zz_generated.listers.extensions.go",
        "zz_generated.listers.imagepolicy.go",
        "zz_generated.listers.policy.go",
        "zz_generated.listers.rbac.go",
        "zz_generated.listers.storage.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apis/authorization:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/imagepolicy:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
