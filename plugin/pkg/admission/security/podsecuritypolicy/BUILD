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
    srcs = ["admission.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/security/podsecuritypolicy:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//pkg/util/maps:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/security/podsecuritypolicy:go_default_library",
        "//pkg/security/podsecuritypolicy/seccomp:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
