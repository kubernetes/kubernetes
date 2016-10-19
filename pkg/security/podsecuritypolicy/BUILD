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
        "doc.go",
        "factory.go",
        "provider.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/security/podsecuritypolicy/apparmor:go_default_library",
        "//pkg/security/podsecuritypolicy/capabilities:go_default_library",
        "//pkg/security/podsecuritypolicy/group:go_default_library",
        "//pkg/security/podsecuritypolicy/seccomp:go_default_library",
        "//pkg/security/podsecuritypolicy/selinux:go_default_library",
        "//pkg/security/podsecuritypolicy/sysctl:go_default_library",
        "//pkg/security/podsecuritypolicy/user:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/maps:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["provider_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/security/podsecuritypolicy/seccomp:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//vendor:github.com/davecgh/go-spew/spew",
    ],
)
