package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "factory.go",
        "provider.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/security/podsecuritypolicy",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/security/podsecuritypolicy/apparmor:go_default_library",
        "//pkg/security/podsecuritypolicy/capabilities:go_default_library",
        "//pkg/security/podsecuritypolicy/group:go_default_library",
        "//pkg/security/podsecuritypolicy/seccomp:go_default_library",
        "//pkg/security/podsecuritypolicy/selinux:go_default_library",
        "//pkg/security/podsecuritypolicy/sysctl:go_default_library",
        "//pkg/security/podsecuritypolicy/user:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["provider_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/v1:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/security/podsecuritypolicy/seccomp:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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
        "//pkg/security/podsecuritypolicy/apparmor:all-srcs",
        "//pkg/security/podsecuritypolicy/capabilities:all-srcs",
        "//pkg/security/podsecuritypolicy/group:all-srcs",
        "//pkg/security/podsecuritypolicy/seccomp:all-srcs",
        "//pkg/security/podsecuritypolicy/selinux:all-srcs",
        "//pkg/security/podsecuritypolicy/sysctl:all-srcs",
        "//pkg/security/podsecuritypolicy/user:all-srcs",
        "//pkg/security/podsecuritypolicy/util:all-srcs",
    ],
    tags = ["automanaged"],
)
