package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["admission.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion:go_default_library",
        "//pkg/client/listers/extensions/internalversion:go_default_library",
        "//pkg/kubeapiserver/admission:go_default_library",
        "//pkg/security/podsecuritypolicy:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//pkg/util/maps:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion:go_default_library",
        "//pkg/client/listers/extensions/internalversion:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/security/podsecuritypolicy:go_default_library",
        "//pkg/security/podsecuritypolicy/seccomp:go_default_library",
        "//pkg/security/podsecuritypolicy/util:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
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
