package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
        "import_known_versions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/componentconfig/install:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/internalversion:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/install",
        "//vendor:k8s.io/apis/pkg/apis/apps/install",
        "//vendor:k8s.io/apis/pkg/apis/authentication/install",
        "//vendor:k8s.io/apis/pkg/apis/authorization/install",
        "//vendor:k8s.io/apis/pkg/apis/autoscaling/install",
        "//vendor:k8s.io/apis/pkg/apis/batch/install",
        "//vendor:k8s.io/apis/pkg/apis/certificates/install",
        "//vendor:k8s.io/apis/pkg/apis/extensions/install",
        "//vendor:k8s.io/apis/pkg/apis/policy/install",
        "//vendor:k8s.io/apis/pkg/apis/rbac/install",
        "//vendor:k8s.io/apis/pkg/apis/storage/install",
        "//vendor:k8s.io/client-go/discovery",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/flowcontrol",
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
        "//pkg/client/clientset_generated/internalclientset/fake:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:all-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/internalversion:all-srcs",
    ],
    tags = ["automanaged"],
)
