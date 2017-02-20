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
        "//pkg/client/clientset_generated/clientset/typed/apps/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authentication/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authentication/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/batch/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/batch/v2alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/policy/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/storage/v1beta1:go_default_library",
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
        "//pkg/client/clientset_generated/clientset/fake:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/apps/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/authentication/v1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/authentication/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/batch/v1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/batch/v2alpha1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/extensions/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/policy/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1beta1:all-srcs",
        "//pkg/client/clientset_generated/clientset/typed/storage/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
