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
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authentication/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authentication/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authorization/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authorization/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/autoscaling/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/autoscaling/v2alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/batch/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/batch/v2alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/certificates/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/policy/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/rbac/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/settings/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/storage/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/storage/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/apps/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authentication/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authorization/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/autoscaling/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/batch/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/certificates/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/extensions/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/policy/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/rbac/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/settings/install:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/storage/install:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
    ],
)
