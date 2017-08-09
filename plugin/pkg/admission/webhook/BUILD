package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "admission_test.go",
        "certs_test.go",
        "rules_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/admission/install:go_default_library",
        "//vendor/k8s.io/api/admission/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "doc.go",
        "rules.go",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/admission/install:go_default_library",
        "//pkg/apis/admission/v1alpha1:go_default_library",
        "//pkg/kubeapiserver/admission:go_default_library",
        "//pkg/kubeapiserver/admission/configuration:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/admission/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
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
