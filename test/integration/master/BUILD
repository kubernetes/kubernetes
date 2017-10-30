package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    size = "large",
    srcs = [
        "crd_test.go",
        "kube_apiserver_test.go",
        "main_test.go",
        "synthetic_master_test.go",
    ],
    importpath = "k8s.io/kubernetes/test/integration/master",
    tags = ["integration"],
    deps = [
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/master:go_default_library",
        "//test/integration:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/networking/v1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/group:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/bearertoken:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizerfactory:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/features:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature/testing:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest:go_default_library",
        "//vendor/k8s.io/client-go/dynamic:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
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
