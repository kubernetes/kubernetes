load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrapauthenticator.go",
        "expirationvalidator.go",
        "interfaces.go",
        "timeoutvalidator.go",
        "tokenauthenticator.go",
        "uidvalidator.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/authentication/oauth",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/authentication/oauth/rankedset:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/api/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/api/user/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/listers/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/user/clientset/versioned/typed/user/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/authentication/bootstrapauthenticator:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "expirationvalidator_test.go",
        "tokenauthenticator_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//vendor/github.com/openshift/api/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/api/user/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned/fake:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/user/clientset/versioned/fake:go_default_library",
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
        "//openshift-kube-apiserver/authentication/oauth/rankedset:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
