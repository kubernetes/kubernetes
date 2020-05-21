load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "intializers.go",
        "restrictusers.go",
        "subjectchecker.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/authorization/restrictusers",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/authorization/restrictusers/usercache:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/api/user/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/authorization/clientset/versioned/typed/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/user/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/user/clientset/versioned/typed/user/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/user/informers/externalversions:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "groupcache_test.go",
        "restrictusers_test.go",
        "subjectchecker_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/api/user/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/authorization/clientset/versioned/fake:go_default_library",
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
        "//openshift-kube-apiserver/admission/authorization/restrictusers/usercache:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
