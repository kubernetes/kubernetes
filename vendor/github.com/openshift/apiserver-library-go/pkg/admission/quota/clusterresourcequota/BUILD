load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "accessor.go",
        "admission.go",
        "intializers.go",
        "lockfactory.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/admission/quota/clusterresourcequota",
    importpath = "github.com/openshift/apiserver-library-go/pkg/admission/quota/clusterresourcequota",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/quota/v1:go_default_library",
        "//pkg/quota/v1/install:go_default_library",
        "//plugin/pkg/admission/resourcequota:go_default_library",
        "//plugin/pkg/admission/resourcequota/apis/resourcequota:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/hashicorp/golang-lru:go_default_library",
        "//vendor/github.com/openshift/api/quota/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/quota/clientset/versioned/typed/quota/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/quota/informers/externalversions/quota/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/quota/listers/quota/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/quota/clusterquotamapping:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/quota/quotautil:go_default_library",
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
    visibility = ["//visibility:public"],
)
