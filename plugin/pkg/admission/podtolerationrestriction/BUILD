package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion:go_default_library",
        "//pkg/kubeapiserver/admission:go_default_library",
        "//pkg/util/tolerations:go_default_library",
        "//plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "config.go",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/helper/qos:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion:go_default_library",
        "//pkg/client/listers/core/internalversion:go_default_library",
        "//pkg/kubeapiserver/admission:go_default_library",
        "//pkg/kubeapiserver/admission/util:go_default_library",
        "//pkg/util/tolerations:go_default_library",
        "//plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction:go_default_library",
        "//plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction/install:go_default_library",
        "//plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction/v1alpha1:go_default_library",
        "//plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction/validation:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
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
        "//plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction:all-srcs",
    ],
    tags = ["automanaged"],
)
