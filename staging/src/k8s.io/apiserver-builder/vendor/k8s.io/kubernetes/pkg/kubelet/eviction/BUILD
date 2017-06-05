package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "cgo_genrule",
    "go_library",
    "go_test",
)

cgo_genrule(
    name = "cgo_codegen",
    srcs = ["threshold_notifier_linux.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "eviction_manager_test.go",
        "helpers_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/api/v1alpha1/stats:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/quota:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apiserver/pkg/util/feature",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/record",
        "//vendor:k8s.io/client-go/util/clock",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "eviction_manager.go",
        "helpers.go",
        "types.go",
    ],
    library = ":cgo_codegen",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/api/v1alpha1/stats:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/quota/evaluator/core:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/util/feature",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/record",
        "//vendor:k8s.io/client-go/util/clock",
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
        "//pkg/kubelet/eviction/api:all-srcs",
    ],
    tags = ["automanaged"],
)
