package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "endpoints.go",
        "service.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/proxy",
    deps = [
        "//pkg/api/v1/service:go_default_library",
        "//pkg/proxy/config:go_default_library",
        "//pkg/proxy/metrics:go_default_library",
        "//pkg/proxy/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
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
        "//pkg/proxy/apis:all-srcs",
        "//pkg/proxy/config:all-srcs",
        "//pkg/proxy/healthcheck:all-srcs",
        "//pkg/proxy/iptables:all-srcs",
        "//pkg/proxy/ipvs:all-srcs",
        "//pkg/proxy/metrics:all-srcs",
        "//pkg/proxy/userspace:all-srcs",
        "//pkg/proxy/util:all-srcs",
        "//pkg/proxy/winkernel:all-srcs",
        "//pkg/proxy/winuserspace:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "endpoints_test.go",
        "service_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
    ],
)
