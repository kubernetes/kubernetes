package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "metrics.go",
        "proxier.go",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/helper:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/proxy:go_default_library",
        "//pkg/proxy/healthcheck:go_default_library",
        "//pkg/proxy/util:go_default_library",
        "//pkg/util/async:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["proxier_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/proxy:go_default_library",
        "//pkg/util/async:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/iptables/testing:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
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
