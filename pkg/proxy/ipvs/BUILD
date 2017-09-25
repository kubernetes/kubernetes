package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "proxier_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/api:go_default_library",
            "//pkg/proxy:go_default_library",
            "//pkg/proxy/util:go_default_library",
            "//pkg/util/iptables:go_default_library",
            "//pkg/util/iptables/testing:go_default_library",
            "//pkg/util/ipvs:go_default_library",
            "//pkg/util/ipvs/testing:go_default_library",
            "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
            "//vendor/k8s.io/utils/exec/testing:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_library(
    name = "go_default_library",
    srcs = ["proxier.go"],
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
        "//pkg/util/ipvs:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
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
