package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "configmaps.go",
        "deployments.go",
        "horizontal_pod_autoscalers.go",
        "secrets.go",
        "services.go",
        "statefulset.go",
        "upgrade.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/controller/deployment/util:go_default_library",
        "//test/e2e/common:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/gomega",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/uuid",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
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
