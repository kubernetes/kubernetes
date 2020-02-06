load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "decode_metric.go",
        "error.go",
        "find_stable_metric.go",
        "main.go",
        "metric.go",
    ],
    importpath = "k8s.io/kubernetes/test/instrumentation",
    visibility = ["//visibility:private"],
    deps = [
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
    ],
)

go_binary(
    name = "instrumentation",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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

genrule(
    name = "list_stable_metrics",
    srcs = [
        "//:all-srcs",
    ],
    outs = ["stable-metrics-list.yaml"],
    cmd = "for loc in $(locations //:all-srcs); do echo $$loc; done | ./$(locations :instrumentation) - > $@",
    message = "Listing all stable metrics.",
    tools = [":instrumentation"],
)

sh_test(
    name = "verify_stable_metric",
    srcs = ["verify-stable-metrics.sh"],
    data = [
        "testdata/stable-metrics-list.yaml",
        ":list_stable_metrics",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["main_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/component-base/metrics:go_default_library"],
)
