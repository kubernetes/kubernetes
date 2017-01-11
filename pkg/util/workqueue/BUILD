package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "default_rate_limiters.go",
        "delaying_queue.go",
        "doc.go",
        "metrics.go",
        "parallelizer.go",
        "queue.go",
        "rate_limitting_queue.go",
        "timed_queue.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/clock:go_default_library",
        "//vendor:github.com/juju/ratelimit",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "default_rate_limiters_test.go",
        "delaying_queue_test.go",
        "rate_limitting_queue_test.go",
        "timed_queue_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["queue_test.go"],
    tags = ["automanaged"],
    deps = ["//pkg/util/workqueue:go_default_library"],
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
        "//pkg/util/workqueue/prometheus:all-srcs",
    ],
    tags = ["automanaged"],
)
