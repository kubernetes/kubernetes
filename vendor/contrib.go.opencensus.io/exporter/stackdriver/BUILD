load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "label.go",
        "sanitize.go",
        "stackdriver.go",
        "stats.go",
        "trace.go",
        "trace_proto.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/contrib.go.opencensus.io/exporter/stackdriver",
    importpath = "contrib.go.opencensus.io/exporter/stackdriver",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/cloud.google.com/go/monitoring/apiv3:go_default_library",
        "//vendor/cloud.google.com/go/trace/apiv2:go_default_library",
        "//vendor/contrib.go.opencensus.io/exporter/stackdriver/monitoredresource:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/wrappers:go_default_library",
        "//vendor/go.opencensus.io:go_default_library",
        "//vendor/go.opencensus.io/plugin/ochttp:go_default_library",
        "//vendor/go.opencensus.io/stats:go_default_library",
        "//vendor/go.opencensus.io/stats/view:go_default_library",
        "//vendor/go.opencensus.io/tag:go_default_library",
        "//vendor/go.opencensus.io/trace:go_default_library",
        "//vendor/golang.org/x/oauth2/google:go_default_library",
        "//vendor/google.golang.org/api/option:go_default_library",
        "//vendor/google.golang.org/api/support/bundler:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/distribution:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/label:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/metric:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/monitoredres:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/devtools/cloudtrace/v2:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/monitoring/v3:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/rpc/status:go_default_library",
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
        "//vendor/contrib.go.opencensus.io/exporter/stackdriver/monitoredresource:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
