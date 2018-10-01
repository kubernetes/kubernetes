load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "alert.pb.go",
        "alert_service.pb.go",
        "common.pb.go",
        "dropped_labels.pb.go",
        "group.pb.go",
        "group_service.pb.go",
        "metric.pb.go",
        "metric_service.pb.go",
        "mutation_record.pb.go",
        "notification.pb.go",
        "notification_service.pb.go",
        "span_context.pb.go",
        "uptime.pb.go",
        "uptime_service.pb.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/genproto/googleapis/monitoring/v3",
    importpath = "google.golang.org/genproto/googleapis/monitoring/v3",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/duration:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/empty:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/wrappers:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/annotations:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/distribution:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/label:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/metric:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/monitoredres:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/rpc/status:go_default_library",
        "//vendor/google.golang.org/genproto/protobuf/field_mask:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
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
