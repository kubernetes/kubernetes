package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "amount_test.go",
        "math_test.go",
        "quantity_example_test.go",
        "quantity_proto_test.go",
        "quantity_test.go",
        "scale_int_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/gopkg.in/inf.v0:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "amount.go",
        "generated.pb.go",
        "math.go",
        "quantity.go",
        "quantity_proto.go",
        "scale_int.go",
        "suffix.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/api/resource",
    importpath = "k8s.io/apimachinery/pkg/api/resource",
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/gopkg.in/inf.v0:go_default_library",
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
