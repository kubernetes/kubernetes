load("//build:deepcopy.bzl", "k8s_deepcopy", "k8s_conversion")

package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "generated.pb.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/metrics/pkg/apis/external_metrics/v1beta1",
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/sortkeys:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/metrics/pkg/apis/external_metrics:go_default_library",
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

filegroup(
    name = "go_default_library_protos",
    srcs = ["generated.proto"],
)

k8s_deepcopy(outs = ["zz_generated.deepcopy.go"])

k8s_conversion(outs = ["zz_generated.conversion.go"])
