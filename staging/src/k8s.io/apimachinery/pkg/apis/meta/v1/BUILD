package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "duration_test.go",
        "group_version_test.go",
        "helpers_test.go",
        "labels_test.go",
        "time_test.go",
        "types_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/ugorji/go/codec:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "doc.go",
        "duration.go",
        "generated.pb.go",
        "group_version.go",
        "helpers.go",
        "labels.go",
        "meta.go",
        "register.go",
        "time.go",
        "time_proto.go",
        "types.go",
        "types_swagger_doc_generated.go",
        "watch.go",
        "well_known_annotations.go",
        "well_known_labels.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/sortkeys:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/openapi:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/selection:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
    ],
)
