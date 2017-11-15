load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

filegroup(
    name = "go_default_library_protos",
    srcs = ["generated.proto"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "controller_ref.go",
        "conversion.go",
        "doc.go",
        "duration.go",
        "generated.pb.go",
        "group_version.go",
        "helpers.go",
        "labels.go",
        "meta.go",
        "micro_time.go",
        "micro_time_proto.go",
        "register.go",
        "time.go",
        "time_proto.go",
        "types.go",
        "types_swagger_doc_generated.go",
        "watch.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/selection:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/sortkeys:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "controller_ref_test.go",
        "duration_test.go",
        "group_version_test.go",
        "helpers_test.go",
        "labels_test.go",
        "micro_time_test.go",
        "time_test.go",
        "types_test.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1",
    library = ":go_default_library",
    deps = [
        "//pkg/labels:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/json-iterator/go:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["conversion_test.go"],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1_test",
    deps = [
        ":go_default_library",
        "//pkg/api/equality:go_default_library",
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
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
