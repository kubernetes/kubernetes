load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "controller_ref.go",
        "conversion.go",
        "deepcopy.go",
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
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/sortkeys:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/fields:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/labels:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/selection:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/intstr:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/watch:go_default_library",
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
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1/unstructured:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
