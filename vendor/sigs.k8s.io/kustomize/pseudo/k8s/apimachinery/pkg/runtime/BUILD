load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "codec.go",
        "codec_check.go",
        "conversion.go",
        "converter.go",
        "doc.go",
        "embedded.go",
        "error.go",
        "extension.go",
        "generated.pb.go",
        "helper.go",
        "interfaces.go",
        "mapper.go",
        "register.go",
        "scheme.go",
        "scheme_builder.go",
        "swagger_doc_generator.go",
        "types.go",
        "types_proto.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion/queryparams:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/json:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/naming:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/sets:go_default_library",
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
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/serializer:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
