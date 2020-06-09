package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "deepcopy.go",
        "defaults.go",
        "doc.go",
        "generated.pb.go",
        "marshal.go",
        "register.go",
        "types.go",
        "types_jsonschema.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
        "zz_generated.prerelease-lifecycle.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1",
    importpath = "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1",
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/sortkeys:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "conversion_test.go",
        "defaults_test.go",
        "marshal_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)
