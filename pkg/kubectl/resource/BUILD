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
        "builder.go",
        "doc.go",
        "helper.go",
        "interfaces.go",
        "mapper.go",
        "result.go",
        "selector.go",
        "visitor.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/text/encoding/unicode",
        "//vendor:golang.org/x/text/transform",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/rest",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "builder_test.go",
        "helper_test.go",
        "visitor_test.go",
    ],
    data = [
        "//examples:config",
        "//test/fixtures",
    ],
    library = ":go_default_library",
    tags = [
        "automanaged",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/rest/fake",
        "//vendor:k8s.io/client-go/rest/watch",
        "//vendor:k8s.io/client-go/util/testing",
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
