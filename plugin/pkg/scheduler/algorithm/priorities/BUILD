package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "interpod_affinity.go",
        "metadata.go",
        "node_affinity.go",
        "priorities.go",
        "selector_spreading.go",
        "taint_toleration.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//plugin/pkg/scheduler/algorithm/predicates:go_default_library",
        "//plugin/pkg/scheduler/algorithm/priorities/util:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "interpod_affinity_test.go",
        "node_affinity_test.go",
        "priorities_test.go",
        "selector_spreading_test.go",
        "taint_toleration_test.go",
    ],
    library = "go_default_library",
    tags = [
        "automanaged",
        "skip",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/util/codeinspector:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//plugin/pkg/scheduler/algorithm/priorities/util:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor:k8s.io/gengo/parser",
        "//vendor:k8s.io/gengo/types",
    ],
)
