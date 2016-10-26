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
        "fakes.go",
        "identity_mappers.go",
        "iterator.go",
        "pet.go",
        "pet_set.go",
        "pet_set_utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/pod:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:gopkg.in/inf.v0",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "identity_mappers_test.go",
        "iterator_test.go",
        "pet_set_test.go",
        "pet_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/pod:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion/fake:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/testing:go_default_library",
    ],
)
