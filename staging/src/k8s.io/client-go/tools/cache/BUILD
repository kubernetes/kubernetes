package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "controller_test.go",
        "delta_fifo_test.go",
        "expiration_cache_test.go",
        "fifo_test.go",
        "heap_test.go",
        "index_test.go",
        "mutation_detector_test.go",
        "processor_listener_test.go",
        "reflector_test.go",
        "shared_informer_test.go",
        "store_test.go",
        "undelta_store_test.go",
    ],
    features = ["-race"],
    importpath = "k8s.io/client-go/tools/cache",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache/testing:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "controller.go",
        "delta_fifo.go",
        "doc.go",
        "expiration_cache.go",
        "expiration_cache_fakes.go",
        "fake_custom_store.go",
        "fifo.go",
        "heap.go",
        "index.go",
        "listers.go",
        "listwatch.go",
        "mutation_cache.go",
        "mutation_detector.go",
        "reflector.go",
        "reflector_metrics.go",
        "shared_informer.go",
        "store.go",
        "thread_safe_store.go",
        "undelta_store.go",
    ],
    importpath = "k8s.io/client-go/tools/cache",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/cache:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/pager:go_default_library",
        "//vendor/k8s.io/client-go/util/buffer:go_default_library",
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
        "//staging/src/k8s.io/client-go/tools/cache/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
