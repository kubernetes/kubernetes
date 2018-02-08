load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "gce_projects.go",
        "gen.go",
        "op.go",
        "project.go",
        "ratelimit.go",
        "service.go",
        "utils.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/cloudprovider/providers/gce/cloud/filter:go_default_library",
        "//pkg/cloudprovider/providers/gce/cloud/meta:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.alpha:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.beta:go_default_library",
        "//vendor/google.golang.org/api/compute/v1:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "gen_test.go",
        "mock_test.go",
        "utils_test.go",
    ],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud",
    deps = [
        "//pkg/cloudprovider/providers/gce/cloud/filter:go_default_library",
        "//pkg/cloudprovider/providers/gce/cloud/meta:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.alpha:go_default_library",
        "//vendor/google.golang.org/api/compute/v0.beta:go_default_library",
        "//vendor/google.golang.org/api/compute/v1:go_default_library",
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
        "//pkg/cloudprovider/providers/gce/cloud/filter:all-srcs",
        "//pkg/cloudprovider/providers/gce/cloud/gen:all-srcs",
        "//pkg/cloudprovider/providers/gce/cloud/meta:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
