load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["walk.go"],
    importpath = "k8s.io/kubernetes/test/conformance",
    visibility = ["//visibility:private"],
)

go_binary(
    name = "conformance",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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
        "//test/conformance/kubetestgen:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "list_conformance_tests",
    srcs = [
        "//test/e2e:all-srcs",
        "//test/e2e_node:all-srcs",
    ],
    outs = ["conformance.txt"],
    cmd = "./$(location :conformance) $(locations //test/e2e:all-srcs) > $@",
    message = "Listing all conformance tests.",
    tools = [":conformance"],
)

sh_test(
    name = "conformance_test",
    srcs = ["conformance_test.sh"],
    data = [
        "testdata/conformance.txt",
        ":list_conformance_tests",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["walk_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
)
