load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "walk.go",
    ],
    importpath = "k8s.io/kubernetes/test/conformance",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
    ],
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
        "//test/conformance/testdata:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "list_conformance_specs",
    srcs = [
        "//test/e2e:all-srcs",
        "//test/e2e_node:all-srcs",
        "//test/e2e:e2e.test",
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
    outs = ["specsummaries.json"],
    cmd = "$(location //vendor/github.com/onsi/ginkgo/ginkgo) --dryRun=true --focus=[Conformance] $(location //test/e2e:e2e.test) -- --spec-dump $$(pwd)/$@ > /dev/null",
    message = "Getting all conformance spec summaries.",
)

genrule(
    name = "list_conformance_tests",
    srcs = [
        ":list_conformance_specs",
        ":conformance",
        "//test/e2e:all-srcs",
    ],
    outs = ["conformance.yaml"],
    cmd = "$(location :conformance) $(location :list_conformance_specs) > $@",
    message = "Listing all conformance tests.",
)

sh_test(
    name = "conformance_test",
    srcs = ["conformance_test.sh"],
    data = [
        ":list_conformance_tests",
        ":testdata",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["walk_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
)

genrule(
    name = "gen_conformance_docs",
    srcs = [
        ":list_conformance_specs",
        ":conformance",
        "//test/e2e:all-srcs",
        ":package-srcs",
    ],
    outs = ["conformance.md"],
    cmd = "$(location :conformance) --docs $(location :list_conformance_specs) > $@",
    message = "Listing all conformance tests.",
)
