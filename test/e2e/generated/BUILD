package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "gobindata_util.go",
        "main.go",
        ":bindata",
    ],
    deps = [
        "//vendor:github.com/golang/glog",
    ],
)

genrule(
    name = "bindata",
    srcs = [
        "//examples:sources",
        "//test/images:sources",
        "//test/fixtures:sources",
        "//test/e2e/testing-manifests:sources",
    ],
    outs = ["bindata.go"],
    cmd = """
$(location //vendor:github.com/jteeuwen/go-bindata/go-bindata_bin) \
  -nometadata -o "$(OUTS)" -pkg generated \
  -prefix $$(pwd) \
  -ignore .jpg -ignore .png -ignore .md \
  $(SRCS)
""",
    tools = [
        "//vendor:github.com/jteeuwen/go-bindata/go-bindata_bin",
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
