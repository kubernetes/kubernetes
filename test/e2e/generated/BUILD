package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "bindata.go",
        "gobindata_util.go",
        "main.go",
    ],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

genrule(
    name = "bindata",
    srcs = [
        "//examples:all-srcs",
        "//test/images:all-srcs",
        "//test/fixtures:all-srcs",
        "//test/e2e/testing-manifests:all-srcs",
    ],
    outs = ["bindata.go"],
    cmd = """
$(location //vendor/github.com/jteeuwen/go-bindata/go-bindata:go-bindata) \
  -nometadata -o "$(OUTS)" -pkg generated \
  -prefix $$(pwd) \
  -ignore .jpg -ignore .png -ignore .md \
  $(SRCS)
""",
    tools = [
        "//vendor/github.com/jteeuwen/go-bindata/go-bindata",
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
