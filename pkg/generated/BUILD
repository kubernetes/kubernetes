package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

genrule(
    name = "bindata",
    srcs = [
        "//translations:all-srcs",
    ],
    outs = ["bindata.go"],
    cmd = """
$(location //vendor/github.com/jteeuwen/go-bindata/go-bindata:go-bindata) \
  -nometadata -nocompress -o "$(OUTS)" -pkg generated \
  -prefix $$(pwd) \
  -ignore .jpg -ignore .png -ignore .md -ignore 'BUILD(\.bazel)?' \
  $(SRCS)
""",
    tools = [
        "//vendor/github.com/jteeuwen/go-bindata/go-bindata",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "bindata.go",
        "doc.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/generated",
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
        "//pkg/generated/openapi:all-srcs",
    ],
    tags = ["automanaged"],
)
