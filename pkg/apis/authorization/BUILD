package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

genrule(
    name = "generate-deepcopy",
    srcs = glob(["**/*.go"], exclude=["zz_generated.deepcopy.go"]) + [
        "//vendor/k8s.io/code-generator/hack:boilerplate.go.txt",
        "//:all-srcs",
    ],
    outs = ["zz_generated.deepcopy.go"],
    tools = [
        "//vendor/k8s.io/code-generator/cmd/deepcopy-gen",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
    ],
    cmd = """
      export O=$$PWD
      export GOPATH=$$PWD/.go
      export GOROOT=/usr/lib/google-golang
      dcg=$$PWD/'$(location //vendor/k8s.io/code-generator/cmd/deepcopy-gen)'
      mkdir -p $$GOPATH/src/k8s.io
      touch $$GOPATH/BUILD.bazel
      ln -snf $$PWD $$GOPATH/src/k8s.io/kubernetes
      for i in $$(ls staging/src/k8s.io); do
        ln -snf $$PWD/staging/src/k8s.io/$$i $$GOPATH/src/k8s.io/$$i
      done
      cd $$GOPATH/src/k8s.io/kubernetes
      $$dcg \
      -v 1 \
      -i k8s.io/kubernetes/pkg/apis/authorization \
      --bounding-dirs k8s.io/kubernetes,k8s.io/api \
      -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
      -O zz_generated.deepcopy
      ls $$GOPATH/src/k8s.io/kubernetes/pkg/apis/authorization/zz_generated.deepcopy.go
      ln pkg/apis/authorization/zz_generated.deepcopy.go $$O/'$(location zz_generated.deepcopy.go)'
    """,
    message = "generating",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/apis/authorization",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"], exclude=["**/zz_generated*.go"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/apis/authorization/fuzzer:all-srcs",
        "//pkg/apis/authorization/install:all-srcs",
        "//pkg/apis/authorization/v1:all-srcs",
        "//pkg/apis/authorization/v1beta1:all-srcs",
        "//pkg/apis/authorization/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
