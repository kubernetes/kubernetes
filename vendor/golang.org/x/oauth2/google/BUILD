load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "appengine.go",
        "appengine_gen2_flex.go",
        "default.go",
        "doc.go",
        "google.go",
        "jwt.go",
        "sdk.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/oauth2/google",
    importpath = "golang.org/x/oauth2/google",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/cloud.google.com/go/compute/metadata:go_default_library",
        "//vendor/golang.org/x/oauth2:go_default_library",
        "//vendor/golang.org/x/oauth2/internal:go_default_library",
        "//vendor/golang.org/x/oauth2/jws:go_default_library",
        "//vendor/golang.org/x/oauth2/jwt:go_default_library",
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
    visibility = ["//visibility:public"],
)
