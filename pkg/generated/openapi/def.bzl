load("@io_bazel_rules_go//go:def.bzl", "go_library")
load("@io_kubernetes_build//defs:go.bzl", "go_genrule")

def openapi_library(name, tags, srcs, openapi_targets=[], vendor_targets=[]):
  deps = [
      "//vendor/github.com/go-openapi/spec:go_default_library",
      "//vendor/k8s.io/apimachinery/pkg/openapi:go_default_library",
  ] + ["//%s:go_default_library" % target for target in openapi_targets] + ["//vendor/%s:go_default_library" % target for target in vendor_targets]
  go_library(
      name=name,
      tags=tags,
      srcs=srcs + [":zz_generated.openapi"],
      deps=deps,
  )
  go_genrule(
      name = "zz_generated.openapi",
      srcs = srcs + ["//hack/boilerplate:boilerplate.go.txt"],
      outs = ["zz_generated.openapi.go"],
      cmd = " ".join([
        "$(location //cmd/libs/go2idl/openapi-gen)",
        "--v 1",
        "--logtostderr",
        "--go-header-file $(location //hack/boilerplate:boilerplate.go.txt)",
        "--output-file-base zz_generated.openapi",
        "--output-package k8s.io/kubernetes/pkg/generated/openapi",
        "--input-dirs " + ",".join(["k8s.io/kubernetes/" + target for target in openapi_targets] + ["k8s.io/kubernetes/vendor/" + target for target in vendor_targets]),
        "&& cp pkg/generated/openapi/zz_generated.openapi.go $(GENDIR)/pkg/generated/openapi",
      ]),
      go_deps = deps,
      tools = ["//cmd/libs/go2idl/openapi-gen"],
)
