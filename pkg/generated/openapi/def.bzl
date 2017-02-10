load("@io_bazel_rules_go//go:def.bzl", "go_library")
load("@io_kubernetes_build//defs:go.bzl", "go_genrule")

OPENAPI_TARGETS = [
    "cmd/libs/go2idl/client-gen/test_apis/testgroup/v1",
    "cmd/libs/go2idl/openapi-gen/generators",
    "federation/apis/federation/v1beta1",
    "pkg/api/v1",
    "pkg/apis/abac/v0",
    "pkg/apis/abac/v1beta1",
    "pkg/apis/apps/v1beta1",
    "pkg/apis/authentication/v1",
    "pkg/apis/authentication/v1beta1",
    "pkg/apis/authorization/v1",
    "pkg/apis/authorization/v1beta1",
    "pkg/apis/autoscaling/v1",
    "pkg/apis/autoscaling/v2alpha1",
    "pkg/apis/batch/v1",
    "pkg/apis/batch/v2alpha1",
    "pkg/apis/certificates/v1beta1",
    "pkg/apis/componentconfig/v1alpha1",
    "pkg/apis/extensions/v1beta1",
    "pkg/apis/imagepolicy/v1alpha1",
    "pkg/apis/policy/v1beta1",
    "pkg/apis/rbac/v1alpha1",
    "pkg/apis/rbac/v1beta1",
    "pkg/apis/storage/v1beta1",
    "pkg/version",
]

OPENAPI_VENDOR_TARGETS = [
    "k8s.io/apimachinery/pkg/api/resource",
    "k8s.io/apimachinery/pkg/apis/meta/v1",
    "k8s.io/apimachinery/pkg/runtime",
    "k8s.io/apimachinery/pkg/util/intstr",
    "k8s.io/apimachinery/pkg/version",
    "k8s.io/apiserver/pkg/apis/example/v1",
    "k8s.io/apiserver/pkg/server/openapi",
    "k8s.io/client-go/pkg/api/v1",
]


def openapi_library(name, tags, srcs):
  deps = [
      "//vendor:github.com/go-openapi/spec",
      "//vendor:k8s.io/apimachinery/pkg/openapi",
  ] + ["//%s:go_default_library" % target for target in OPENAPI_TARGETS] + ["//vendor:%s" % target for target in OPENAPI_VENDOR_TARGETS]
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
        "--input-dirs " + ",".join(["k8s.io/kubernetes/" + target for target in OPENAPI_TARGETS] + ["k8s.io/kubernetes/vendor/" + target for target in OPENAPI_VENDOR_TARGETS]),
        "&& cp pkg/generated/openapi/zz_generated.openapi.go $(GENDIR)/pkg/generated/openapi",
      ]),
      go_deps = deps,
      tools = ["//cmd/libs/go2idl/openapi-gen"],
)
