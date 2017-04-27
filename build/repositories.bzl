load("@io_bazel_rules_go//go:def.bzl", "go_repositories")

def repositories():
  go_repositories(
      go_version = "1.8.1",
  )
