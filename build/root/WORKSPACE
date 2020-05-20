workspace(name = "io_k8s_kubernetes")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//build:workspace_mirror.bzl", "mirror")

http_archive(
    name = "io_k8s_repo_infra",
    sha256 = "55e56b332ead9c32e1d53c9834a5c918a4cecd6859b70645eba6cd10372fd68f",
    strip_prefix = "repo-infra-0.0.5",
    urls = [
        "https://github.com/kubernetes/repo-infra/archive/v0.0.5.tar.gz",
    ],
)

load("@io_k8s_repo_infra//:load.bzl", repo_infra_repositories = "repositories")

repo_infra_repositories()

load("@io_k8s_repo_infra//:repos.bzl", repo_infra_configure = "configure", repo_infra_go_repositories = "go_repositories")

repo_infra_configure(
    go_version = "1.13.9",
    minimum_bazel_version = "2.2.0",
)

repo_infra_go_repositories()

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
    strip_prefix = "rules_docker-0.7.0",
    urls = mirror("https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"),
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

container_pull(
    name = "distroless_base",
    digest = "sha256:7fa7445dfbebae4f4b7ab0e6ef99276e96075ae42584af6286ba080750d6dfe5",
    registry = "gcr.io",
    repository = "distroless/base",
    tag = "latest",  # ignored when digest provided, but kept here for documentation.
)

load("//build:workspace.bzl", "release_dependencies")

release_dependencies()

load("//build:workspace_mirror.bzl", "export_urls")

export_urls("workspace_urls")
