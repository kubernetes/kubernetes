# A project wanting to use bazel to build vendored k8s.io/kubernetes
# will need to set the following variables in //build/vendored.bzl in
# their project and customize the go prefix:
#
# vendored_go_prefix = "k8s.io/myproject/"
# vendored_path_prefix = "vendor/k8s.io/kubernetes/"

vendored_go_prefix = "k8s.io/kubernetes/"
vendored_path_prefix = ""
