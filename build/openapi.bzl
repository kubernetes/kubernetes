# A project wanting to generate openapi code for vendored
# k8s.io/kubernetes will need to set the following variables in
# //build/openapi.bzl in their project and customize the go prefix:
#
# openapi_go_prefix = "k8s.io/myproject/"
# openapi_vendor_prefix = "vendor/k8s.io/kubernetes/"

openapi_go_prefix = "k8s.io/kubernetes/"
openapi_vendor_prefix = ""
