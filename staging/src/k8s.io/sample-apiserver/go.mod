// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/sample-apiserver

go 1.16

require (
	github.com/NYTimes/gziphandler v1.1.1 // indirect
	github.com/google/gofuzz v1.1.0
	github.com/matttproud/golang_protobuf_extensions v1.0.2-0.20181231171920-c182affec369 // indirect
	github.com/spf13/cobra v1.1.3
	google.golang.org/appengine v1.6.5 // indirect
	k8s.io/apimachinery v0.20.6
	k8s.io/apiserver v0.20.6
	k8s.io/client-go v0.20.6
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.20.6
	k8s.io/klog/v2 v2.9.0
	k8s.io/kube-openapi v0.0.0-20210421082810-95288971da7e
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/sample-apiserver => ../sample-apiserver
)
