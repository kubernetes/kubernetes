// This is a generated file. Do not edit directly.

module k8s.io/sample-apiserver

go 1.13

require (
	github.com/coreos/go-semver v0.3.0 // indirect
	github.com/dustin/go-humanize v1.0.0 // indirect
	github.com/go-openapi/spec v0.19.3
	github.com/golang/glog v0.0.0-20190322123450-3c92600d7533 // indirect
	github.com/google/gofuzz v1.1.0
	github.com/spf13/cobra v1.0.0
	go.etcd.io/bbolt v1.3.5 // indirect
	k8s.io/apimachinery v0.19.0-rc.2
	k8s.io/apiserver v0.19.0-rc.2
	k8s.io/client-go v0.19.0-rc.2
	k8s.io/code-generator v0.19.0-rc.2
	k8s.io/component-base v0.19.0-rc.2
	k8s.io/klog/v2 v2.2.0
	k8s.io/kube-openapi v0.0.0-20200427153329-656914f816f9
)

replace (
	github.com/evanphx/json-patch => github.com/evanphx/json-patch v0.0.0-20190815234213-e83c0a1c26c8
	github.com/golang/glog => github.com/openshift/golang-glog v0.0.0-20190322123450-3c92600d7533
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	go.uber.org/multierr => go.uber.org/multierr v1.1.0
	golang.org/x/net => golang.org/x/net v0.0.0-20200324143707-d3edc9973b7e
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.2.8
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/sample-apiserver => ../sample-apiserver
)
