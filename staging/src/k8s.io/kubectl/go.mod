// This is a generated file. Do not edit directly.

module k8s.io/kubectl

go 1.13

require (
	github.com/MakeNowJust/heredoc v0.0.0-20170808103936-bb23615498cd
	github.com/chai2010/gettext-go v0.0.0-20160711120539-c6fed771bfd5
	github.com/davecgh/go-spew v1.1.1
	github.com/daviddengcn/go-colortext v0.0.0-20160507010035-511bcaf42ccd
	github.com/docker/distribution v2.7.1+incompatible
	github.com/evanphx/json-patch v4.2.0+incompatible
	github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d
	github.com/fatih/camelcase v1.0.0
	github.com/go-openapi/spec v0.19.3
	github.com/golangplus/bytes v0.0.0-20160111154220-45c989fe5450 // indirect
	github.com/golangplus/fmt v0.0.0-20150411045040-2a5d6d7d2995 // indirect
	github.com/golangplus/testing v0.0.0-20180327235837-af21d9c3145e // indirect
	github.com/google/go-cmp v0.4.0
	github.com/googleapis/gnostic v0.4.1
	github.com/jonboulle/clockwork v0.1.0
	github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de
	github.com/lithammer/dedent v1.1.0
	github.com/mitchellh/go-wordwrap v1.0.0
	github.com/moby/term v0.0.0-20200312100748-672ec06f55cd
	github.com/onsi/ginkgo v1.11.0
	github.com/onsi/gomega v1.7.0
	github.com/opencontainers/go-digest v1.0.0-rc1 // indirect
	github.com/russross/blackfriday v1.5.2
	github.com/spf13/cobra v1.0.0
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.4.0
	github.com/xlab/handysort v0.0.0-20150421192137-fb3537ed64a1 // indirect
	golang.org/x/sys v0.0.0-20200622214017-ed371f2e16b4
	gopkg.in/yaml.v2 v2.3.0
	k8s.io/api v0.19.0-rc.2
	k8s.io/apimachinery v0.19.0-rc.2
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.19.0-rc.2
	k8s.io/component-base v0.19.0-rc.2
	k8s.io/klog/v2 v2.2.0
	k8s.io/kube-openapi v0.0.0-20200427153329-656914f816f9
	k8s.io/metrics v0.0.0
	k8s.io/utils v0.0.0-20200720150651-0bdb4ca86cbc
	sigs.k8s.io/kustomize v2.0.3+incompatible
	sigs.k8s.io/yaml v1.2.0
	vbom.ml/util v0.0.0-20180919145318-efcd4e0f9787
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
	k8s.io/cli-runtime => ../cli-runtime
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/kubectl => ../kubectl
	k8s.io/metrics => ../metrics
	vbom.ml/util => vbom.ml/util v0.0.0-20160121211510-db5cfe13f5cc
)
