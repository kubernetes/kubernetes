// This is a generated file. Do not edit directly.

module k8s.io/client-go

go 1.24.0

godebug default=go1.24

require (
	github.com/go-logr/logr v1.4.2
	github.com/gogo/protobuf v1.3.2
	github.com/google/gnostic-models v0.6.9
	github.com/google/go-cmp v0.6.0
	github.com/google/gofuzz v1.2.0
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/gregjones/httpcache v0.0.0-20190611155906-901d90724c79
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/peterbourgon/diskv v2.0.1+incompatible
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.10.0
	go.uber.org/goleak v1.3.0
	golang.org/x/net v0.33.0
	golang.org/x/oauth2 v0.27.0
	golang.org/x/term v0.29.0
	golang.org/x/time v0.7.0
	google.golang.org/protobuf v1.35.2
	gopkg.in/evanphx/json-patch.v4 v4.12.0
	k8s.io/klog/v2 v2.130.1
	k8s.io/kube-openapi v0.0.0-20241212222426-2c72e554b1e7
	k8s.io/utils v0.0.0-20241104100929-3ea5e8cea738
	sigs.k8s.io/json v0.0.0-20241010143419-9aa6b5e7a4b3
	sigs.k8s.io/structured-merge-diff/v4 v4.4.2
	sigs.k8s.io/yaml v1.4.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/emicklei/go-restful/v3 v3.11.0 // indirect
	github.com/fxamacker/cbor/v2 v2.7.0 // indirect
	github.com/go-openapi/jsonpointer v0.21.0 // indirect
	github.com/go-openapi/jsonreference v0.20.2 // indirect
	github.com/go-openapi/swag v0.23.0 // indirect
	github.com/google/btree v1.1.3 // indirect
	github.com/google/pprof v0.0.0-20241029153458-d1b30febd7db // indirect
	github.com/josharian/intern v1.0.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	github.com/moby/spdystream v0.5.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/mxk/go-flowrate v0.0.0-20140419014527-cca7078d478f // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/stretchr/objx v0.5.2 // indirect
	github.com/x448/float16 v0.8.4 // indirect
	golang.org/x/sys v0.30.0 // indirect
	golang.org/x/text v0.22.0 // indirect
	golang.org/x/tools v0.26.0 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/api v0.0.0
	k8s.io/apiextensions-apiserver v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/cli-runtime v0.0.0
	k8s.io/cloud-provider v0.0.0
	k8s.io/cluster-bootstrap v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/component-helpers v0.0.0
	k8s.io/controller-manager v0.0.0
	k8s.io/cri-api v0.0.0
	k8s.io/cri-client v0.0.0
	k8s.io/csi-translation-lib v0.0.0
	k8s.io/dynamic-resource-allocation v0.0.0
	k8s.io/endpointslice v0.0.0
	k8s.io/externaljwt v0.0.0
	k8s.io/kms v0.0.0
	k8s.io/kube-aggregator v0.0.0
	k8s.io/kube-controller-manager v0.0.0
	k8s.io/kube-proxy v0.0.0
	k8s.io/kube-scheduler v0.0.0
	k8s.io/kubectl v0.0.0
	k8s.io/kubelet v0.0.0
	k8s.io/metrics v0.0.0
	k8s.io/mount-utils v0.0.0
	k8s.io/pod-security-admission v0.0.0
	k8s.io/sample-apiserver v0.0.0
	k8s.io/sample-cli-plugin v0.0.0
	k8s.io/sample-controller v0.0.0
)

replace k8s.io/api => ../api

replace k8s.io/apiextensions-apiserver => ../apiextensions-apiserver

replace k8s.io/apimachinery => ../apimachinery

replace k8s.io/apiserver => ../apiserver

replace k8s.io/cli-runtime => ../cli-runtime

replace k8s.io/client-go => ../client-go

replace k8s.io/cloud-provider => ../cloud-provider

replace k8s.io/cluster-bootstrap => ../cluster-bootstrap

replace k8s.io/code-generator => ../code-generator

replace k8s.io/component-base => ../component-base

replace k8s.io/component-helpers => ../component-helpers

replace k8s.io/controller-manager => ../controller-manager

replace k8s.io/cri-api => ../cri-api

replace k8s.io/cri-client => ../cri-client

replace k8s.io/csi-translation-lib => ../csi-translation-lib

replace k8s.io/dynamic-resource-allocation => ../dynamic-resource-allocation

replace k8s.io/endpointslice => ../endpointslice

replace k8s.io/externaljwt => ../externaljwt

replace k8s.io/kms => ../kms

replace k8s.io/kube-aggregator => ../kube-aggregator

replace k8s.io/kube-controller-manager => ../kube-controller-manager

replace k8s.io/kube-proxy => ../kube-proxy

replace k8s.io/kube-scheduler => ../kube-scheduler

replace k8s.io/kubectl => ../kubectl

replace k8s.io/kubelet => ../kubelet

replace k8s.io/metrics => ../metrics

replace k8s.io/mount-utils => ../mount-utils

replace k8s.io/pod-security-admission => ../pod-security-admission

replace k8s.io/sample-apiserver => ../sample-apiserver

replace k8s.io/sample-cli-plugin => ../sample-cli-plugin

replace k8s.io/sample-controller => ../sample-controller
