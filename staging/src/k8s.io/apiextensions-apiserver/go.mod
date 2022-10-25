// This is a generated file. Do not edit directly.

module k8s.io/apiextensions-apiserver

go 1.16

require (
	github.com/emicklei/go-restful v2.9.5+incompatible
	github.com/gogo/protobuf v1.3.2
	github.com/google/cel-go v0.10.1
	github.com/google/gnostic v0.5.7-v3refs
	github.com/google/go-cmp v0.5.8
	github.com/google/gofuzz v1.1.0
	github.com/google/uuid v1.1.2
	github.com/kcp-dev/apimachinery v0.0.0-20221019133255-9e1e13940519
	github.com/kcp-dev/client-go v0.0.0-20221025140308-a18ccea074a6
	github.com/kcp-dev/logicalcluster/v2 v2.0.0-alpha.1
	github.com/mitchellh/mapstructure v1.4.1 // indirect
	github.com/spf13/cobra v1.4.0
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.1
	go.etcd.io/etcd/client/pkg/v3 v3.5.1
	go.etcd.io/etcd/client/v3 v3.5.1
	google.golang.org/genproto v0.0.0-20220107163113-42d7afdf6368
	google.golang.org/grpc v1.40.0
	google.golang.org/protobuf v1.28.0
	gopkg.in/yaml.v2 v2.4.0
	k8s.io/api v0.24.3
	k8s.io/apimachinery v0.24.3
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.24.3
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog/v2 v2.70.1
	k8s.io/kube-openapi v0.0.0-20220328201542-3ee0da9b0b42
	k8s.io/utils v0.0.0-20220728103510-ee6ede2d64ed
	sigs.k8s.io/json v0.0.0-20220713155537-f223a00ba0e2
	sigs.k8s.io/structured-merge-diff/v4 v4.2.3
	sigs.k8s.io/yaml v1.3.0
)

replace (
	github.com/go-logr/logr => github.com/go-logr/logr v1.2.0
	github.com/google/go-cmp => github.com/google/go-cmp v0.5.5
	github.com/stretchr/testify => github.com/stretchr/testify v1.7.0
	golang.org/x/net => golang.org/x/net v0.0.0-20220127200216-cd36cc0744dd
	golang.org/x/sys => golang.org/x/sys v0.0.0-20220209214540-3681064d5158
	google.golang.org/protobuf => google.golang.org/protobuf v1.27.1
	k8s.io/api => ../api
	k8s.io/apiextensions-apiserver => ../apiextensions-apiserver
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/klog/v2 => k8s.io/klog/v2 v2.60.1
	k8s.io/utils => k8s.io/utils v0.0.0-20220210201930-3a6ce19ff2f9
	sigs.k8s.io/json => sigs.k8s.io/json v0.0.0-20211208200746-9f7c6b3444d2
	sigs.k8s.io/structured-merge-diff/v4 => sigs.k8s.io/structured-merge-diff/v4 v4.2.1
	sigs.k8s.io/yaml => sigs.k8s.io/yaml v1.2.0
)
