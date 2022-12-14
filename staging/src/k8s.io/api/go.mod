// This is a generated file. Do not edit directly.

module k8s.io/api

go 1.19

require (
	github.com/gogo/protobuf v1.3.2
	github.com/stretchr/testify v1.8.0
	k8s.io/apimachinery v0.26.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/go-logr/logr v1.2.3 // indirect
	github.com/golang/protobuf v1.5.2 // indirect
	github.com/google/go-cmp v0.5.9 // indirect
	github.com/google/gofuzz v1.2.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/niemeyer/pretty v0.0.0-20200227124842-a10e7caefd8e // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/net v0.3.1-0.20221206200815-1e63c2f08a10 // indirect
	golang.org/x/text v0.5.0 // indirect
	google.golang.org/protobuf v1.28.1 // indirect
	gopkg.in/check.v1 v1.0.0-20200227125254-8fa46927fb4f // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/klog/v2 v2.80.1 // indirect
	k8s.io/utils v0.0.0-20221107191617-1a15be271d1d // indirect
	sigs.k8s.io/json v0.0.0-20220713155537-f223a00ba0e2 // indirect
	sigs.k8s.io/structured-merge-diff/v4 v4.2.3 // indirect
	sigs.k8s.io/yaml v1.3.0 // indirect
)

replace (
	github.com/onsi/ginkgo/v2 => github.com/openshift/onsi-ginkgo/v2 v2.4.1-0.20221214150008-e73634cb3870
	github.com/openshift/api => github.com/sanchezl/api v0.0.0-20221212212126-51bab418a0a6
	github.com/openshift/apiserver-library-go => github.com/sanchezl/apiserver-library-go v0.0.0-20221213204702-3d7b8923e8be
	github.com/openshift/client-go => github.com/sanchezl/client-go v0.0.0-20221213024106-45c6078f5efe
	github.com/openshift/library-go => github.com/sanchezl/library-go v0.0.0-20221213154741-26ea0c85fa9d
	k8s.io/api => ../api
	k8s.io/apiextensions-apiserver => ../apiextensions-apiserver
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/component-helpers => ../component-helpers
	k8s.io/kms => ../kms
	k8s.io/kube-aggregator => ../kube-aggregator
)
