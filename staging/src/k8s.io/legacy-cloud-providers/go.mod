// This is a generated file. Do not edit directly.

module k8s.io/legacy-cloud-providers

go 1.16

require (
	cloud.google.com/go v0.54.0
	github.com/Azure/azure-sdk-for-go v43.0.0+incompatible
	github.com/Azure/go-autorest/autorest v0.11.12
	github.com/Azure/go-autorest/autorest/adal v0.9.5
	github.com/Azure/go-autorest/autorest/mocks v0.4.1
	github.com/Azure/go-autorest/autorest/to v0.2.0
	github.com/Azure/go-autorest/autorest/validation v0.1.0 // indirect
	github.com/GoogleCloudPlatform/k8s-cloud-provider v0.0.0-20200415212048-7901bc822317
	github.com/aws/aws-sdk-go v1.35.24
	github.com/blang/semver v3.5.1+incompatible
	github.com/dnaeon/go-vcr v1.0.1 // indirect
	github.com/golang/mock v1.4.4
	github.com/google/go-cmp v0.5.4
	github.com/gophercloud/gophercloud v0.1.0
	github.com/mitchellh/mapstructure v1.1.2
	github.com/rubiojr/go-vhd v0.0.0-20200706105327-02e210299021
	github.com/satori/go.uuid v1.2.0 // indirect
	github.com/stretchr/testify v1.6.1
	github.com/vmware/govmomi v0.20.3
	golang.org/x/crypto v0.0.0-20210220033148-5ea612d1eb83
	golang.org/x/oauth2 v0.0.0-20200107190931-bf48bf16ab8d
	google.golang.org/api v0.20.0
	google.golang.org/genproto v0.0.0-20201110150050-8816d57aaa9a // indirect
	gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/warnings.v0 v0.1.1 // indirect
	k8s.io/api v0.21.0-rc.0
	k8s.io/apimachinery v0.21.0-rc.0
	k8s.io/apiserver v0.21.0-rc.0
	k8s.io/client-go v0.21.0-rc.0
	k8s.io/cloud-provider v0.0.0
	k8s.io/component-base v0.21.0-rc.0
	k8s.io/csi-translation-lib v0.0.0
	k8s.io/klog/v2 v2.8.0
	k8s.io/utils v0.0.0-20201110183641-67b214c5f920
	sigs.k8s.io/yaml v1.2.0
)

replace (
	github.com/go-bindata/go-bindata => github.com/go-bindata/go-bindata v3.1.1+incompatible
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/mattn/go-colorable => github.com/mattn/go-colorable v0.0.9
	github.com/onsi/ginkgo => github.com/openshift/ginkgo v4.7.0-origin.0+incompatible
	github.com/openshift/client-go => github.com/openshift/client-go v0.0.0-20210422153130-25c8450d1535
	github.com/openshift/library-go => github.com/openshift/library-go v0.0.0-20210407092538-7021fda6f427
	github.com/robfig/cron => github.com/robfig/cron v1.1.0
	go.uber.org/multierr => go.uber.org/multierr v1.1.0
	k8s.io/api => ../api
	k8s.io/apiextensions-apiserver => ../apiextensions-apiserver
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/code-generator => ../code-generator
	k8s.io/component-base => ../component-base
	k8s.io/controller-manager => ../controller-manager
	k8s.io/csi-translation-lib => ../csi-translation-lib
	k8s.io/kube-aggregator => ../kube-aggregator
	k8s.io/legacy-cloud-providers => ../legacy-cloud-providers
)
