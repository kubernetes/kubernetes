module k8s.io/kubernetes/test/images/agnhost

go 1.17

replace (
	k8s.io/api => ../../../staging/src/k8s.io/api
	k8s.io/apiextensions-apiserver => ../../../staging/src/k8s.io/apiextensions-apiserver
	k8s.io/apimachinery => ../../../staging/src/k8s.io/apimachinery
	k8s.io/apiserver => ../../../staging/src/k8s.io/apiserver
	k8s.io/cli-runtime => ../../../staging/src/k8s.io/cli-runtime
	k8s.io/client-go => ../../../staging/src/k8s.io/client-go
	k8s.io/cloud-provider => ../../../staging/src/k8s.io/cloud-provider
	k8s.io/cluster-bootstrap => ../../../staging/src/k8s.io/cluster-bootstrap
	k8s.io/code-generator => ../../../staging/src/k8s.io/code-generator
	k8s.io/component-base => ../../../staging/src/k8s.io/component-base
	k8s.io/component-helpers => ../../../staging/src/k8s.io/component-helpers
	k8s.io/controller-manager => ../../../staging/src/k8s.io/controller-manager
	k8s.io/cri-api => ../../../staging/src/k8s.io/cri-api
	k8s.io/csi-translation-lib => ../../../staging/src/k8s.io/csi-translation-lib
	k8s.io/kube-aggregator => ../../../staging/src/k8s.io/kube-aggregator
	k8s.io/kube-controller-manager => ../../../staging/src/k8s.io/kube-controller-manager
	k8s.io/kube-proxy => ../../../staging/src/k8s.io/kube-proxy
	k8s.io/kube-scheduler => ../../../staging/src/k8s.io/kube-scheduler
	k8s.io/kubectl => ../../../staging/src/k8s.io/kubectl
	k8s.io/kubelet => ../../../staging/src/k8s.io/kubelet
	k8s.io/kubernetes => ../../../
	k8s.io/legacy-cloud-providers => ../../../staging/src/k8s.io/legacy-cloud-providers
	k8s.io/metrics => ../../../staging/src/k8s.io/metrics
	k8s.io/mount-utils => ../../../staging/src/k8s.io/mount-utils
	k8s.io/pod-security-admission => ../../../staging/src/k8s.io/pod-security-admission
	k8s.io/sample-apiserver => ../../../staging/src/k8s.io/sample-apiserver
	k8s.io/sample-cli-plugin => ../../../staging/src/k8s.io/sample-cli-plugin
	k8s.io/sample-controller => ../../../staging/src/k8s.io/sample-controller
)

require (
	github.com/coreos/go-oidc v2.2.1+incompatible
	github.com/evanphx/json-patch v4.12.0+incompatible
	github.com/google/gofuzz v1.2.0
	github.com/ishidawataru/sctp v0.0.0-20210707070123-9a39160e9062
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/spf13/cobra v1.2.1
	golang.org/x/oauth2 v0.0.0-20211104180415-d3ed0bb246c8
	golang.org/x/sys v0.0.0-20211213223007-03aa0b5f6827
	gopkg.in/square/go-jose.v2 v2.6.0
	k8s.io/api v0.0.0
	k8s.io/apiextensions-apiserver v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog/v2 v2.30.0
	k8s.io/kubernetes v0.0.0-00010101000000-000000000000
	k8s.io/utils v0.0.0-20211208161948-7d6a63dca704
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/blang/semver v3.5.1+incompatible // indirect
	github.com/cespare/xxhash/v2 v2.1.1 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/go-logr/logr v1.2.0 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/protobuf v1.5.2 // indirect
	github.com/google/go-cmp v0.5.5 // indirect
	github.com/google/uuid v1.1.2 // indirect
	github.com/googleapis/gnostic v0.5.5 // indirect
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.2-0.20181231171920-c182affec369 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/onsi/ginkgo v1.16.4 // indirect
	github.com/onsi/gomega v1.13.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pquerna/cachecontrol v0.0.0-20171018203845-0dec1b30a021 // indirect
	github.com/prometheus/client_golang v1.11.0 // indirect
	github.com/prometheus/client_model v0.2.0 // indirect
	github.com/prometheus/common v0.28.0 // indirect
	github.com/prometheus/procfs v0.6.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/crypto v0.0.0-20210817164053-32db794688a5 // indirect
	golang.org/x/net v0.0.0-20211209124913-491a49abca63 // indirect
	golang.org/x/term v0.0.0-20210615171337-6886f2dfbf5b // indirect
	golang.org/x/text v0.3.7 // indirect
	golang.org/x/time v0.0.0-20210723032227-1f47c861a9ac // indirect
	google.golang.org/appengine v1.6.7 // indirect
	google.golang.org/protobuf v1.27.1 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.0-20210107192922-496545a6307b // indirect
	k8s.io/kube-openapi v0.0.0-20211115234752-e816edb12b65 // indirect
	sigs.k8s.io/json v0.0.0-20211020170558-c049b76a60c6 // indirect
	sigs.k8s.io/structured-merge-diff/v4 v4.1.2 // indirect
	sigs.k8s.io/yaml v1.2.0 // indirect
)
