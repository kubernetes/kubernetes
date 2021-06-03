// This is a generated file. Do not edit directly.

module k8s.io/cloud-provider

go 1.16

require (
	github.com/google/go-cmp v0.5.4
	github.com/spf13/cobra v1.1.1
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/controller-manager v0.0.0
	k8s.io/klog/v2 v2.9.0
	k8s.io/utils v0.0.0-20201110183641-67b214c5f920
)

replace (
	github.com/alecthomas/units => github.com/alecthomas/units v0.0.0-20190717042225-c3de453c63f4
	github.com/go-logfmt/logfmt => github.com/go-logfmt/logfmt v0.4.0
	github.com/julienschmidt/httprouter => github.com/julienschmidt/httprouter v1.2.0
	github.com/mwitkow/go-conntrack => github.com/mwitkow/go-conntrack v0.0.0-20161129095857-cc309e4a2223
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/component-base => ../component-base
	k8s.io/controller-manager => ../controller-manager
)
