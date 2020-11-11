// This is a generated file. Do not edit directly.

module k8s.io/cluster-bootstrap

go 1.15

require (
	github.com/stretchr/testify v1.4.0
	golang.org/x/crypto v0.0.0-20201002170205-7f63de1d35b0 // indirect
	gopkg.in/square/go-jose.v2 v2.2.2
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog/v2 v2.4.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cluster-bootstrap => ../cluster-bootstrap
	sigs.k8s.io/structured-merge-diff/v4 => github.com/kwiesmueller/structured-merge-diff/v4 v4.0.0-20201110160604-4a28bb34fff5
)
