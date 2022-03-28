// This is a generated file. Do not edit directly.

module k8s.io/cluster-bootstrap

go 1.16

require (
	github.com/stretchr/testify v1.6.1
	golang.org/x/crypto v0.0.0-20211202192323-5770296d904e // indirect
	gopkg.in/square/go-jose.v2 v2.2.2
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/klog/v2 v2.9.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/cluster-bootstrap => ../cluster-bootstrap
)
