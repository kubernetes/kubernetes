// This is a generated file. Do not edit directly.

module k8s.io/metrics

go 1.16

require (
	github.com/gogo/protobuf v1.3.2
	github.com/stretchr/testify v1.7.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/code-generator v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/code-generator => ../code-generator
	k8s.io/gengo => github.com/liggitt/gengo v0.0.0-20210812145509-5cf49e200430
	k8s.io/metrics => ../metrics
)
