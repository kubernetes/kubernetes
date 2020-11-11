// This is a generated file. Do not edit directly.

module k8s.io/api

go 1.15

require (
	github.com/gogo/protobuf v1.3.1
	github.com/stretchr/testify v1.4.0
	k8s.io/apimachinery v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	sigs.k8s.io/structured-merge-diff/v4 => github.com/kwiesmueller/structured-merge-diff/v4 v4.0.0-20201110160604-4a28bb34fff5
)
