// This is a generated file. Do not edit directly.

module k8s.io/api

go 1.14

require (
	github.com/gogo/protobuf v1.3.1
	github.com/stretchr/testify v1.4.0
	k8s.io/apimachinery v0.0.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20200201011859-915c9c3d4ccf // pinned to release-branch.go1.14-std
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
)
