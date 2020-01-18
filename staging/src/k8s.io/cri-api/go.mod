// This is a generated file. Do not edit directly.

module k8s.io/cri-api

go 1.13

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/gogo/protobuf v1.3.1
	github.com/google/go-cmp v0.3.0 // indirect
	github.com/kr/pretty v0.1.0 // indirect
	github.com/stretchr/testify v1.4.0
	golang.org/x/net v0.0.0-20191004110552-13f9640d40b9 // indirect
	golang.org/x/sys v0.0.0-20190826190057-c7b8b68b1456 // indirect
	golang.org/x/text v0.3.2 // indirect
	google.golang.org/grpc v1.26.0
	gopkg.in/check.v1 v1.0.0-20180628173108-788fd7840127 // indirect
	gopkg.in/yaml.v2 v2.2.7 // indirect
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/cri-api => ../cri-api
)
