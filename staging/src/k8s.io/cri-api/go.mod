// This is a generated file. Do not edit directly.

module k8s.io/cri-api

go 1.13

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/gogo/protobuf v1.3.2
	github.com/google/go-cmp v0.3.0 // indirect
	github.com/kr/pretty v0.1.0 // indirect
	github.com/stretchr/testify v1.4.0
	golang.org/x/net v0.0.0-20201110031124-69a78807bb2b // indirect
	golang.org/x/sys v0.0.0-20201112073958-5cba982894dd // indirect
	google.golang.org/grpc v1.26.0
	gopkg.in/check.v1 v1.0.0-20180628173108-788fd7840127 // indirect
	gopkg.in/yaml.v2 v2.2.8 // indirect
)

replace (
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20200220183623-bac4c82f6975
	golang.org/x/text => golang.org/x/text v0.3.2
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/cri-api => ../cri-api
)
