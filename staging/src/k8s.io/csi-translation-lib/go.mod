// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.13

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/cloud-provider v0.0.0
)

replace (
	cloud.google.com/go => cloud.google.com/go v0.38.0
	golang.org/x/net => golang.org/x/net v0.0.0-20191004110552-13f9640d40b9
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20190604053449-0f29369cfe45
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	google.golang.org/api => google.golang.org/api v0.6.1-0.20190607001116-5213b8090861
	google.golang.org/genproto => google.golang.org/genproto v0.0.0-20190819201941-24fa4b261c55
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/csi-translation-lib => ../csi-translation-lib
)
