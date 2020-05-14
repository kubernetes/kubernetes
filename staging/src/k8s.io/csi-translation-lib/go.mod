// This is a generated file. Do not edit directly.

module k8s.io/csi-translation-lib

go 1.13

require (
	github.com/stretchr/testify v1.4.0
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/cloud-provider v0.0.0
	k8s.io/klog v1.0.0
)

replace (
	golang.org/x/image => golang.org/x/image v0.0.0-20190227222117-0694c2d4d067
	golang.org/x/mobile => golang.org/x/mobile v0.0.0-20190312151609-d3739f865fa6
	golang.org/x/mod => golang.org/x/mod v0.0.0-20190513183733-4bf6d317e70e
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20190604053449-0f29369cfe45
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/cloud-provider => ../cloud-provider
	k8s.io/csi-translation-lib => ../csi-translation-lib
)
