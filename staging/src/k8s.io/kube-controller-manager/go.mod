// This is a generated file. Do not edit directly.

module k8s.io/kube-controller-manager

go 1.13

require (
	k8s.io/apimachinery v0.0.0
	k8s.io/component-base v0.0.0
)

replace (
	cloud.google.com/go => cloud.google.com/go v0.38.0
	github.com/kr/pty => github.com/kr/pty v1.1.5
	github.com/kr/text => github.com/kr/text v0.1.0
	github.com/onsi/gomega => github.com/onsi/gomega v1.7.0
	github.com/stretchr/testify => github.com/stretchr/testify v1.4.0
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20190604053449-0f29369cfe45
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	google.golang.org/api => google.golang.org/api v0.15.1
	google.golang.org/appengine => google.golang.org/appengine v1.5.0
	google.golang.org/genproto => google.golang.org/genproto v0.0.0-20191230161307-f3c370f40bfb
	google.golang.org/grpc => google.golang.org/grpc v1.26.0
	gopkg.in/check.v1 => gopkg.in/check.v1 v1.0.0-20190902080502-41f04d3bba15
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
	k8s.io/kube-controller-manager => ../kube-controller-manager
	sigs.k8s.io/structured-merge-diff/v3 => github.com/kwiesmueller/structured-merge-diff/v3 v3.0.0-20200518173228-02d431a3e322
)
