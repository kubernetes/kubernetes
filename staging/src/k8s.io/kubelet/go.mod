// This is a generated file. Do not edit directly.

module k8s.io/kubelet

go 1.12

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
)

replace (
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/text => golang.org/x/text v0.3.1-0.20181227161524-e6919f6577db
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/gengo => github.com/wk8/gengo v0.0.0-20191001015530-73ff7e40e4d8d96c736cc6ae3fd7f3997f6c70a6
	k8s.io/kubelet => ../kubelet
)
