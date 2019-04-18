// This is a generated file. Do not edit directly.

module k8s.io/component-base

go 1.12

require (
	github.com/spf13/pflag v1.0.1
	k8s.io/apimachinery v0.0.0
	k8s.io/klog v0.0.0-20190306015804-8e90cee79f82
	k8s.io/utils v0.0.0-20190221042446-c2654d5206da
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190313210603-aa82965741a9
	k8s.io/apimachinery => ../apimachinery
	k8s.io/component-base => ../component-base
)
