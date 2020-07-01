module github.com/JeremyOT/mcs-api

go 1.13

require (
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/mcs-api => ../mcs-api
)
