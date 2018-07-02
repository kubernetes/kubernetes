module k8s.io/metrics

require (
	k8s.io/api v1.12.0
	k8s.io/apimachinery v1.12.0
	k8s.io/client-go v1.12.0
)

replace (
	k8s.io/api v1.12.0 => ../api
	k8s.io/apimachinery v1.12.0 => ../apimachinery
	k8s.io/apiserver v1.12.0 => ../apiserver
	k8s.io/client-go v1.12.0 => ../client-go
)

require github.com/gogo/protobuf v0.0.0-20170330071051-c0656edd0d9e
