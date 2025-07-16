This package is copy/pasted from [k8s.io/apimachinery](https://github.com/kubernetes/apimachinery/tree/master/pkg/util/sets) 
to avoid a circular dependency with `openshift/kubernetes` as it requires OTE and, without having done this, 
OTE would require `kubernetes/kubernetes`.
