package crdinstall

import "k8s.io/apimachinery/pkg/runtime"

var installCRDs []runtime.Object

func init() {
	// objects added to installCRDs will be installed during the cluster boot up
	// the objects are "applied" in the given order
	installCRDs = append(installCRDs, panipuriCRD)
}

func Objects() []runtime.Object {
	return installCRDs
}
