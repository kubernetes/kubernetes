package main

import (
	"k8s.io/kubernetes/openshift-hack/e2e/annotate"
)

func main() {
	annotate.Run(annotate.TestMaps, func(name string) bool { return false })
}
