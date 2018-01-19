package vsphere

import (
	"k8s.io/kubernetes/test/e2e/framework"
)

var BootstrapCounter = 0

func Bootstrap(testContext *framework.TestContextType) {
	BootstrapCounter++
	framework.Logf("VSphere BootstrapCounter %d ", BootstrapCounter)
	testContext.VSphereTestContext = &Context{}
	testContext.VSphereTestContext.TestString = "string is set"
	// TBD
	// 1. Read vSphere conf and get VSphere instances
	// 2. Get Node to VSphere mapping
	// 3. Set NodeMapper in vSphere context
}