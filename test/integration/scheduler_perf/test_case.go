package benchmark

import (
	v1 "k8s.io/api/core/v1"
	testutils "k8s.io/kubernetes/test/utils"
)

// benchmarkCase configures a test case to run the scheduler benchmark test. Users should be able to
// provide this via a Json/YAML file.
//
// It specifies nodes and pods in the cluster before running the test. It also specifies the pods to
// schedule during the test. The config can be as simple as just specify number of nodes/pods, where
// default spec will be applied. It also allows the user to specify a pod spec template for more compicated
// test cases.
//
// It also specifies the metrics to be collected after the test. If nothing is specified, default metrics
// such as scheduling throughput and latencies will be collected.
type benchmarkCase struct {
	// configures nodes in the cluster
	nodes []nodeCase
	// configures pods in the cluster before running the tests
	initPods []podCase
	// pods to be scheduled during the test.
	podsToSchedule []podCase

	// optional, if not specified, a default prometheusMetricCollector will be used.
	prometheusMetricCollector *prometheusMetricCollector

	// Custom metrics collectors can be configured here.
	myCollector *myCollector
}

type nodeCase struct {
	num int
	// At most one of the following strategies can be defined. If not specified, default to TrivialNodePrepareStrategy.
	nodeAllocatableStrategy *testutils.NodeAllocatableStrategy
	labelNodePrepareStrategy *testutils.LabelNodePrepareStrategy
}

type podCase struct {
	num int
	// Optional, if not provided, will default to a default pod spec
	podTemplate *v1.Pod
	// Optional, if provided, will call CreatePodWithPersistentVolume
	pvcClaimTemplate  *v1.PersistentVolumeClaim
}
