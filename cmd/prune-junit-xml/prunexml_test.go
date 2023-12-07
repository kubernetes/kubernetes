/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"bufio"
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPruneXML(t *testing.T) {
	sourceXML := `<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
	<testsuite tests="3" failures="1" time="271.610000" name="k8s.io/kubernetes/test/integration/apiserver" timestamp="">
		<properties>
			<property name="go.version" value="go1.18 linux/amd64"></property>
		</properties>
		<testcase classname="k8s.io/kubernetes/test/integration/apimachinery" name="TestWatchRestartsIfTimeoutNotReached/group/InformerWatcher_survives_closed_watches" time="30.050000"></testcase>
		<testcase classname="k8s.io/kubernetes/test/integration/apiserver" name="TestMaxResourceSize/JSONPatchType_should_handle_a_patch_just_under_the_max_limit" time="0.000000">
			<skipped message="=== RUN   TestMaxResourceSize/JSONPatchType_should_handle_a_patch_just_under_the_max_limit&#xA;    max_request_body_bytes_test.go:89: skipping expensive test&#xA;    --- SKIP: TestMaxResourceSize/JSONPatchType_should_handle_a_patch_just_under_the_max_limit (0.00s)&#xA;"></skipped>
		</testcase>
		<testcase classname="k8s.io/kubernetes/test/integration/apimachinery" name="TestSchedulerInformers" time="-0.000000">
			<failure message="Failed" type="">&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/transport/transport.go:169 +0x147&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/transport.(*transportReader).Read(0xc0e5f8edb0, {0xc0efe16f88?, 0xc1169d3a88?, 0x1804787?})&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/transport/transport.go:483 +0x32&#xA;io.ReadAtLeast({0x55c5720, 0xc0e5f8edb0}, {0xc0efe16f88, 0x5, 0x5}, 0x5)&#xA;&#x9;/usr/local/go/src/io/io.go:331 +0x9a&#xA;io.ReadFull(...)&#xA;&#x9;/usr/local/go/src/io/io.go:350&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/transport.(*Stream).Read(0xc0f3cd67e0, {0xc0efe16f88, 0x5, 0x5})&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/transport/transport.go:467 +0xa5&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.(*parser).recvMsg(0xc0efe16f78, 0x7fffffff)&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/rpc_util.go:559 +0x47&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.recvAndDecompress(0xc1169d3c58?, 0xc0f3cd67e0, {0x0, 0x0}, 0x7fffffff, 0x0, {0x0, 0x0})&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/rpc_util.go:690 +0x66&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.recv(0x172b28f?, {0x7f837c291d58, 0x7f84350}, 0x6f5a274d6e8f284c?, {0x0?, 0x0?}, {0x4be7d40, 0xc0f8c01d50}, 0x0?, 0x0, ...)&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/rpc_util.go:758 +0x6e&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.(*csAttempt).recvMsg(0xc0eb72d800, {0x4be7d40?, 0xc0f8c01d50}, 0x2?)&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/stream.go:970 +0x2b0&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.(*clientStream).RecvMsg.func1(0x4be7d40?)&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/stream.go:821 +0x25&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.(*clientStream).withRetry(0xc0f3cd65a0, 0xc1169d3e78, 0xc1169d3e48)&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/stream.go:675 +0x2f6&#xA;k8s.io/kubernetes/vendor/google.golang.org/grpc.(*clientStream).RecvMsg(0xc0f3cd65a0, {0x4be7d40?, 0xc0f8c01d50?})&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/google.golang.org/grpc/stream.go:820 +0x11f&#xA;k8s.io/kubernetes/vendor/github.com/grpc-ecosystem/go-grpc-prometheus.(*monitoredClientStream).RecvMsg(0xc0efe16f90, {0x4be7d40?, 0xc0f8c01d50?})&#xA;&#x9;/home/prow/go/src/k8s.io/kubernetes/_output/local/go/src/k8s.io/kubernetes/vendor/github.com/grpc-ecosystem/go-grpc-prometheus/client_metrics.go:160</failure>
		</testcase>
	</testsuite>
</testsuites>`

	outputXML := `<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
	<testsuite tests="3" failures="1" time="271.610000" name="k8s.io/kubernetes/test/integration/apiserver" timestamp="">
		<properties>
			<property name="go.version" value="go1.18 linux/amd64"></property>
		</properties>
		<testcase classname="k8s.io/kubernetes/test/integration/apimachinery" name="TestWatchRestartsIfTimeoutNotReached/group/InformerWatcher_survives_closed_watches" time="30.050000"></testcase>
		<testcase classname="k8s.io/kubernetes/test/integration/apiserver" name="TestMaxResourceSize/JSONPatchType_should_handle_a_patch_just_under_the_max_limit" time="0.000000">
			<skipped message="=== RUN   TestMa[...clipped...]x_limit (0.00s)&#xA;"></skipped>
		</testcase>
		<testcase classname="k8s.io/kubernetes/test/integration/apimachinery" name="TestSchedulerInformers" time="-0.000000">
			<failure message="Failed" type="">&#xA;&#x9;/home/prow/go/[...clipped...]t_metrics.go:160</failure>
		</testcase>
	</testsuite>
</testsuites>`
	suites, _ := fetchXML(strings.NewReader(sourceXML))
	pruneXML(suites, 32)
	var output bytes.Buffer
	writer := bufio.NewWriter(&output)
	_ = streamXML(writer, suites)
	_ = writer.Flush()
	assert.Equal(t, outputXML, string(output.Bytes()), "xml was not pruned correctly")
}

func TestPruneTESTS(t *testing.T) {
	sourceXML := `<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
	<testsuite tests="6" failures="0" time="5.50000" name="k8s.io/kubernetes/cluster/gce/cos" timestamp="">
		<properties>
			<property name="go.version" value="go1.18 linux/amd64"></property>
		</properties>
		<testcase classname="k8s.io/kubernetes/cluster/gce/cos" name="TestServerOverride/ETCD-SERVERS_is_not_set_-_default_override" time="0.950000"></testcase>
		<testcase classname="k8s.io/kubernetes/cluster/gce/cos" name="TestServerOverride/ETCD-SERVERS_and_ETCD_SERVERS_OVERRIDES_are_set" time="0.660000"></testcase>
		<testcase classname="k8s.io/kubernetes/cluster/gce/cos" name="TestServerOverride" time="1.610000"></testcase>
		<testcase classname="k8s.io/kubernetes/cluster/gce/cos" name="TestStorageOptions/storage_options_are_supplied" time="0.860000"></testcase>
		<testcase classname="k8s.io/kubernetes/cluster/gce/cos" name="TestStorageOptions/storage_options_are_not_supplied" time="0.280000"></testcase>
		<testcase classname="k8s.io/kubernetes/cluster/gce/cos" name="TestStorageOptions" time="1.140000"></testcase>
	</testsuite>
	<testsuite tests="2" failures="1" time="30.050000" name="k8s.io/kubernetes/test/integration/apimachinery" timestamp="">
		<properties>
			<property name="go.version" value="go1.18 linux/amd64"></property>
		</properties>
		<testcase classname="k8s.io/kubernetes/test/integration/apimachinery" name="TestWatchRestartsIfTimeoutNotReached/group/InformerWatcher_survives_closed_watches" time="30.050000"></testcase>
		<testcase classname="k8s.io/kubernetes/test/integration/apimachinery" name="TestSchedulerInformers" time="-0.000000">
			<failure message="Failed" type="">FailureContent</failure>
		</testcase>
	</testsuite>
</testsuites>`

	outputXML := `<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
	<testsuite tests="6" failures="0" time="5.50000" name="k8s.io/kubernetes/cluster/gce/cos" timestamp="">
		<properties>
			<property name="go.version" value="go1.18 linux/amd64"></property>
		</properties>
		<testcase classname="k8s.io/kubernetes/cluster/gce" name="cos" time="5.50000"></testcase>
	</testsuite>
	<testsuite tests="2" failures="1" time="30.050000" name="k8s.io/kubernetes/test/integration/apimachinery" timestamp="">
		<properties>
			<property name="go.version" value="go1.18 linux/amd64"></property>
		</properties>
		<testcase classname="k8s.io/kubernetes/test/integration" name="apimachinery" time="30.050000">
			<failure message="Failed;" type="">FailureContent;</failure>
		</testcase>
	</testsuite>
</testsuites>`
	suites, _ := fetchXML(strings.NewReader(sourceXML))
	pruneTESTS(suites)
	var output bytes.Buffer
	writer := bufio.NewWriter(&output)
	_ = streamXML(writer, suites)
	_ = writer.Flush()
	assert.Equal(t, outputXML, string(output.Bytes()), "tests in xml was not pruned correctly")
}
