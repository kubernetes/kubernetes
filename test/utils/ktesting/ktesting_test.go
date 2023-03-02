/*
Copyright 2023 The Kubernetes Authors.

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

package ktesting_test

import (
	"bytes"
	"flag"
	"fmt"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	utilsktesting "k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/pointer"
)

func TestNewTestContext(t *testing.T) {
	var buffer logToBuf

	logger, _ := utilsktesting.NewTestContext(&buffer)
	logger.V(6).Info("You don't see me, default verbosity is 5.")
	logger.V(5).Info("You see me.")

	// utilsktesting uses the normal -v command line flag for its verbosity.
	// This cannot be changed at runtime, but we can create a new logger to
	// test this.
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	assert.NoError(t, fs.Set("v", "6"))
	logger.V(6).Info("You don't see me either.")
	logger, _ = utilsktesting.NewTestContext(&buffer)
	logger.V(6).Info("You see me too.")

	logger.Info("Some values",
		"ints", []int{1, 2, 3},
		"strings", []string{"a", "b", "c", `line 1
line2`},
		"pointers", []*int{pointer.Int(1), pointer.Int(2), pointer.Int(3)},
		"pod", v1.Pod{},
	)
	actual := headerRe.ReplaceAllString(buffer.String(), `$1 <header> `)
	actual = hexRe.ReplaceAllString(actual, `<hex>`)
	expected := `I <header> You see me.
I <header> You see me too.
I <header> Some values ints=[1 2 3] strings=<
	[a b c line 1
	line2]
 > pointers=[<hex> <hex> <hex>] pod={TypeMeta:{Kind: APIVersion:} ObjectMeta:{Name: GenerateName: Namespace: SelfLink: UID: ResourceVersion: Generation:0 CreationTimestamp:0001-01-01 00:00:00 +0000 UTC DeletionTimestamp:<nil> DeletionGracePeriodSeconds:<nil> Labels:map[] Annotations:map[] OwnerReferences:[] Finalizers:[] ManagedFields:[]} Spec:{Volumes:[] InitContainers:[] Containers:[] EphemeralContainers:[] RestartPolicy: TerminationGracePeriodSeconds:<nil> ActiveDeadlineSeconds:<nil> DNSPolicy: NodeSelector:map[] ServiceAccountName: DeprecatedServiceAccount: AutomountServiceAccountToken:<nil> NodeName: HostNetwork:false HostPID:false HostIPC:false ShareProcessNamespace:<nil> SecurityContext:nil ImagePullSecrets:[] Hostname: Subdomain: Affinity:nil SchedulerName: Tolerations:[] HostAliases:[] PriorityClassName: Priority:<nil> DNSConfig:nil ReadinessGates:[] RuntimeClassName:<nil> EnableServiceLinks:<nil> PreemptionPolicy:<nil> Overhead:map[] TopologySpreadConstraints:[] SetHostnameAsFQDN:<nil> OS:nil HostUsers:<nil> SchedulingGates:[] ResourceClaims:[]} Status:{Phase: Conditions:[] Message: Reason: NominatedNodeName: HostIP: PodIP: PodIPs:[] StartTime:<nil> InitContainerStatuses:[] ContainerStatuses:[] QOSClass: EphemeralContainerStatuses:[] Resize:}}
`
	assert.Equal(t, expected, actual)
}

var headerRe = regexp.MustCompile(`([IE])[[:digit:]]{4} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}\.[[:digit:]]{6}\] `)
var hexRe = regexp.MustCompile(`0x[A-Fa-f0-9]+`)

type logToBuf struct {
	ktesting.NopTL
	bytes.Buffer
}

func (l *logToBuf) Helper() {
}

func (l *logToBuf) Log(args ...interface{}) {
	for i, arg := range args {
		if i > 0 {
			l.Write([]byte(" "))
		}
		l.Write([]byte(fmt.Sprintf("%s", arg)))
	}
	l.Write([]byte("\n"))
}
