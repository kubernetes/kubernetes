/*
Copyright 2017 The Kubernetes Authors.

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

package config

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestKubeletConfigurationPathFields(t *testing.T) {
	// ensure the intersection of kubeletConfigurationPathFieldPaths and KubeletConfigurationNonPathFields is empty
	if i := kubeletConfigurationPathFieldPaths.Intersection(kubeletConfigurationNonPathFieldPaths); len(i) > 0 {
		t.Fatalf("expect the intersection of kubeletConfigurationPathFieldPaths and "+
			"KubeletConfigurationNonPathFields to be empty, got:\n%s",
			strings.Join(i.List(), "\n"))
	}

	// ensure that kubeletConfigurationPathFields U kubeletConfigurationNonPathFields == allPrimitiveFieldPaths(KubeletConfiguration)
	expect := sets.NewString().Union(kubeletConfigurationPathFieldPaths).Union(kubeletConfigurationNonPathFieldPaths)
	result := allPrimitiveFieldPaths(t, reflect.TypeOf(&KubeletConfiguration{}), nil)
	if !expect.Equal(result) {
		// expected fields missing from result
		missing := expect.Difference(result)
		// unexpected fields in result but not specified in expect
		unexpected := result.Difference(expect)
		if len(missing) > 0 {
			t.Errorf("the following fields were expected, but missing from the result. "+
				"If the field has been removed, please remove it from the kubeletConfigurationPathFieldPaths set "+
				"and the KubeletConfigurationPathRefs function, "+
				"or remove it from the kubeletConfigurationNonPathFieldPaths set, as appropriate:\n%s",
				strings.Join(missing.List(), "\n"))
		}
		if len(unexpected) > 0 {
			t.Errorf("the following fields were in the result, but unexpected. "+
				"If the field is new, please add it to the kubeletConfigurationPathFieldPaths set "+
				"and the KubeletConfigurationPathRefs function, "+
				"or add it to the kubeletConfigurationNonPathFieldPaths set, as appropriate:\n%s",
				strings.Join(unexpected.List(), "\n"))
		}
	}
}

func allPrimitiveFieldPaths(t *testing.T, tp reflect.Type, path *field.Path) sets.String {
	paths := sets.NewString()
	switch tp.Kind() {
	case reflect.Ptr:
		paths.Insert(allPrimitiveFieldPaths(t, tp.Elem(), path).List()...)
	case reflect.Struct:
		for i := 0; i < tp.NumField(); i++ {
			field := tp.Field(i)
			paths.Insert(allPrimitiveFieldPaths(t, field.Type, path.Child(field.Name)).List()...)
		}
	case reflect.Map, reflect.Slice:
		paths.Insert(allPrimitiveFieldPaths(t, tp.Elem(), path.Key("*")).List()...)
	case reflect.Interface:
		t.Fatalf("unexpected interface{} field %s", path.String())
	default:
		// if we hit a primitive type, we're at a leaf
		paths.Insert(path.String())
	}
	return paths
}

// dummy helper types
type foo struct {
	foo int
}
type bar struct {
	str    string
	strptr *string

	ints      []int
	stringMap map[string]string

	foo    foo
	fooptr *foo

	bars   []foo
	barMap map[string]foo
}

func TestAllPrimitiveFieldPaths(t *testing.T) {
	expect := sets.NewString(
		"str",
		"strptr",
		"ints[*]",
		"stringMap[*]",
		"foo.foo",
		"fooptr.foo",
		"bars[*].foo",
		"barMap[*].foo",
	)
	result := allPrimitiveFieldPaths(t, reflect.TypeOf(&bar{}), nil)
	if !expect.Equal(result) {
		// expected fields missing from result
		missing := expect.Difference(result)

		// unexpected fields in result but not specified in expect
		unexpected := result.Difference(expect)

		if len(missing) > 0 {
			t.Errorf("the following fields were exepcted, but missing from the result:\n%s", strings.Join(missing.List(), "\n"))
		}
		if len(unexpected) > 0 {
			t.Errorf("the following fields were in the result, but unexpected:\n%s", strings.Join(unexpected.List(), "\n"))
		}
	}
}

var (
	// KubeletConfiguration fields that contain file paths. If you update this, also update KubeletConfigurationPathRefs!
	kubeletConfigurationPathFieldPaths = sets.NewString(
		"StaticPodPath",
		"Authentication.X509.ClientCAFile",
		"TLSCertFile",
		"TLSPrivateKeyFile",
		"ResolverConfig",
	)

	// KubeletConfiguration fields that do not contain file paths.
	kubeletConfigurationNonPathFieldPaths = sets.NewString(
		"Address",
		"Authentication.Anonymous.Enabled",
		"Authentication.Webhook.CacheTTL.Duration",
		"Authentication.Webhook.Enabled",
		"Authorization.Mode",
		"Authorization.Webhook.CacheAuthorizedTTL.Duration",
		"Authorization.Webhook.CacheUnauthorizedTTL.Duration",
		"CPUCFSQuota",
		"CPUCFSQuotaPeriod.Duration",
		"CPUManagerPolicy",
		"CPUManagerReconcilePeriod.Duration",
		"QOSReserved[*]",
		"CgroupDriver",
		"CgroupRoot",
		"CgroupsPerQOS",
		"ClusterDNS[*]",
		"ClusterDomain",
		"ConfigMapAndSecretChangeDetectionStrategy",
		"ContainerLogMaxFiles",
		"ContainerLogMaxSize",
		"ContentType",
		"EnableContentionProfiling",
		"EnableControllerAttachDetach",
		"EnableDebuggingHandlers",
		"EnforceNodeAllocatable[*]",
		"EventBurst",
		"EventRecordQPS",
		"EvictionHard[*]",
		"EvictionMaxPodGracePeriod",
		"EvictionMinimumReclaim[*]",
		"EvictionPressureTransitionPeriod.Duration",
		"EvictionSoft[*]",
		"EvictionSoftGracePeriod[*]",
		"FailSwapOn",
		"FeatureGates[*]",
		"FileCheckFrequency.Duration",
		"HTTPCheckFrequency.Duration",
		"HairpinMode",
		"HealthzBindAddress",
		"HealthzPort",
		"TLSCipherSuites[*]",
		"TLSMinVersion",
		"IPTablesDropBit",
		"IPTablesMasqueradeBit",
		"ImageGCHighThresholdPercent",
		"ImageGCLowThresholdPercent",
		"ImageMinimumGCAge.Duration",
		"KubeAPIBurst",
		"KubeAPIQPS",
		"KubeReservedCgroup",
		"KubeReserved[*]",
		"KubeletCgroups",
		"MakeIPTablesUtilChains",
		"RotateCertificates",
		"ServerTLSBootstrap",
		"StaticPodURL",
		"StaticPodURLHeader[*][*]",
		"MaxOpenFiles",
		"MaxPods",
		"NodeStatusUpdateFrequency.Duration",
		"NodeStatusReportFrequency.Duration",
		"NodeLeaseDurationSeconds",
		"OOMScoreAdj",
		"PodCIDR",
		"PodPidsLimit",
		"PodsPerCore",
		"Port",
		"ProtectKernelDefaults",
		"ReadOnlyPort",
		"RegistryBurst",
		"RegistryPullQPS",
		"RuntimeRequestTimeout.Duration",
		"SerializeImagePulls",
		"StreamingConnectionIdleTimeout.Duration",
		"SyncFrequency.Duration",
		"SystemCgroups",
		"SystemReservedCgroup",
		"SystemReserved[*]",
		"TypeMeta.APIVersion",
		"TypeMeta.Kind",
		"VolumeStatsAggPeriod.Duration",
	)
)
