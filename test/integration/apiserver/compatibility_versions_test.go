/*
Copyright 2024 The Kubernetes Authors.

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

package apiserver

import (
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestFeatureGateCompatibilityEmulationVersion(t *testing.T) {
	type featureGateInfo struct {
		stage             string
		lockToDefaultTrue bool
		enabled           int
	}
	tcs := []struct {
		emulatedVersion                string
		expectedFeatureGatesForVersion map[string]featureGateInfo
	}{
		// TODO(#128193): Update this test cases with each new k8s version so that n-1..3 are the correct versions (add or remove them as the k8s version evolves)
		{
			emulatedVersion: "1.31",
			expectedFeatureGatesForVersion: map[string]featureGateInfo{
				"APIListChunking":                 {"", true, 1},
				"APIResponseCompression":          {"BETA", false, 1},
				"APIServerIdentity":               {"BETA", false, 1},
				"APIServerTracing":                {"BETA", false, 1},
				"APIServingWithRoutine":           {"ALPHA", false, 0},
				"AdmissionWebhookMatchConditions": {"", true, 1},
				"AggregatedDiscoveryEndpoint":     {"", true, 1},
				"AllAlpha":                        {"ALPHA", false, 0},
				"AllBeta":                         {"BETA", false, 0},
				"AllowDNSOnlyNodeCSR":             {"DEPRECATED", false, 0},
				"AllowInsecureKubeletCertificateSigningRequests": {"DEPRECATED", false, 0},
				"AllowServiceLBStatusOnNonLB":                    {"DEPRECATED", false, 0},
				"AnonymousAuthConfigurableEndpoints":             {"ALPHA", false, 0},
				"AnyVolumeDataSource":                            {"BETA", false, 1},
				"AppArmor":                                       {"", true, 1},
				"AppArmorFields":                                 {"", true, 1},
				"AuthorizeNodeWithSelectors":                     {"ALPHA", false, 0},
				"AuthorizeWithSelectors":                         {"ALPHA", false, 0},
				"CPUManager":                                     {"", true, 1},
				"CPUManagerPolicyAlphaOptions":                   {"ALPHA", false, 0},
				"CPUManagerPolicyBetaOptions":                    {"BETA", false, 1},
				"CPUManagerPolicyOptions":                        {"BETA", false, 1},
				"CRDValidationRatcheting":                        {"BETA", false, 1},
				"CSIMigrationPortworx":                           {"BETA", false, 1},
				"CSIVolumeHealth":                                {"ALPHA", false, 0},
				"CloudControllerManagerWebhook":                  {"ALPHA", false, 0},
				"CloudDualStackNodeIPs":                          {"", true, 1},
				"ClusterTrustBundle":                             {"ALPHA", false, 0},
				"ClusterTrustBundleProjection":                   {"ALPHA", false, 0},
				"ComponentSLIs":                                  {"BETA", false, 1},
				"ConcurrentWatchObjectDecode":                    {"BETA", false, 0},
				"ConsistentListFromCache":                        {"BETA", false, 1},
				"ContainerCheckpoint":                            {"BETA", false, 1},
				"ContextualLogging":                              {"BETA", false, 1},
				"CoordinatedLeaderElection":                      {"ALPHA", false, 0},
				"CronJobsScheduledAnnotation":                    {"BETA", false, 1},
				"CrossNamespaceVolumeDataSource":                 {"ALPHA", false, 0},
				"CustomCPUCFSQuotaPeriod":                        {"ALPHA", false, 0},
				"CustomResourceFieldSelectors":                   {"BETA", false, 1},
				"DRAControlPlaneController":                      {"ALPHA", false, 0},
				"DevicePluginCDIDevices":                         {"", true, 1},
				"DisableAllocatorDualWrite":                      {"ALPHA", false, 0},
				"DisableCloudProviders":                          {"", true, 1},
				"DisableKubeletCloudCredentialProviders":         {"", true, 1},
				"DisableNodeKubeProxyVersion":                    {"DEPRECATED", false, 0},
				"DynamicResourceAllocation":                      {"ALPHA", false, 0},
				"EfficientWatchResumption":                       {"", true, 1},
				"ElasticIndexedJob":                              {"", true, 1},
				"EventedPLEG":                                    {"ALPHA", false, 0},
				"ExecProbeTimeout":                               {"", false, 1},
				"GracefulNodeShutdown":                           {"BETA", false, 1},
				"GracefulNodeShutdownBasedOnPodPriority":         {"BETA", false, 1},
				"HPAContainerMetrics":                            {"", true, 1},
				"HPAScaleToZero":                                 {"ALPHA", false, 0},
				"HonorPVReclaimPolicy":                           {"BETA", false, 1},
				"ImageMaximumGCAge":                              {"BETA", false, 1},
				"ImageVolume":                                    {"ALPHA", false, 0},
				"InPlacePodVerticalScaling":                      {"ALPHA", false, 0},
				"InTreePluginPortworxUnregister":                 {"ALPHA", false, 0},
				"InformerResourceVersion":                        {"ALPHA", false, 0},
				"JobBackoffLimitPerIndex":                        {"BETA", false, 1},
				"JobManagedBy":                                   {"ALPHA", false, 0},
				"JobPodFailurePolicy":                            {"", true, 1},
				"JobPodReplacementPolicy":                        {"BETA", false, 1},
				"JobSuccessPolicy":                               {"BETA", false, 1},
				"KMSv1":                                          {"DEPRECATED", false, 0},
				"KMSv2":                                          {"", true, 1},
				"KMSv2KDF":                                       {"", true, 1},
				"KubeProxyDrainingTerminatingNodes":              {"", true, 1},
				"KubeletCgroupDriverFromCRI":                     {"BETA", false, 1},
				"KubeletInUserNamespace":                         {"ALPHA", false, 0},
				"KubeletPodResourcesDynamicResources":            {"ALPHA", false, 0},
				"KubeletPodResourcesGet":                         {"ALPHA", false, 0},
				"KubeletSeparateDiskGC":                          {"BETA", false, 1},
				"KubeletTracing":                                 {"BETA", false, 1},
				"LegacyServiceAccountTokenCleanUp":               {"", true, 1},
				"LoadBalancerIPMode":                             {"BETA", false, 1},
				"LocalStorageCapacityIsolationFSQuotaMonitoring": {"BETA", false, 0},
				"LogarithmicScaleDown":                           {"", true, 1},
				"LoggingAlphaOptions":                            {"ALPHA", false, 0},
				"LoggingBetaOptions":                             {"BETA", false, 1},
				"MatchLabelKeysInPodAffinity":                    {"BETA", false, 1},
				"MatchLabelKeysInPodTopologySpread":              {"BETA", false, 1},
				"MaxUnavailableStatefulSet":                      {"ALPHA", false, 0},
				"MemoryManager":                                  {"BETA", false, 1},
				"MemoryQoS":                                      {"ALPHA", false, 0},
				"MinDomainsInPodTopologySpread":                  {"", true, 1},
				"MultiCIDRServiceAllocator":                      {"BETA", false, 0},
				"MutatingAdmissionPolicy":                        {"ALPHA", false, 0},
				"NFTablesProxyMode":                              {"BETA", false, 1},
				"NewVolumeManagerReconstruction":                 {"", true, 1},
				"NodeInclusionPolicyInPodTopologySpread":         {"BETA", false, 1},
				"NodeLogQuery":                                   {"BETA", false, 0},
				"NodeOutOfServiceVolumeDetach":                   {"", true, 1},
				"NodeSwap":                                       {"BETA", false, 1},
				"OpenAPIEnums":                                   {"BETA", false, 1},
				"PDBUnhealthyPodEvictionPolicy":                  {"", true, 1},
				"PersistentVolumeLastPhaseTransitionTime":        {"", true, 1},
				"PodAndContainerStatsFromCRI":                    {"ALPHA", false, 0},
				"PodDeletionCost":                                {"BETA", false, 1},
				"PodDisruptionConditions":                        {"", true, 1},
				"PodHostIPs":                                     {"", true, 1},
				"PodIndexLabel":                                  {"BETA", false, 1},
				"PodLifecycleSleepAction":                        {"BETA", false, 1},
				"PodReadyToStartContainersCondition":             {"BETA", false, 1},
				"PodSchedulingReadiness":                         {"", true, 1},
				"PortForwardWebsockets":                          {"BETA", false, 1},
				"ProcMountType":                                  {"BETA", false, 0},
				"QOSReserved":                                    {"ALPHA", false, 0},
				"RecoverVolumeExpansionFailure":                  {"ALPHA", false, 0},
				"RecursiveReadOnlyMounts":                        {"BETA", false, 1},
				"RelaxedEnvironmentVariableValidation":           {"ALPHA", false, 0},
				"ReloadKubeletServerCertificateFile":             {"BETA", false, 1},
				"RemainingItemCount":                             {"", true, 1},
				"ResilientWatchCacheInitialization":              {"BETA", false, 1},
				"ResourceHealthStatus":                           {"ALPHA", false, 0},
				"RetryGenerateName":                              {"BETA", false, 1},
				"RotateKubeletServerCertificate":                 {"BETA", false, 1},
				"RuntimeClassInImageCriApi":                      {"ALPHA", false, 0},
				"SELinuxMount":                                   {"ALPHA", false, 0},
				"SELinuxMountReadWriteOncePod":                   {"BETA", false, 1},
				"SchedulerQueueingHints":                         {"BETA", false, 0},
				"SeparateCacheWatchRPC":                          {"BETA", false, 1},
				"SeparateTaintEvictionController":                {"BETA", false, 1},
				"ServerSideApply":                                {"", true, 1},
				"ServerSideFieldValidation":                      {"", true, 1},
				"ServiceAccountTokenJTI":                         {"BETA", false, 1},
				"ServiceAccountTokenNodeBinding":                 {"BETA", false, 1},
				"ServiceAccountTokenNodeBindingValidation":       {"BETA", false, 1},
				"ServiceAccountTokenPodNodeInfo":                 {"BETA", false, 1},
				"ServiceTrafficDistribution":                     {"BETA", false, 1},
				"SidecarContainers":                              {"BETA", false, 1},
				"SizeMemoryBackedVolumes":                        {"BETA", false, 1},
				"StableLoadBalancerNodeSet":                      {"", true, 1},
				"StatefulSetAutoDeletePVC":                       {"BETA", false, 1},
				"StatefulSetStartOrdinal":                        {"", true, 1},
				"StorageNamespaceIndex":                          {"BETA", false, 1},
				"StorageVersionAPI":                              {"ALPHA", false, 0},
				"StorageVersionHash":                             {"BETA", false, 1},
				"StorageVersionMigrator":                         {"ALPHA", false, 0},
				"StrictCostEnforcementForVAP":                    {"BETA", false, 0},
				"StrictCostEnforcementForWebhooks":               {"BETA", false, 0},
				"StructuredAuthenticationConfiguration":          {"BETA", false, 1},
				"StructuredAuthorizationConfiguration":           {"BETA", false, 1},
				"SupplementalGroupsPolicy":                       {"ALPHA", false, 0},
				"TopologyAwareHints":                             {"BETA", false, 1},
				"TopologyManagerPolicyAlphaOptions":              {"ALPHA", false, 0},
				"TopologyManagerPolicyBetaOptions":               {"BETA", false, 1},
				"TopologyManagerPolicyOptions":                   {"BETA", false, 1},
				"TranslateStreamCloseWebsocketRequests":          {"BETA", false, 1},
				"UnauthenticatedHTTP2DOSMitigation":              {"BETA", false, 1},
				"UnknownVersionInteroperabilityProxy":            {"ALPHA", false, 0},
				"UserNamespacesPodSecurityStandards":             {"ALPHA", false, 0},
				"UserNamespacesSupport":                          {"BETA", false, 0},
				"ValidatingAdmissionPolicy":                      {"", true, 1},
				"VolumeAttributesClass":                          {"BETA", false, 0},
				"VolumeCapacityPriority":                         {"ALPHA", false, 0},
				"WatchBookmark":                                  {"", true, 1},
				"WatchCacheInitializationPostStartHook":          {"BETA", false, 0},
				"WatchFromStorageWithoutResourceVersion":         {"BETA", false, 0},
				"WatchList":                                      {"ALPHA", false, 0},
				"WatchListClient":                                {"BETA", false, 0},
				"WinDSR":                                         {"ALPHA", false, 0},
				"WinOverlay":                                     {"BETA", false, 1},
				"WindowsHostNetwork":                             {"ALPHA", false, 0},
				"ZeroLimitedNominalConcurrencyShares":            {"", true, 1},
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.emulatedVersion, func(t *testing.T) {
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tc.emulatedVersion))
			server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
			defer server.TearDownFn()

			rt, err := restclient.TransportFor(server.ClientConfig)
			if err != nil {
				t.Fatal(err)
			}

			req, err := http.NewRequest(http.MethodGet, server.ClientConfig.Host+"/metrics", nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := rt.RoundTrip(req)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				_ = resp.Body.Close()
			}()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			actualFeatureGates, err := parseFeatureGates(string(body))
			if err != nil {
				t.Fatal(err)
			}

			for expectedFeatureName, expectedFeatureInfo := range tc.expectedFeatureGatesForVersion {
				actualValue, exists := actualFeatureGates[expectedFeatureName]
				if !exists && !expectedFeatureInfo.lockToDefaultTrue && expectedFeatureInfo.stage != "ALPHA" {
					t.Errorf("expected feature gate %s not found in /metrics output", expectedFeatureName)
					continue
				}
				if exists && expectedFeatureInfo.stage != "ALPHA" && actualValue != expectedFeatureInfo.enabled {
					t.Errorf("feature %s expected value %d, got %d", expectedFeatureName, expectedFeatureInfo.enabled, actualValue)
				}
			}

			for featureName := range actualFeatureGates {
				prevFeatureGateInfo, exists := tc.expectedFeatureGatesForVersion[featureName]
				if !exists && prevFeatureGateInfo.enabled == 1 {
					// new feature gates can exist in emulated version n+1..3 but must be zero/off by default
					t.Errorf("unexpected feature %s found in /metrics output", featureName)
					continue
				}
			}
		})
	}
}

func parseFeatureGates(metricsBody string) (map[string]int, error) {
	featureGates := make(map[string]int)
	re := regexp.MustCompile(`(?m)^kubernetes_feature_enabled{name="([^"]+)",stage="[^"]*"} (\d+)`)
	matches := re.FindAllStringSubmatch(metricsBody, -1) // Find all matches

	for _, match := range matches {
		if len(match) != 3 { // Each match should have the full match and two capture groups
			return nil, fmt.Errorf("parsing feature gates where match did not have expected length of 3: %s", match)
		}
		value, err := strconv.Atoi(match[2])
		if err != nil {
			return nil, fmt.Errorf("unable to convert value %s to integer for feature gate %s", match[2], match[1])
		}
		featureGates[match[1]] = value
	}
	return featureGates, nil
}
