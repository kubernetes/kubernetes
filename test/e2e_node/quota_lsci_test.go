/*
Copyright 2019 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/volume/util/fsquota"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/mount-utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	LSCIQuotaFeature = features.LocalStorageCapacityIsolationFSQuotaMonitoring
)

func runOneQuotaTest(f *framework.Framework, quotasRequested bool) {
	evictionTestTimeout := 10 * time.Minute
	sizeLimit := resource.MustParse("100Mi")
	useOverLimit := 101 /* Mb */
	useUnderLimit := 99 /* Mb */
	// TODO: remove hardcoded kubelet volume directory path
	// framework.TestContext.KubeVolumeDir is currently not populated for node e2e
	// As for why we do this: see comment below at isXfs.
	if isXfs("/var/lib/kubelet") {
		useUnderLimit = 50 /* Mb */
	}
	priority := 0
	if quotasRequested {
		priority = 1
	}
	ginkgo.Context(fmt.Sprintf(testContextFmt, fmt.Sprintf("use quotas for LSCI monitoring (quotas enabled: %v)", quotasRequested)), func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			defer withFeatureGate(LSCIQuotaFeature, quotasRequested)()
			// TODO: remove hardcoded kubelet volume directory path
			// framework.TestContext.KubeVolumeDir is currently not populated for node e2e
			if quotasRequested && !supportsQuotas("/var/lib/kubelet") {
				// No point in running this as a positive test if quotas are not
				// enabled on the underlying filesystem.
				e2eskipper.Skipf("Cannot run LocalStorageCapacityIsolationFSQuotaMonitoring on filesystem without project quota enabled")
			}
			// setting a threshold to 0% disables; non-empty map overrides default value (necessary due to omitempty)
			initialConfig.EvictionHard = map[string]string{"memory.available": "0%"}

			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates[string(LSCIQuotaFeature)] = quotasRequested
		})
		runEvictionTest(f, evictionTestTimeout, noPressure, noStarvedResource, logDiskMetrics, []podEvictSpec{
			{
				evictionPriority: priority, // This pod should be evicted because of emptyDir violation only if quotas are enabled
				pod: diskConcealingPod(fmt.Sprintf("emptydir-concealed-disk-over-sizelimit-quotas-%v", quotasRequested), useOverLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: &sizeLimit},
				}, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0, // This pod should not be evicted because it uses less than its limit (test for quotas)
				pod: diskConcealingPod(fmt.Sprintf("emptydir-concealed-disk-under-sizelimit-quotas-%v", quotasRequested), useUnderLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: &sizeLimit},
				}, v1.ResourceRequirements{}),
			},
		})
	})
}

// LocalStorageCapacityIsolationFSQuotaMonitoring tests that quotas are
// used for monitoring rather than du.  The mechanism is to create a
// pod that creates a file, deletes it, and writes data to it.  If
// quotas are used to monitor, it will detect this deleted-but-in-use
// file; if du is used to monitor, it will not detect this.
var _ = SIGDescribe("LocalStorageCapacityIsolationFSQuotaMonitoring", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), feature.LocalStorageCapacityIsolationQuota, nodefeature.LSCIQuotaMonitoring, func() {
	f := framework.NewDefaultFramework("localstorage-quota-monitoring-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	runOneQuotaTest(f, true)
	runOneQuotaTest(f, false)
	addAfterEachForCleaningUpPods(f)
})

const (
	writeConcealedPodCommand = `
my $file = "%s.bin";
open OUT, ">$file" || die "Cannot open $file: $!\n";
unlink "$file" || die "Cannot unlink $file: $!\n";
my $a = "a";
foreach (1..20) { $a = "$a$a"; }
foreach (1..%d) { syswrite(OUT, $a); }
sleep 999999;`
)

// This is needed for testing eviction of pods using disk space in concealed files; the shell has no convenient
// way of performing I/O to a concealed file, and the busybox image doesn't contain Perl.
func diskConcealingPod(name string, diskConsumedMB int, volumeSource *v1.VolumeSource, resources v1.ResourceRequirements) *v1.Pod {
	path := ""
	volumeMounts := []v1.VolumeMount{}
	volumes := []v1.Volume{}
	if volumeSource != nil {
		path = volumeMountPath
		volumeMounts = []v1.VolumeMount{{MountPath: volumeMountPath, Name: volumeName}}
		volumes = []v1.Volume{{Name: volumeName, VolumeSource: *volumeSource}}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%s-pod", name)},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image: imageutils.GetE2EImage(imageutils.Perl),
					Name:  fmt.Sprintf("%s-container", name),
					Command: []string{
						"perl",
						"-e",
						fmt.Sprintf(writeConcealedPodCommand, filepath.Join(path, "file"), diskConsumedMB),
					},
					Resources:    resources,
					VolumeMounts: volumeMounts,
				},
			},
			Volumes: volumes,
		},
	}
}

// Don't bother returning an error; if something goes wrong,
// simply treat it as "no".
func supportsQuotas(dir string) bool {
	supportsQuota, err := fsquota.SupportsQuotas(mount.New(""), dir)
	return supportsQuota && err == nil
}
