/*
Copyright 2020 The Kubernetes Authors.

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

package framework

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
)

// VolumeResource is a generic implementation of TestResource that wil be able to
// be used in most of TestSuites.
// See volume_io.go or volumes.go in test/e2e/storage/testsuites/ for how to use this resource.
// Also, see subpath.go in the same directory for how to extend and use it.
type VolumeResource struct {
	Config    *PerTestConfig
	Pattern   TestPattern
	VolSource *v1.VolumeSource
	Pvc       *v1.PersistentVolumeClaim
	Pv        *v1.PersistentVolume
	Sc        *storagev1.StorageClass

	Volume TestVolume
}

// CreateVolumeResource constructs a VolumeResource for the current test. It knows how to deal with
// different test pattern volume types.
func CreateVolumeResource(ctx context.Context, driver TestDriver, config *PerTestConfig, pattern TestPattern, testVolumeSizeRange e2evolume.SizeRange) *VolumeResource {
	return CreateVolumeResourceWithAccessModes(ctx, driver, config, pattern, testVolumeSizeRange, driver.GetDriverInfo().RequiredAccessModes, nil)
}

// CreateVolumeResource constructs a VolumeResource for the current test using the specified VAC name.
func CreateVolumeResourceWithVAC(ctx context.Context, driver TestDriver, config *PerTestConfig, pattern TestPattern, testVolumeSizeRange e2evolume.SizeRange, vacName *string) *VolumeResource {
	if pattern.VolType != DynamicPV {
		framework.Failf("Creating volume with VAC only supported on dynamic PV tests")
	}
	return CreateVolumeResourceWithAccessModes(ctx, driver, config, pattern, testVolumeSizeRange, driver.GetDriverInfo().RequiredAccessModes, vacName)
}

// CreateVolumeResourceWithAccessModes constructs a VolumeResource for the current test with the provided access modes.
func CreateVolumeResourceWithAccessModes(ctx context.Context, driver TestDriver, config *PerTestConfig, pattern TestPattern, testVolumeSizeRange e2evolume.SizeRange, accessModes []v1.PersistentVolumeAccessMode, vacName *string) *VolumeResource {
	r := VolumeResource{
		Config:  config,
		Pattern: pattern,
	}
	dInfo := driver.GetDriverInfo()
	f := config.Framework
	cs := f.ClientSet

	// Create volume for pre-provisioned volume tests
	r.Volume = CreateVolume(ctx, driver, config, pattern.VolType)

	switch pattern.VolType {
	case InlineVolume:
		framework.Logf("Creating resource for inline volume")
		if iDriver, ok := driver.(InlineVolumeTestDriver); ok {
			r.VolSource = iDriver.GetVolumeSource(false, pattern.FsType, r.Volume)
		}
	case PreprovisionedPV:
		framework.Logf("Creating resource for pre-provisioned PV")
		if pDriver, ok := driver.(PreprovisionedPVTestDriver); ok {
			pvSource, volumeNodeAffinity := pDriver.GetPersistentVolumeSource(false, pattern.FsType, r.Volume)
			if pvSource != nil {
				r.Pv, r.Pvc = createPVCPV(ctx, f, dInfo.Name, pvSource, volumeNodeAffinity, pattern.VolMode, accessModes)
				r.VolSource = storageutils.CreateVolumeSource(r.Pvc.Name, false /* readOnly */)
			}
		}
	case DynamicPV, GenericEphemeralVolume:
		framework.Logf("Creating resource for dynamic PV")
		if dDriver, ok := driver.(DynamicPVTestDriver); ok {
			var err error
			driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
			claimSize, err := storageutils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
			framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
			framework.Logf("Using claimSize:%s, test suite supported size:%v, driver(%s) supported size:%v ", claimSize, testVolumeSizeRange, dDriver.GetDriverInfo().Name, testVolumeSizeRange)
			r.Sc = dDriver.GetDynamicProvisionStorageClass(ctx, r.Config, pattern.FsType)

			if pattern.BindingMode != "" {
				r.Sc.VolumeBindingMode = &pattern.BindingMode
			}
			r.Sc.AllowVolumeExpansion = &pattern.AllowExpansion

			ginkgo.By("creating a StorageClass " + r.Sc.Name)

			r.Sc, err = cs.StorageV1().StorageClasses().Create(ctx, r.Sc, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			switch pattern.VolType {
			case DynamicPV:
				r.Pv, r.Pvc = createPVCPVFromDynamicProvisionSC(
					ctx, f, dInfo.Name, claimSize, r.Sc, pattern.VolMode, accessModes, vacName)
				r.VolSource = storageutils.CreateVolumeSource(r.Pvc.Name, false /* readOnly */)
			case GenericEphemeralVolume:
				driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
				claimSize, err := storageutils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
				framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
				r.VolSource = createEphemeralVolumeSource(r.Sc.Name, pattern.VolMode, accessModes, claimSize)
			}
		}
	case CSIInlineVolume:
		framework.Logf("Creating resource for CSI ephemeral inline volume")
		if eDriver, ok := driver.(EphemeralTestDriver); ok {
			attributes, _, _ := eDriver.GetVolume(config, 0)
			r.VolSource = &v1.VolumeSource{
				CSI: &v1.CSIVolumeSource{
					Driver:           eDriver.GetCSIDriverName(config),
					VolumeAttributes: attributes,
				},
			}
			if pattern.FsType != "" {
				r.VolSource.CSI.FSType = &pattern.FsType
			}
		}
	default:
		framework.Failf("VolumeResource doesn't support: %s", pattern.VolType)
	}

	if r.VolSource == nil {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
	}

	return &r
}

func createEphemeralVolumeSource(scName string, volMode v1.PersistentVolumeMode, accessModes []v1.PersistentVolumeAccessMode, claimSize string) *v1.VolumeSource {
	if len(accessModes) == 0 {
		accessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	}
	if volMode == "" {
		volMode = v1.PersistentVolumeFilesystem
	}
	return &v1.VolumeSource{
		Ephemeral: &v1.EphemeralVolumeSource{
			VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{
				Spec: v1.PersistentVolumeClaimSpec{
					StorageClassName: &scName,
					AccessModes:      accessModes,
					VolumeMode:       &volMode,
					Resources: v1.VolumeResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceStorage: resource.MustParse(claimSize),
						},
					},
				},
			},
		},
	}
}

// CleanupResource cleans up VolumeResource
func (r *VolumeResource) CleanupResource(ctx context.Context) error {
	f := r.Config.Framework
	var cleanUpErrs []error
	if r.Pvc != nil || r.Pv != nil {
		switch r.Pattern.VolType {
		case PreprovisionedPV:
			ginkgo.By("Deleting pv and pvc")
			if errs := e2epv.PVPVCCleanup(ctx, f.ClientSet, f.Namespace.Name, r.Pv, r.Pvc); len(errs) != 0 {
				framework.Failf("Failed to delete PVC or PV: %v", utilerrors.NewAggregate(errs))
			}
		case DynamicPV:
			ginkgo.By("Deleting pvc")
			// We only delete the PVC so that PV (and disk) can be cleaned up by dynamic provisioner
			if r.Pv != nil && r.Pv.Spec.PersistentVolumeReclaimPolicy != v1.PersistentVolumeReclaimDelete {
				framework.Failf("Test framework does not currently support Dynamically Provisioned Persistent Volume %v specified with reclaim policy that isn't %v",
					r.Pv.Name, v1.PersistentVolumeReclaimDelete)
			}
			if r.Pvc != nil {
				cs := f.ClientSet
				pv := r.Pv
				if pv == nil && r.Pvc.Name != "" {
					// This happens for late binding. Check whether we have a volume now that we need to wait for.
					pvc, err := cs.CoreV1().PersistentVolumeClaims(r.Pvc.Namespace).Get(ctx, r.Pvc.Name, metav1.GetOptions{})
					switch {
					case err == nil:
						if pvc.Spec.VolumeName != "" {
							pv, err = cs.CoreV1().PersistentVolumes().Get(ctx, pvc.Spec.VolumeName, metav1.GetOptions{})
							if err != nil {
								cleanUpErrs = append(cleanUpErrs, fmt.Errorf("failed to find PV %v: %w", pvc.Spec.VolumeName, err))
							}
						}
					case apierrors.IsNotFound(err):
						// Without the PVC, we cannot locate the corresponding PV. Let's
						// hope that it is gone.
					default:
						cleanUpErrs = append(cleanUpErrs, fmt.Errorf("failed to find PVC %v: %w", r.Pvc.Name, err))
					}
				}

				err := e2epv.DeletePersistentVolumeClaim(ctx, f.ClientSet, r.Pvc.Name, f.Namespace.Name)
				if err != nil {
					cleanUpErrs = append(cleanUpErrs, fmt.Errorf("failed to delete PVC %v: %w", r.Pvc.Name, err))
				}

				if pv != nil {
					err = e2epv.WaitForPersistentVolumeDeleted(ctx, f.ClientSet, pv.Name, 5*time.Second, f.Timeouts.PVDeleteSlow)
					if err != nil {
						cleanUpErrs = append(cleanUpErrs, fmt.Errorf(
							"persistent Volume %v not deleted by dynamic provisioner: %w", pv.Name, err))
					}
				}
			}
		default:
			framework.Failf("Found PVC (%v) or PV (%v) but not running Preprovisioned or Dynamic test pattern", r.Pvc, r.Pv)
		}
	}

	if r.Sc != nil {
		ginkgo.By("Deleting sc")
		if err := storageutils.DeleteStorageClass(ctx, f.ClientSet, r.Sc.Name); err != nil {
			cleanUpErrs = append(cleanUpErrs, fmt.Errorf("failed to delete StorageClass %v: %w", r.Sc.Name, err))
		}
	}

	// Cleanup volume for pre-provisioned volume tests
	if r.Volume != nil {
		if err := storageutils.TryFunc(func() {
			r.Volume.DeleteVolume(ctx)
		}); err != nil {
			cleanUpErrs = append(cleanUpErrs, fmt.Errorf("failed to delete Volume: %w", err))
		}
	}
	return utilerrors.NewAggregate(cleanUpErrs)
}

func createPVCPV(
	ctx context.Context,
	f *framework.Framework,
	name string,
	pvSource *v1.PersistentVolumeSource,
	volumeNodeAffinity *v1.VolumeNodeAffinity,
	volMode v1.PersistentVolumeMode,
	accessModes []v1.PersistentVolumeAccessMode,
) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	pvConfig := e2epv.PersistentVolumeConfig{
		NamePrefix:       fmt.Sprintf("%s-", name),
		StorageClassName: f.Namespace.Name,
		PVSource:         *pvSource,
		NodeAffinity:     volumeNodeAffinity,
		AccessModes:      accessModes,
	}

	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		StorageClassName: &f.Namespace.Name,
		AccessModes:      accessModes,
	}

	if volMode != "" {
		pvConfig.VolumeMode = &volMode
		pvcConfig.VolumeMode = &volMode
	}

	framework.Logf("Creating PVC and PV")
	pv, pvc, err := e2epv.CreatePVCPV(ctx, f.ClientSet, f.Timeouts, pvConfig, pvcConfig, f.Namespace.Name, false)
	framework.ExpectNoError(err, "PVC, PV creation failed")

	err = e2epv.WaitOnPVandPVC(ctx, f.ClientSet, f.Timeouts, f.Namespace.Name, pv, pvc)
	framework.ExpectNoError(err, "PVC, PV failed to bind")

	return pv, pvc
}

func createPVCPVFromDynamicProvisionSC(
	ctx context.Context,
	f *framework.Framework,
	name string,
	claimSize string,
	sc *storagev1.StorageClass,
	volMode v1.PersistentVolumeMode,
	accessModes []v1.PersistentVolumeAccessMode,
	vacName *string,
) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	cs := f.ClientSet
	ns := f.Namespace.Name

	ginkgo.By("creating a claim")
	pvcCfg := e2epv.PersistentVolumeClaimConfig{
		NamePrefix:                name,
		ClaimSize:                 claimSize,
		StorageClassName:          &(sc.Name),
		VolumeAttributesClassName: vacName,
		AccessModes:               accessModes,
		VolumeMode:                &volMode,
	}

	pvc := e2epv.MakePersistentVolumeClaim(pvcCfg, ns)

	var err error
	pvc, err = e2epv.CreatePVC(ctx, cs, ns, pvc)
	framework.ExpectNoError(err)

	if !isDelayedBinding(sc) {
		err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, f.Timeouts.ClaimProvision)
		framework.ExpectNoError(err)
	}

	pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	var pv *v1.PersistentVolume
	if !isDelayedBinding(sc) {
		pv, err = cs.CoreV1().PersistentVolumes().Get(ctx, pvc.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
	}

	return pv, pvc
}

func isDelayedBinding(sc *storagev1.StorageClass) bool {
	if sc.VolumeBindingMode != nil {
		return *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer
	}
	return false
}
