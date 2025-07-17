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

package external

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/scheme"
	klog "k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	"github.com/onsi/gomega"
)

// DriverDefinition needs to be filled in via a .yaml or .json
// file. Its methods then implement the TestDriver interface, using
// nothing but the information in this struct.
type driverDefinition struct {
	// DriverInfo is the static information that the storage testsuite
	// expects from a test driver. See test/e2e/storage/testsuites/testdriver.go
	// for details. The only field with a non-zero default is the list of
	// supported file systems (SupportedFsType): it is set so that tests using
	// the default file system are enabled.
	DriverInfo storageframework.DriverInfo

	// StorageClass must be set to enable dynamic provisioning tests.
	// The default is to not run those tests.
	StorageClass struct {
		// FromName set to true enables the usage of a storage
		// class with DriverInfo.Name as provisioner and no
		// parameters.
		FromName bool

		// FromFile is used only when FromName is false.  It
		// loads a storage class from the given .yaml or .json
		// file. File names are resolved by the
		// framework.testfiles package, which typically means
		// that they can be absolute or relative to the test
		// suite's --repo-root parameter.
		//
		// This can be used when the storage class is meant to have
		// additional parameters.
		FromFile string

		// FromExistingClassName specifies the name of a pre-installed
		// StorageClass that will be copied and used for the tests.
		FromExistingClassName string
	}

	// VolumeAttributesClass must be set to enable volume modification tests.
	// The default is to not run those tests.
	VolumeAttributesClass struct {
		// FromName set to true enables the usage of a
		// VolumeAttributesClass with DriverInfo.Name as
		// provisioner and no parameters.
		FromName bool

		// FromFile is used only when FromName is false.  It
		// loads a storage class from the given .yaml or .json
		// file. File names are resolved by the
		// framework.testfiles package, which typically means
		// that they can be absolute or relative to the test
		// suite's --repo-root parameter.
		//
		// This can be used when the VolumeAttributesClass
		// is meant to have additional parameters.
		FromFile string

		// FromExistingClassName specifies the name of a pre-installed
		// VolumeAttributesClass that will be copied and used for the tests.
		FromExistingClassName string
	}

	// SnapshotClass must be set to enable snapshotting tests.
	// The default is to not run those tests.
	SnapshotClass struct {
		// FromName set to true enables the usage of a
		// snapshotter class with DriverInfo.Name as provisioner.
		FromName bool

		// FromFile is used only when FromName is false.  It
		// loads a snapshot class from the given .yaml or .json
		// file. File names are resolved by the
		// framework.testfiles package, which typically means
		// that they can be absolute or relative to the test
		// suite's --repo-root parameter.
		//
		// This can be used when the snapshot class is meant to have
		// additional parameters.
		FromFile string

		// FromExistingClassName specifies the name of a pre-installed
		// SnapshotClass that will be copied and used for the tests.
		FromExistingClassName string
	}

	// InlineVolumes defines one or more volumes for use as inline
	// ephemeral volumes. At least one such volume has to be
	// defined to enable testing of inline ephemeral volumes.  If
	// a test needs more volumes than defined, some of the defined
	// volumes will be used multiple times.
	//
	// DriverInfo.Name is used as name of the driver in the inline volume.
	InlineVolumes []struct {
		// Attributes are passed as NodePublishVolumeReq.volume_context.
		// Can be empty.
		Attributes map[string]string
		// Shared defines whether the resulting volume is
		// shared between different pods (i.e.  changes made
		// in one pod are visible in another)
		Shared bool
		// ReadOnly must be set to true if the driver does not
		// support mounting as read/write.
		ReadOnly bool
	}

	// ClientNodeName selects a specific node for scheduling test pods.
	// Can be left empty. Most drivers should not need this and instead
	// use topology to ensure that pods land on the right node(s).
	ClientNodeName string

	// NodeSelectors is used to specify nodeSelector information for pod deployment
	// during the tests.  This is beneficial when needing to control placement
	// for specialized environments.  Most drivers should not need this and
	// instead can use topolgy to ensure that pods land on the right node(s).
	NodeSelectors map[string]string

	// Timeouts contains the custom timeouts used during the test execution.
	// The values specified here will override the default values specified in
	// the framework.TimeoutContext struct.
	Timeouts map[string]string
}

func init() {
	e2econfig.Flags.Var(testDriverParameter{}, "storage.testdriver", "name of a .yaml or .json file that defines a driver for storage testing, can be used more than once")
}

// testDriverParameter is used to hook loading of the driver
// definition file and test instantiation into argument parsing: for
// each of potentially many parameters, Set is called and then does
// both immediately. There is no other code location between argument
// parsing and starting of the test suite where those test could be
// defined.
type testDriverParameter struct {
}

var _ flag.Value = testDriverParameter{}

func (t testDriverParameter) String() string {
	return "<.yaml or .json file>"
}

func (t testDriverParameter) Set(filename string) error {
	return AddDriverDefinition(filename)
}

// AddDriverDefinition defines ginkgo tests for CSI driver definition file.
// Either --storage.testdriver cmdline argument or AddDriverDefinition can be used
// to define the tests.
func AddDriverDefinition(filename string) error {
	driver, err := loadDriverDefinition(filename)
	framework.Logf("Driver loaded from path [%s]: %+v", filename, driver)
	if err != nil {
		return err
	}
	if driver.DriverInfo.Name == "" {
		return fmt.Errorf("%q: DriverInfo.Name not set", filename)
	}

	args := []interface{}{"External Storage"}
	args = append(args, storageframework.GetDriverNameWithFeatureTags(driver)...)
	args = append(args, func() {
		storageframework.DefineTestSuites(driver, testsuites.CSISuites)
	})
	framework.Describe(args...)

	return nil
}

func loadDriverDefinition(filename string) (*driverDefinition, error) {
	if filename == "" {
		return nil, fmt.Errorf("missing file name")
	}
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	// Some reasonable defaults follow.
	driver := &driverDefinition{
		DriverInfo: storageframework.DriverInfo{
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
		},
	}
	// TODO: strict checking of the file content once https://github.com/kubernetes/kubernetes/pull/71589
	// or something similar is merged.
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, driver); err != nil {
		return nil, fmt.Errorf("%s: %w", filename, err)
	}

	// To ensure backward compatibility: if controller expansion is enabled,
	// then set both online and offline expansion to true
	if _, ok := driver.GetDriverInfo().Capabilities[storageframework.CapOnlineExpansion]; !ok &&
		driver.GetDriverInfo().Capabilities[storageframework.CapControllerExpansion] {
		caps := driver.DriverInfo.Capabilities
		caps[storageframework.CapOnlineExpansion] = true
		driver.DriverInfo.Capabilities = caps
	}
	if _, ok := driver.GetDriverInfo().Capabilities[storageframework.CapOfflineExpansion]; !ok &&
		driver.GetDriverInfo().Capabilities[storageframework.CapControllerExpansion] {
		caps := driver.DriverInfo.Capabilities
		caps[storageframework.CapOfflineExpansion] = true
		driver.DriverInfo.Capabilities = caps
	}
	return driver, nil
}

var _ storageframework.TestDriver = &driverDefinition{}

// We have to implement the interface because dynamic PV may or may
// not be supported. driverDefinition.SkipUnsupportedTest checks that
// based on the actual driver definition.
var _ storageframework.DynamicPVTestDriver = &driverDefinition{}

// Same for snapshotting.
var _ storageframework.SnapshottableTestDriver = &driverDefinition{}

// And for ephemeral volumes.
var _ storageframework.EphemeralTestDriver = &driverDefinition{}

var _ storageframework.CustomTimeoutsTestDriver = &driverDefinition{}

// runtime.DecodeInto needs a runtime.Object but doesn't do any
// deserialization of it and therefore none of the methods below need
// an implementation.
var _ runtime.Object = &driverDefinition{}

func (d *driverDefinition) DeepCopyObject() runtime.Object {
	return nil
}

func (d *driverDefinition) GetObjectKind() schema.ObjectKind {
	return nil
}

func (d *driverDefinition) GetDriverInfo() *storageframework.DriverInfo {
	return &d.DriverInfo
}

func (d *driverDefinition) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	supported := false
	// TODO (?): add support for more volume types
	switch pattern.VolType {
	case "":
		supported = true
	case storageframework.DynamicPV, storageframework.GenericEphemeralVolume:
		if d.StorageClass.FromName || d.StorageClass.FromFile != "" || d.StorageClass.FromExistingClassName != "" {
			supported = true
		}
	case storageframework.CSIInlineVolume:
		supported = len(d.InlineVolumes) != 0
	}
	if !supported {
		e2eskipper.Skipf("Driver %q does not support volume type %q - skipping", d.DriverInfo.Name, pattern.VolType)
	}

}

func (d *driverDefinition) GetDynamicProvisionStorageClass(ctx context.Context, e2econfig *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	var (
		sc  *storagev1.StorageClass
		err error
	)

	f := e2econfig.Framework
	switch {
	case d.StorageClass.FromName:
		sc = &storagev1.StorageClass{Provisioner: d.DriverInfo.Name}
	case d.StorageClass.FromExistingClassName != "":
		sc, err = f.ClientSet.StorageV1().StorageClasses().Get(ctx, d.StorageClass.FromExistingClassName, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting storage class %s", d.StorageClass.FromExistingClassName)
	case d.StorageClass.FromFile != "":
		var ok bool
		items, err := utils.LoadFromManifests(d.StorageClass.FromFile)
		framework.ExpectNoError(err, "load storage class from %s", d.StorageClass.FromFile)
		gomega.Expect(items).To(gomega.HaveLen(1), "exactly one item from %s", d.StorageClass.FromFile)
		err = utils.PatchItems(f, f.Namespace, items...)
		framework.ExpectNoError(err, "patch items")

		sc, ok = items[0].(*storagev1.StorageClass)
		if !ok {
			framework.Failf("storage class from %s", d.StorageClass.FromFile)
		}
	}

	gomega.Expect(sc).ToNot(gomega.BeNil(), "storage class is unexpectantly nil")

	if fsType != "" {
		if sc.Parameters == nil {
			sc.Parameters = map[string]string{}
		}
		// This limits the external storage test suite to only CSI drivers, which may need to be
		// reconsidered if we eventually need to move in-tree storage tests out.
		sc.Parameters["csi.storage.k8s.io/fstype"] = fsType
	}
	return storageframework.CopyStorageClass(sc, f.Namespace.Name, "e2e-sc")
}

func (d *driverDefinition) GetTimeouts() *framework.TimeoutContext {
	timeouts := framework.NewTimeoutContext()
	if d.Timeouts == nil {
		return timeouts
	}

	// Use a temporary map to hold the timeouts specified in the manifest file
	c := make(map[string]time.Duration)
	for k, v := range d.Timeouts {
		duration, err := time.ParseDuration(v)
		if err != nil {
			// We can't use ExpectNoError() because his method can be called out of an It(),
			// so we simply log the error and return the default timeouts.
			klog.Errorf("Could not parse duration for key %s, will use default values: %v", k, err)
			return timeouts
		}
		c[k] = duration
	}

	// Convert the temporary map holding the custom timeouts to JSON
	t, err := json.Marshal(c)
	if err != nil {
		klog.Errorf("Could not marshal custom timeouts, will use default values: %v", err)
		return timeouts
	}

	// Override the default timeouts with the custom ones
	err = json.Unmarshal(t, &timeouts)
	if err != nil {
		klog.Errorf("Could not unmarshal custom timeouts, will use default values: %v", err)
		return timeouts
	}

	return timeouts
}

func loadSnapshotClass(filename string) (*unstructured.Unstructured, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	snapshotClass := &unstructured.Unstructured{}

	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, snapshotClass); err != nil {
		return nil, fmt.Errorf("%s: %w", filename, err)
	}

	return snapshotClass, nil
}

func (d *driverDefinition) GetSnapshotClass(ctx context.Context, e2econfig *storageframework.PerTestConfig, parameters map[string]string) *unstructured.Unstructured {
	if !d.SnapshotClass.FromName && d.SnapshotClass.FromFile == "" && d.SnapshotClass.FromExistingClassName == "" {
		e2eskipper.Skipf("Driver %q does not support snapshotting - skipping", d.DriverInfo.Name)
	}

	f := e2econfig.Framework
	snapshotter := d.DriverInfo.Name
	ns := e2econfig.Framework.Namespace.Name

	switch {
	case d.SnapshotClass.FromName:
		// Do nothing (just use empty parameters)
	case d.SnapshotClass.FromExistingClassName != "":
		snapshotClass, err := f.DynamicClient.Resource(utils.SnapshotClassGVR).Get(ctx, d.SnapshotClass.FromExistingClassName, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting snapshot class %s", d.SnapshotClass.FromExistingClassName)

		if params, ok := snapshotClass.Object["parameters"].(map[string]interface{}); ok {
			for k, v := range params {
				parameters[k] = v.(string)
			}
		}

		if snapshotProvider, ok := snapshotClass.Object["driver"]; ok {
			snapshotter = snapshotProvider.(string)
		}
	case d.SnapshotClass.FromFile != "":
		snapshotClass, err := loadSnapshotClass(d.SnapshotClass.FromFile)
		framework.ExpectNoError(err, "load snapshot class from %s", d.SnapshotClass.FromFile)

		if params, ok := snapshotClass.Object["parameters"].(map[string]interface{}); ok {
			for k, v := range params {
				parameters[k] = v.(string)
			}
		}

		if snapshotProvider, ok := snapshotClass.Object["driver"]; ok {
			snapshotter = snapshotProvider.(string)
		}
	}

	return utils.GenerateSnapshotClassSpec(snapshotter, parameters, ns)
}

func (d *driverDefinition) GetVolumeAttributesClass(ctx context.Context, e2econfig *storageframework.PerTestConfig) *storagev1.VolumeAttributesClass {
	if !d.VolumeAttributesClass.FromName && d.VolumeAttributesClass.FromFile == "" && d.VolumeAttributesClass.FromExistingClassName == "" {
		e2eskipper.Skipf("Driver %q has no configured VolumeAttributesClass - skipping", d.DriverInfo.Name)
		return nil
	}

	var (
		vac *storagev1.VolumeAttributesClass
		err error
	)

	f := e2econfig.Framework
	switch {
	case d.VolumeAttributesClass.FromName:
		vac = &storagev1.VolumeAttributesClass{DriverName: d.DriverInfo.Name}
	case d.VolumeAttributesClass.FromExistingClassName != "":
		vac, err = f.ClientSet.StorageV1().VolumeAttributesClasses().Get(ctx, d.VolumeAttributesClass.FromExistingClassName, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting VolumeAttributesClass %s", d.VolumeAttributesClass.FromExistingClassName)
	case d.VolumeAttributesClass.FromFile != "":
		var ok bool
		items, err := utils.LoadFromManifests(d.VolumeAttributesClass.FromFile)
		framework.ExpectNoError(err, "load VolumeAttributesClass from %s", d.VolumeAttributesClass.FromFile)
		gomega.Expect(items).To(gomega.HaveLen(1), "exactly one item from %s", d.VolumeAttributesClass.FromFile)
		err = utils.PatchItems(f, f.Namespace, items...)
		framework.ExpectNoError(err, "patch VolumeAttributesClass from %s", d.VolumeAttributesClass.FromFile)

		vac, ok = items[0].(*storagev1.VolumeAttributesClass)
		if !ok {
			framework.Failf("cast VolumeAttributesClass from %s", d.VolumeAttributesClass.FromFile)
		}
	}

	gomega.Expect(vac).ToNot(gomega.BeNil(), "VolumeAttributesClass is unexpectantly nil")

	return storageframework.CopyVolumeAttributesClass(vac, f.Namespace.Name, "e2e-vac")
}

func (d *driverDefinition) GetVolume(e2econfig *storageframework.PerTestConfig, volumeNumber int) (map[string]string, bool, bool) {
	if len(d.InlineVolumes) == 0 {
		e2eskipper.Skipf("%s does not have any InlineVolumeAttributes defined", d.DriverInfo.Name)
	}
	e2evolume := d.InlineVolumes[volumeNumber%len(d.InlineVolumes)]
	return e2evolume.Attributes, e2evolume.Shared, e2evolume.ReadOnly
}

func (d *driverDefinition) GetCSIDriverName(e2econfig *storageframework.PerTestConfig) string {
	return d.DriverInfo.Name
}

func (d *driverDefinition) PrepareTest(ctx context.Context, f *framework.Framework) *storageframework.PerTestConfig {
	e2econfig := &storageframework.PerTestConfig{
		Driver:              d,
		Prefix:              "external",
		Framework:           f,
		ClientNodeSelection: e2epod.NodeSelection{Name: d.ClientNodeName},
	}

	if framework.NodeOSDistroIs("windows") {
		e2econfig.ClientNodeSelection.Selector = map[string]string{"kubernetes.io/os": "windows"}
	} else {
		e2econfig.ClientNodeSelection.Selector = map[string]string{"kubernetes.io/os": "linux"}
	}

	// Add all provided nodeSelector settings
	for key, value := range d.NodeSelectors {
		e2econfig.ClientNodeSelection.Selector[key] = value
	}

	return e2econfig
}
