package crd_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/crd"
)

func Test_CRD_CSIDriver_Validation(t *testing.T) {
	crd := crd.CSIDriver()

	if crd.Spec.Validation == nil {
		t.Fatal("Expected Driver CRD to have validations configured")
	}

	spec, hasSpec := crd.Spec.Validation.OpenAPIV3Schema.Properties["spec"]
	if !hasSpec {
		t.Fatal("Expected to find a validator for 'spec'")
	}

	attachRequired, hasAttachRequired := spec.Properties["attachRequired"]
	if !hasAttachRequired {
		t.Fatal("Expected to find validator for 'spec.attachRequired'")
	}
	if attachRequired.Type != "boolean" {
		t.Fatalf("Expected 'spec.attachRequired' to have a type validator for boolean, got: %s", attachRequired.Type)
	}

	podInfoOnMountVersion := spec.Properties["podInfoOnMountVersion"]
	if podInfoOnMountVersion.Type != "string" {
		t.Fatalf("Expected 'spec.podInfoOnMountVersion' to have a type validator for string, got: %s", podInfoOnMountVersion.Type)
	}
}

func Test_CRD_CSINodeInfo_Validation(t *testing.T) {
	crd := crd.CSINodeInfo()

	if crd.Spec.Validation == nil {
		t.Fatal("Expected NodeInfo CRD to have validations configured")
	}

	drivers, hasDrivers := crd.Spec.Validation.OpenAPIV3Schema.Properties["csiDrivers"]
	if !hasDrivers {
		t.Fatal("Expected to find validator for 'csiDrivers'")
	}
	if drivers.Type != "array" {
		t.Fatalf("Expected 'csiDrivers' to have a type validator for array, got: %s", drivers.Type)
	}
	if drivers.Items == nil {
		t.Fatal("Expected 'csiDrivers' to specify properties for array items")
	}
	if drivers.Items.Schema == nil {
		t.Fatal("Expected 'csiDrivers' to specify a schema for array items")
	}

	schema := drivers.Items.Schema

	driver := schema.Properties["driver"]
	if driver.Type != "string" {
		t.Fatalf("Expected 'csiDrivers[].driver' to have a type validator for string, got: %s", driver.Type)
	}

	nodeID := schema.Properties["nodeID"]
	if nodeID.Type != "string" {
		t.Fatalf("Expected 'csiDrivers[].nodeID' to have a type validator for string, got: %s", nodeID.Type)
	}

	topologyKeys := schema.Properties["topologyKeys"]
	if topologyKeys.Type != "array" {
		t.Fatalf("Expected 'csiDrivers[].topologyKeys' to have a type validator for array, got: %s", topologyKeys.Type)
	}
	if topologyKeys.Items == nil {
		t.Fatal("Expected 'csiDrivers[].topologyKeys' to specify properties for array items")
	}
	if topologyKeys.Items.Schema == nil {
		t.Fatal("Expected 'csiDrivers[].topologyKeys' to specify a schema for array items")
	}
	topologyKeysSchema := topologyKeys.Items.Schema
	if topologyKeysSchema.Type != "string" {
		t.Fatalf("Expected 'csiDrivers[].topologyKeys[]' to have a type validator for string, got: %s", topologyKeysSchema.Type)
	}
}
