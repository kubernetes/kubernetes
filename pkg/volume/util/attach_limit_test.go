/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"crypto/sha1"
	"encoding/hex"
	"testing"

	"k8s.io/api/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

func TestGetCSIAttachLimitKey(t *testing.T) {
	// When driverName is less than 39 characters
	csiLimitKey := GetCSIAttachLimitKey("com.amazon.ebs")
	if csiLimitKey != "attachable-volumes-csi-com.amazon.ebs" {
		t.Errorf("Expected com.amazon.ebs got %s", csiLimitKey)
	}

	// When driver is longer than 39 chars
	longDriverName := "com.amazon.kubernetes.eks.ec2.ebs/csi-driver"
	csiLimitKeyLonger := GetCSIAttachLimitKey(longDriverName)
	if !v1helper.IsAttachableVolumeResourceName(v1.ResourceName(csiLimitKeyLonger)) {
		t.Errorf("Expected %s to have attachable prefix", csiLimitKeyLonger)
	}

	expectedCSIKey := getDriverHash(longDriverName)
	if csiLimitKeyLonger != expectedCSIKey {
		t.Errorf("Expected limit to be %s got %s", expectedCSIKey, csiLimitKeyLonger)
	}
}

func getDriverHash(driverName string) string {
	charsFromDriverName := driverName[:23]
	hash := sha1.New()
	hash.Write([]byte(driverName))
	hashed := hex.EncodeToString(hash.Sum(nil))
	hashed = hashed[:16]
	return CSIAttachLimitPrefix + charsFromDriverName + hashed
}
