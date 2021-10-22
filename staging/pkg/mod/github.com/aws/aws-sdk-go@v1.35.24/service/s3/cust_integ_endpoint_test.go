// +build integration

package s3_test

import (
	"testing"
)

func TestInteg_AccessPoint_WriteToObject(t *testing.T) {
	testWriteToObject(t, integMetadata.AccessPoints.Source.ARN)
}

func TestInteg_AccessPoint_PresignedGetPut(t *testing.T) {
	testPresignedGetPut(t, integMetadata.AccessPoints.Source.ARN)
}

func TestInteg_AccessPoint_CopyObject(t *testing.T) {
	t.Skip("API does not support access point")
	testCopyObject(t,
		integMetadata.AccessPoints.Source.ARN,
		integMetadata.AccessPoints.Target.ARN)
}
