// +build integration

package s3_test

import (
	"testing"
)

func TestInteg_WriteToObject(t *testing.T) {
	testWriteToObject(t, integMetadata.Buckets.Source.Name)
}

func TestInteg_PresignedGetPut(t *testing.T) {
	testPresignedGetPut(t, integMetadata.Buckets.Source.Name)
}

func TestInteg_CopyObject(t *testing.T) {
	testCopyObject(t, integMetadata.Buckets.Source.Name, integMetadata.Buckets.Target.Name)
}
