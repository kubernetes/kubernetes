/*
Copyright The Kubernetes Authors.

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

// Package devicemetadata provides consumer-side utilities for reading and decoding
// DRA device metadata files. It is intended for any Go program that needs
// to consume metadata written by a DRA kubelet plugin, including sidecars,
// monitoring tools, and in-container agents.
package devicemetadata

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/dynamic-resource-allocation/api/metadata"
	"k8s.io/dynamic-resource-allocation/api/metadata/install"
)

var (
	scheme       = install.NewScheme()
	codecFactory = runtimeserializer.NewCodecFactory(scheme)
)

// DecodeMetadataFromStream reads the first compatible object from a DRA
// device metadata stream. A metadata file may contain the same data encoded
// in multiple API versions (see k8s.io/dynamic-resource-allocation/api/metadata
// for the file format); this function tries each in order and populates dest
// with the first one it can successfully decode and convert.
//
// Objects that cannot be decoded or converted are skipped so that a driver
// upgrade does not break older consumers.
//
// dest must be a pointer to a type registered in the metadata scheme (e.g.,
// *v1alpha1.DeviceMetadata or *metadata.DeviceMetadata). The internal type
// is recommended because it can be decoded with less conversions and source
// code does not need to be updated when new API versions get added or (at
// some point) removed.
func DecodeMetadataFromStream(decoder *json.Decoder, dest runtime.Object) error {
	gvks, _, err := scheme.ObjectKinds(dest)
	if err != nil {
		return fmt.Errorf("determine target type: %w", err)
	}

	deserializer := codecFactory.UniversalDeserializer()

	var skippedErrors []string
	for decoder.More() {
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return fmt.Errorf("read metadata object from stream: %w", err)
		}

		obj, gvk, err := deserializer.Decode(raw, nil, nil)
		if err != nil {
			if gvk != nil {
				skippedErrors = append(skippedErrors, fmt.Sprintf("%s: %v", gvk.GroupVersion(), err))
			} else {
				skippedErrors = append(skippedErrors, err.Error())
			}
			continue
		}

		if err := scheme.Convert(obj, dest, nil); err != nil {
			skippedErrors = append(skippedErrors, fmt.Sprintf("%s: convert: %v", obj.GetObjectKind().GroupVersionKind().GroupVersion(), err))
			continue
		}

		// scheme.Convert does not propagate TypeMeta. Set it explicitly
		// for versioned types so that callers see the expected
		// apiVersion/kind. Internal types keep an empty TypeMeta.
		if gvks[0].Version != runtime.APIVersionInternal {
			dest.GetObjectKind().SetGroupVersionKind(gvks[0])
		}
		return nil
	}
	if len(skippedErrors) > 0 {
		return fmt.Errorf("no compatible metadata version found in stream (errors: %s)", strings.Join(skippedErrors, "; "))
	}
	return fmt.Errorf("no metadata objects found in stream")
}

// ReadResourceClaimMetadataWithDriverName reads and decodes the metadata file
// for a directly referenced ResourceClaim from a specific driver.
// driverName identifies which driver's metadata to read, since each driver
// writes a separate file per request.
func ReadResourceClaimMetadataWithDriverName(driverName, claimName, requestName string) (*metadata.DeviceMetadata, error) {
	return readMetadata(metadata.ResourceClaimFilePath(driverName, claimName, requestName))
}

// ReadResourceClaimTemplateMetadataWithDriverName reads and decodes the
// metadata file for a template-generated claim from a specific driver.
// podClaimName is the pod-local name (pod.spec.resourceClaims[].name).
// driverName identifies which driver's metadata to read, since each driver
// writes a separate file per request.
func ReadResourceClaimTemplateMetadataWithDriverName(driverName, podClaimName, requestName string) (*metadata.DeviceMetadata, error) {
	return readMetadata(metadata.ResourceClaimTemplateFilePath(driverName, podClaimName, requestName))
}

// ReadResourceClaimMetadata reads and decodes all metadata files for a
// directly referenced ResourceClaim request, regardless of which driver(s)
// wrote them. It globs the request directory for all *-metadata.json files
// and merges the results into a single [metadata.DeviceMetadata].
func ReadResourceClaimMetadata(claimName, requestName string) (*metadata.DeviceMetadata, error) {
	dir := filepath.Join(metadata.ContainerDir, metadata.ResourceClaimsSubDir, claimName, requestName)
	return readRequestDir(dir)
}

// ReadResourceClaimTemplateMetadata reads and decodes all metadata files
// for a template-generated claim request, regardless of which driver(s) wrote
// them. It globs the request directory for all *-metadata.json files and
// merges the results into a single [metadata.DeviceMetadata].
func ReadResourceClaimTemplateMetadata(podClaimName, requestName string) (*metadata.DeviceMetadata, error) {
	dir := filepath.Join(metadata.ContainerDir, metadata.ResourceClaimTemplatesSubDir, podClaimName, requestName)
	return readRequestDir(dir)
}

func readRequestDir(dir string) (*metadata.DeviceMetadata, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*"+metadata.MetadataFileSuffix))
	if err != nil {
		return nil, fmt.Errorf("glob metadata files in %s: %w", dir, err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no metadata files found in %s", dir)
	}

	var merged metadata.DeviceMetadata
	for _, path := range matches {
		dm, err := readMetadata(path)
		if err != nil {
			return nil, err
		}
		if merged.PodClaimName == nil {
			merged.PodClaimName = dm.PodClaimName
		}
		merged.Requests = append(merged.Requests, dm.Requests...)
	}
	return &merged, nil
}

func readMetadata(path string) (*metadata.DeviceMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open metadata file: %w", err)
	}

	var dm metadata.DeviceMetadata
	decodeErr := DecodeMetadataFromStream(json.NewDecoder(f), &dm)
	if closeErr := f.Close(); closeErr != nil && decodeErr == nil {
		return nil, fmt.Errorf("close metadata file: %w", closeErr)
	}
	if decodeErr != nil {
		return nil, fmt.Errorf("%s: %w", path, decodeErr)
	}
	return &dm, nil
}
