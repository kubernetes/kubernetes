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

package kubeletplugin

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/dynamic-resource-allocation/api/metadata/install"
	"k8s.io/dynamic-resource-allocation/api/metadata/v1alpha1"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
)

// metadataScheme has both internal and v1alpha1 metadata types registered.
// metadataCodecFactory provides serializers and deserializers derived from
// the scheme. The encoder automatically sets apiVersion/kind from the scheme
// when encoding, so TypeMeta never needs to be hardcoded.
var (
	metadataScheme       = install.NewScheme()
	metadataCodecFactory = runtimeserializer.NewCodecFactory(metadataScheme)
	metadataEncoder      runtime.Encoder
	metadataDecoder      = metadataCodecFactory.UniversalDecoder(v1alpha1.SchemeGroupVersion)
)

func init() {
	info, ok := runtime.SerializerInfoForMediaType(metadataCodecFactory.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		panic("no JSON serializer registered for metadata scheme")
	}
	metadataEncoder = metadataCodecFactory.EncoderForVersion(info.PrettySerializer, v1alpha1.SchemeGroupVersion)
}

// DecodeMetadataToV1Alpha1 decodes metadata JSON bytes into a
// v1alpha1.DeviceMetadata object. This can be used by drivers and tests
// to read metadata files written by the framework.
func DecodeMetadataToV1Alpha1(data []byte) (*v1alpha1.DeviceMetadata, error) {
	obj, _, err := metadataDecoder.Decode(data, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("decode device metadata: %w", err)
	}
	dm, ok := obj.(*v1alpha1.DeviceMetadata)
	if !ok {
		return nil, fmt.Errorf("expected *v1alpha1.DeviceMetadata, got %T", obj)
	}
	return dm, nil
}

// CDI JSON types for metadata bind-mount specs. These mirror the CDI spec
// structure but only include the fields needed for file bind-mounts.
// These definitions are sufficient to generate simple CDI files which set
// the volume mount path for the metadata file.
// Drivers should use pre-generated CDI files or the
// github.com/container-orchestrated-devices/container-device-interface/pkg/cdi
// helper package to generate files.
// This is not done in Kubernetes to minimize dependencies.
type cdiSpec struct {
	Version string      `json:"cdiVersion"`
	Kind    string      `json:"kind"`
	Devices []cdiDevice `json:"devices"`
}

type cdiDevice struct {
	Name           string            `json:"name"`
	ContainerEdits cdiContainerEdits `json:"containerEdits"`
}

type cdiContainerEdits struct {
	Mounts []cdiMount `json:"mounts,omitempty"`
}

type cdiMount struct {
	HostPath      string   `json:"hostPath"`
	ContainerPath string   `json:"containerPath"`
	Options       []string `json:"options,omitempty"`
}

const (
	// metadataSubDir is the subdirectory under the plugin data directory
	// where metadata files are stored.
	metadataSubDir = "dra-device-metadata"

	// containerMetadataDir is the well-known directory inside containers
	// where metadata files are mounted.
	containerMetadataDir = "/var/run/dra-device-attributes"

	// DefaultCDIDir is the default directory for CDI spec files.
	DefaultCDIDir = "/var/run/cdi"

	cdiVersionStr     = "0.3.0"
	metadataFilePerms = os.FileMode(0644)
	metadataDirPerms  = os.FileMode(0755)
)

// claimRef bundles the identity and pod-level reference for a ResourceClaim.
// It is threaded through the write path so individual functions don't need
// long parameter lists.
type claimRef struct {
	namespace    string
	name         string
	uid          types.UID
	podClaimName *string
}

func newClaimRef(claim *resourceapi.ResourceClaim) claimRef {
	ref := claimRef{
		namespace: claim.Namespace,
		name:      claim.Name,
		uid:       claim.UID,
	}
	if v, ok := claim.Annotations[resourceapi.PodResourceClaimAnnotation]; ok {
		ref.podClaimName = &v
	}
	return ref
}

// metadataWriter handles writing metadata files and CDI specs for the
// device metadata feature. All state is derived from files on disk.
type metadataWriter struct {
	driverName    string
	pluginDataDir string
	cdiDir        string
}

// newMetadataWriter creates a new metadataWriter.
func newMetadataWriter(driverName, pluginDataDir, cdiDir string) *metadataWriter {
	return &metadataWriter{
		driverName:    driverName,
		pluginDataDir: pluginDataDir,
		cdiDir:        cdiDir,
	}
}

// processPreparedClaim writes metadata files and CDI specs for all requests
// in the prepared claim. It returns a map of base request name to CDI device
// ID that the caller should inject into the gRPC response.
func (w *metadataWriter) processPreparedClaim(
	claim *resourceapi.ResourceClaim,
	devices []Device,
) (map[string]string, error) {
	ref := newClaimRef(claim)

	preparedDevicesByRequest := make(map[string][]Device)
	for _, dev := range devices {
		for _, reqName := range dev.Requests {
			preparedDevicesByRequest[reqName] = append(preparedDevicesByRequest[reqName], dev)
		}
	}

	cdiDeviceIDs := make(map[string]string)

	for requestRef, devs := range preparedDevicesByRequest {
		baseReq := resourceclaim.BaseRequestRef(requestRef)

		if err := w.writeMetadataFile(ref, baseReq, requestRef, devs); err != nil {
			return nil, fmt.Errorf("write metadata for request %q: %w", requestRef, err)
		}

		cdiID, err := w.writeCDISpec(claim, baseReq)
		if err != nil {
			return nil, fmt.Errorf("write CDI spec for request %q: %w", requestRef, err)
		}

		cdiDeviceIDs[baseReq] = cdiID
	}

	return cdiDeviceIDs, nil
}

// cleanupClaim removes all metadata files and CDI specs for a claim.
// It is a no-op if no files exist for the given claim.
func (w *metadataWriter) cleanupClaim(claimNamespace, claimName string, claimUID types.UID) error {
	metadataDir := w.claimMetadataDir(claimNamespace, claimName)
	if err := os.RemoveAll(metadataDir); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove metadata directory %s: %w", metadataDir, err)
	}

	pattern := filepath.Join(w.cdiDir, fmt.Sprintf("%s-metadata-%s-*.json", w.driverName, string(claimUID)))
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("glob CDI specs %s: %w", pattern, err)
	}
	for _, cdiPath := range matches {
		if err := os.Remove(cdiPath); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove CDI spec %s: %w", cdiPath, err)
		}
	}

	return nil
}

// updateRequestMetadata overwrites the metadata file for a specific request.
// It reads the existing file to obtain the current generation, increments it,
// and writes back the updated metadata. This is used by drivers during NRI
// hooks when device details (e.g., network info) become available after
// PrepareResourceClaims.
//
// requestName may include a subrequest name (e.g. "gpu/high-memory"); the
// base request name is used for the file path.
func (w *metadataWriter) updateRequestMetadata(
	ref claimRef,
	requestName string,
	devices []Device,
) error {
	baseReq := resourceclaim.BaseRequestRef(requestName)
	filePath := w.metadataFilePath(ref.namespace, ref.name, baseReq)

	existing, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("request %q not found in prepared claim %s/%s", requestName, ref.namespace, ref.name)
		}
		return fmt.Errorf("read metadata file: %w", err)
	}

	prev, err := DecodeMetadataToV1Alpha1(existing)
	if err != nil {
		return fmt.Errorf("decode existing metadata: %w", err)
	}

	ref.podClaimName = prev.PodClaimName

	generation := prev.Generation + 1
	dm := w.buildDeviceMetadata(ref, generation, requestName, devices)

	data, err := runtime.Encode(metadataEncoder, dm)
	if err != nil {
		return fmt.Errorf("encode metadata: %w", err)
	}

	if err := os.WriteFile(filePath, data, metadataFilePerms); err != nil {
		return fmt.Errorf("write metadata file: %w", err)
	}

	return nil
}

// writeMetadataFile writes the metadata JSON for one request in a claim.
// baseRequestName is used for the file path; requestRef (which may include
// a subrequest name) is recorded in the metadata content.
func (w *metadataWriter) writeMetadataFile(
	ref claimRef,
	baseRequestName string,
	requestRef string,
	devices []Device,
) error {
	dm := w.buildDeviceMetadata(ref, 1, requestRef, devices)

	data, err := runtime.Encode(metadataEncoder, dm)
	if err != nil {
		return fmt.Errorf("encode metadata: %w", err)
	}

	filePath := w.metadataFilePath(ref.namespace, ref.name, baseRequestName)
	if err := os.MkdirAll(filepath.Dir(filePath), metadataDirPerms); err != nil {
		return fmt.Errorf("create metadata directory: %w", err)
	}

	if err := os.WriteFile(filePath, data, metadataFilePerms); err != nil {
		return fmt.Errorf("write metadata file: %w", err)
	}

	return nil
}

// buildDeviceMetadata constructs the v1alpha1 DeviceMetadata from claim info
// and device data.
func (w *metadataWriter) buildDeviceMetadata(
	ref claimRef,
	generation int64,
	requestName string,
	devices []Device,
) *v1alpha1.DeviceMetadata {
	metadataDevices := make([]v1alpha1.Device, 0, len(devices))
	for _, dev := range devices {
		d := v1alpha1.Device{
			Driver: w.driverName,
			Pool:   dev.PoolName,
			Name:   dev.DeviceName,
		}
		if dev.Metadata != nil {
			if dev.Metadata.Attributes != nil {
				d.Attributes = make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, len(dev.Metadata.Attributes))
				for k, v := range dev.Metadata.Attributes {
					d.Attributes[resourceapi.QualifiedName(k)] = v
				}
			}
			d.NetworkData = dev.Metadata.NetworkData
		}
		metadataDevices = append(metadataDevices, d)
	}

	return &v1alpha1.DeviceMetadata{
		ObjectMeta: metav1.ObjectMeta{
			Name:       ref.name,
			Namespace:  ref.namespace,
			UID:        ref.uid,
			Generation: generation,
		},
		PodClaimName: ref.podClaimName,
		Requests: []v1alpha1.DeviceMetadataRequest{
			{
				Name:    requestName,
				Devices: metadataDevices,
			},
		},
	}
}

// writeCDISpec writes a CDI spec that bind-mounts the metadata file into the
// container. Returns the CDI device ID.
func (w *metadataWriter) writeCDISpec(
	claim *resourceapi.ResourceClaim,
	requestName string,
) (string, error) {
	deviceName := string(claim.UID) + "_" + requestName

	spec := &cdiSpec{
		Version: cdiVersionStr,
		Kind:    w.driverName + "/metadata",
		Devices: []cdiDevice{
			{
				Name: deviceName,
				ContainerEdits: cdiContainerEdits{
					Mounts: []cdiMount{
						{
							HostPath:      w.metadataFilePath(claim.Namespace, claim.Name, requestName),
							ContainerPath: w.containerFilePath(claim.Name, requestName),
							Options:       []string{"ro", "bind"},
						},
					},
				},
			},
		},
	}

	data, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal CDI spec: %w", err)
	}

	if err := os.MkdirAll(w.cdiDir, metadataDirPerms); err != nil {
		return "", fmt.Errorf("create CDI directory: %w", err)
	}

	cdiPath := w.cdiSpecFilePath(claim.UID, requestName)
	if err := os.WriteFile(cdiPath, data, metadataFilePerms); err != nil {
		return "", fmt.Errorf("write CDI spec file: %w", err)
	}

	// CDI device ID: {driverName}/metadata={claimUID}_{requestName}
	return w.driverName + "/metadata=" + deviceName, nil
}

// Path helpers.

// claimMetadataDir returns the host directory for a claim's metadata files.
// Format: {pluginDataDir}/dra-device-metadata/{claimNs}_{claimName}/
func (w *metadataWriter) claimMetadataDir(claimNs, claimName string) string {
	return filepath.Join(w.pluginDataDir, metadataSubDir, claimNs+"_"+claimName)
}

// metadataFilePath returns the host path for a request's metadata file.
// Format: {pluginDataDir}/dra-device-metadata/{claimNs}_{claimName}/{requestName}/metadata.json
func (w *metadataWriter) metadataFilePath(claimNs, claimName, requestName string) string {
	return filepath.Join(w.claimMetadataDir(claimNs, claimName), requestName, "metadata.json")
}

// cdiSpecFilePath returns the path for the CDI spec file.
func (w *metadataWriter) cdiSpecFilePath(claimUID types.UID, requestName string) string {
	return filepath.Join(w.cdiDir, fmt.Sprintf("%s-metadata-%s-%s.json", w.driverName, string(claimUID), requestName))
}

// containerFilePath returns the path where the metadata file appears inside
// the container.
// Format: /var/run/dra-device-attributes/{claimName}/{requestName}/{driverName}-metadata.json
func (w *metadataWriter) containerFilePath(claimName, requestName string) string {
	return filepath.Join(containerMetadataDir, claimName, requestName, w.driverName+"-metadata.json")
}
