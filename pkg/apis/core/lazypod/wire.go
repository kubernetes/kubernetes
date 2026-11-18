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

package lazypod

import (
	"bytes"
	"fmt"
	"io"
	"math"

	"google.golang.org/protobuf/encoding/protowire"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	core "k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/features"
)

// Kubernetes Protobuf framing constants (k8s.io/apimachinery/pkg/runtime).
var (
	protoEncodingPrefix = []byte{0x6b, 0x38, 0x73, 0x00} // "k8s\x00"

	// Canonical prefix before the raw payload length varint for v1.Pod:
	// "k8s\x00" + Unknown{TypeMeta{APIVersion:"v1", Kind:"Pod"}} + raw field tag (0x12).
	podEnvelopePrefix = []byte{
		0x6b, 0x38, 0x73, 0x00,
		0x0a, 0x09,
		0x0a, 0x02, 'v', '1',
		0x12, 0x03, 'P', 'o', 'd',
		0x12,
	}

	// Canonical prefix after "k8s\x00" and before the raw payload length varint for v1.PodList:
	// Unknown{TypeMeta{APIVersion:"v1", Kind:"PodList"}} + raw field tag (0x12).
	podListEnvelopeHeader = []byte{
		0x0a, 0x0d,
		0x0a, 0x02, 'v', '1',
		0x12, 0x07, 'P', 'o', 'd', 'L', 'i', 's', 't',
		0x12,
	}

	// Empty ContentEncoding (tag 3) and ContentType (tag 4) emitted by runtime.Unknown.
	unknownEnvelopeTrailer = []byte{0x1a, 0x00, 0x22, 0x00}
)

// Protobuf wire field numbers verified against staging/src/k8s.io/apimachinery and core/v1 types.
const (
	// runtime.Unknown
	unknownFieldTypeMeta        protowire.Number = 1
	unknownFieldRaw             protowire.Number = 2
	unknownFieldContentEncoding protowire.Number = 3
	unknownFieldContentType     protowire.Number = 4

	// runtime.TypeMeta
	typeMetaFieldAPIVersion protowire.Number = 1
	typeMetaFieldKind       protowire.Number = 2

	// v1.Pod
	podFieldMetadata protowire.Number = 1
	podFieldSpec     protowire.Number = 2
	podFieldStatus   protowire.Number = 3

	// metav1.ObjectMeta
	metaFieldName                       protowire.Number = 1
	metaFieldGenerateName               protowire.Number = 2
	metaFieldNamespace                  protowire.Number = 3
	metaFieldSelfLink                   protowire.Number = 4
	metaFieldUID                        protowire.Number = 5
	metaFieldResourceVersion            protowire.Number = 6
	metaFieldGeneration                 protowire.Number = 7
	metaFieldDeletionTimestamp          protowire.Number = 9
	metaFieldDeletionGracePeriodSeconds protowire.Number = 10
	metaFieldLabels                     protowire.Number = 11
	metaFieldAnnotations                protowire.Number = 12
	metaFieldOwnerReferences            protowire.Number = 13
	metaFieldFinalizers                 protowire.Number = 14
	metaFieldManagedFields              protowire.Number = 17

	// v1.PodSpec
	specFieldVolumes                       protowire.Number = 1
	specFieldContainers                    protowire.Number = 2
	specFieldRestartPolicy                 protowire.Number = 3
	specFieldTerminationGracePeriodSeconds protowire.Number = 4
	specFieldDNSPolicy                     protowire.Number = 6
	specFieldServiceAccountName            protowire.Number = 8
	specFieldDeprecatedServiceAccount      protowire.Number = 9
	specFieldNodeName                      protowire.Number = 10
	specFieldHostNetwork                   protowire.Number = 11
	specFieldSecurityContext               protowire.Number = 14
	specFieldAffinity                      protowire.Number = 18
	specFieldSchedulerName                 protowire.Number = 19
	specFieldInitContainers                protowire.Number = 20
	specFieldEnableServiceLinks            protowire.Number = 30
	specFieldOverhead                      protowire.Number = 32
	specFieldEphemeralContainers           protowire.Number = 34
	specFieldResources                     protowire.Number = 40

	// v1.PodStatus
	statusFieldPhase                         protowire.Number = 1
	statusFieldPodIP                         protowire.Number = 6
	statusFieldContainerStatuses             protowire.Number = 8
	statusFieldInitContainerStatuses         protowire.Number = 10
	statusFieldNominatedNodeName             protowire.Number = 11
	statusFieldPodIPs                        protowire.Number = 12
	statusFieldEphemeralContainerStatuses    protowire.Number = 13
	statusFieldResourceClaimStatuses         protowire.Number = 15
	statusFieldHostIPs                       protowire.Number = 16
	statusFieldExtendedResourceClaimStatus   protowire.Number = 18
	statusFieldAllocatedResources            protowire.Number = 19
	statusFieldResources                     protowire.Number = 20
	statusFieldExtraNodeAllocatableResources protowire.Number = 24

	// v1.PodIP / v1.HostIP
	ipEntryFieldIP protowire.Number = 1

	// v1.Container
	containerFieldPorts                    protowire.Number = 6
	containerFieldEnv                      protowire.Number = 7
	containerFieldResources                protowire.Number = 8
	containerFieldLivenessProbe            protowire.Number = 10
	containerFieldReadinessProbe           protowire.Number = 11
	containerFieldLifecycle                protowire.Number = 12
	containerFieldTerminationMessagePath   protowire.Number = 13
	containerFieldImagePullPolicy          protowire.Number = 14
	containerFieldTerminationMessagePolicy protowire.Number = 20
	containerFieldStartupProbe             protowire.Number = 22

	// v1.ContainerPort
	portFieldHostPort      protowire.Number = 2
	portFieldContainerPort protowire.Number = 3
	portFieldProtocol      protowire.Number = 4

	// v1.EnvVar / v1.EnvVarSource / v1.ObjectFieldSelector / v1.FileKeySelector
	envVarFieldValueFrom       protowire.Number = 3
	envVarSourceFieldFieldRef  protowire.Number = 1
	envVarSourceFieldFileKey   protowire.Number = 5
	fieldSelectorFieldAPIVer   protowire.Number = 1
	fileKeySelectorFieldOption protowire.Number = 4

	// v1.ResourceRequirements / map entry / resource.Quantity
	resReqFieldLimits   protowire.Number = 1
	resReqFieldRequests protowire.Number = 2
	mapEntryFieldKey    protowire.Number = 1
	mapEntryFieldValue  protowire.Number = 2
	quantityFieldString protowire.Number = 1

	// v1.ContainerStatus
	containerStatusFieldAllocatedResources protowire.Number = 10
	containerStatusFieldResources          protowire.Number = 11

	// v1.Volume / v1.VolumeSource and nested sources
	volumeFieldSource              protowire.Number = 2
	volSrcFieldHostPath            protowire.Number = 1
	volSrcFieldEmptyDir            protowire.Number = 2
	volSrcFieldSecret              protowire.Number = 6
	volSrcFieldISCSI               protowire.Number = 8
	volSrcFieldRBD                 protowire.Number = 11
	volSrcFieldDownwardAPI         protowire.Number = 16
	volSrcFieldConfigMap           protowire.Number = 19
	volSrcFieldAzureDisk           protowire.Number = 22
	volSrcFieldScaleIO             protowire.Number = 25
	volSrcFieldProjected           protowire.Number = 26
	volSrcFieldEphemeral           protowire.Number = 29
	volSrcFieldImage               protowire.Number = 30
	hostPathFieldType              protowire.Number = 2
	secretVolFieldDefaultMode      protowire.Number = 3
	configMapVolFieldDefaultMode   protowire.Number = 3
	downwardVolFieldItems          protowire.Number = 1
	downwardVolFieldDefaultMode    protowire.Number = 2
	downwardFileFieldFieldRef      protowire.Number = 2
	projectedVolFieldSources       protowire.Number = 1
	projectedVolFieldDefaultMode   protowire.Number = 2
	volProjFieldDownwardAPI        protowire.Number = 2
	volProjFieldSAToken            protowire.Number = 4
	saTokenProjFieldExpirationSecs protowire.Number = 2
)

// wirePod holds an immutable, pre-framed core/v1 Pod Protobuf envelope together
// with byte offsets into envelope for zero-allocation GET, LIST, and WATCH serving.
type wirePod struct {
	envelope []byte

	podStart     int32
	podEnd       int32
	metaTagStart int32
	metaStart    int32
	metaEnd      int32
	specStart    int32
	specEnd      int32
	statStart    int32
	statEnd      int32
	rvStart      int32
	rvEnd        int32

	rv string
}

func (w *wirePod) podBytes() []byte {
	return w.envelope[w.podStart:w.podEnd]
}

func (w *wirePod) metaBytes() []byte {
	if w.metaStart == w.metaEnd {
		return nil
	}
	return w.envelope[w.metaStart:w.metaEnd]
}

func (w *wirePod) specBytes() []byte {
	if w.specStart == w.specEnd {
		return nil
	}
	return w.envelope[w.specStart:w.specEnd]
}

func (w *wirePod) statusBytes() []byte {
	if w.statStart == w.statEnd {
		return nil
	}
	return w.envelope[w.statStart:w.statEnd]
}

// parseWirePod validates a Kubernetes Protobuf-framed v1.Pod message and returns
// an immutable wirePod when it is already defaulted and structurally clean.
// Returns (nil, false) for non-protobuf payloads, corrupt framing, duplicate
// top-level sub-messages, or pods requiring SetObjectDefaults_Pod.
func parseWirePod(data []byte) (*wirePod, bool) {
	if len(data) <= len(protoEncodingPrefix) || len(data) > math.MaxInt32-64 {
		return nil, false
	}
	if !bytes.Equal(data[:len(protoEncodingPrefix)], protoEncodingPrefix) {
		return nil, false
	}

	raw, ok := parseUnknownV1Pod(data[len(protoEncodingPrefix):])
	if !ok || len(raw) == 0 {
		return nil, false
	}

	metaTagRel, metaRelStart, metaRelEnd, specRelStart, specRelEnd, statRelStart, statRelEnd, ok := scanPodTopLevel(raw)
	if !ok {
		return nil, false
	}

	metaPayload := raw[metaRelStart:metaRelEnd]
	rvRelStart, rvRelEnd, rv, ok := scanObjectMetaRV(metaPayload)
	if !ok {
		return nil, false
	}

	var specBytes, statusBytes []byte
	if specRelEnd > specRelStart {
		specBytes = raw[specRelStart:specRelEnd]
	}
	if statRelEnd > statRelStart {
		statusBytes = raw[statRelStart:statRelEnd]
	}
	if !isDefaultedPodWire(specBytes, statusBytes) {
		return nil, false
	}

	rawLenSize := protowire.SizeVarint(uint64(len(raw)))
	envLen := len(podEnvelopePrefix) + rawLenSize + len(raw) + len(unknownEnvelopeTrailer)
	env := make([]byte, 0, envLen)
	env = append(env, podEnvelopePrefix...)
	env = protowire.AppendVarint(env, uint64(len(raw)))
	podStart := int32(len(env))
	env = append(env, raw...)
	podEnd := int32(len(env))
	env = append(env, unknownEnvelopeTrailer...)

	wp := &wirePod{
		envelope:     env,
		podStart:     podStart,
		podEnd:       podEnd,
		metaTagStart: podStart + int32(metaTagRel),
		metaStart:    podStart + int32(metaRelStart),
		metaEnd:      podStart + int32(metaRelEnd),
		rvStart:      podStart + int32(metaRelStart+rvRelStart),
		rvEnd:        podStart + int32(metaRelStart+rvRelEnd),
		rv:           rv,
	}
	if specRelEnd > specRelStart {
		wp.specStart = podStart + int32(specRelStart)
		wp.specEnd = podStart + int32(specRelEnd)
	}
	if statRelEnd > statRelStart {
		wp.statStart = podStart + int32(statRelStart)
		wp.statEnd = podStart + int32(statRelEnd)
	}
	return wp, true
}

// withResourceVersion returns a new immutable wirePod with ObjectMeta.resourceVersion
// spliced to newRV inside a canonical Kubernetes Protobuf envelope.
func (w *wirePod) withResourceVersion(newRV string) *wirePod {
	beforeRV := w.envelope[w.metaStart:w.rvStart]
	afterRV := w.envelope[w.rvEnd:w.metaEnd]

	rvFieldLen := 1 + protowire.SizeVarint(uint64(len(newRV))) + len(newRV)
	newMetaLen := len(beforeRV) + rvFieldLen + len(afterRV)
	newMetaFieldLen := 1 + protowire.SizeVarint(uint64(newMetaLen)) + newMetaLen

	podBeforeMeta := w.envelope[w.podStart:w.metaTagStart]
	podAfterMeta := w.envelope[w.metaEnd:w.podEnd]
	newPodLen := len(podBeforeMeta) + newMetaFieldLen + len(podAfterMeta)

	envLen := len(podEnvelopePrefix) + protowire.SizeVarint(uint64(newPodLen)) + newPodLen + len(unknownEnvelopeTrailer)
	env := make([]byte, 0, envLen)
	env = append(env, podEnvelopePrefix...)
	env = protowire.AppendVarint(env, uint64(newPodLen))

	newPodStart := int32(len(env))
	env = append(env, podBeforeMeta...)

	newMetaTagStart := int32(len(env))
	env = append(env, 0x0a)
	env = protowire.AppendVarint(env, uint64(newMetaLen))

	newMetaStart := int32(len(env))
	env = append(env, beforeRV...)

	newRVStart := int32(len(env))
	env = append(env, 0x32) // (metaFieldResourceVersion << 3) | protowire.BytesType
	env = protowire.AppendVarint(env, uint64(len(newRV)))
	env = append(env, newRV...)
	newRVEnd := int32(len(env))

	env = append(env, afterRV...)
	newMetaEnd := int32(len(env))

	env = append(env, podAfterMeta...)
	newPodEnd := int32(len(env))
	env = append(env, unknownEnvelopeTrailer...)

	shift := func(old int32) int32 {
		if old < w.metaTagStart {
			return old + (newPodStart - w.podStart)
		}
		return old + (newMetaEnd - w.metaEnd)
	}

	out := &wirePod{
		envelope:     env,
		podStart:     newPodStart,
		podEnd:       newPodEnd,
		metaTagStart: newMetaTagStart,
		metaStart:    newMetaStart,
		metaEnd:      newMetaEnd,
		rvStart:      newRVStart,
		rvEnd:        newRVEnd,
		rv:           newRV,
	}
	if w.specEnd > w.specStart {
		out.specStart = shift(w.specStart)
		out.specEnd = shift(w.specEnd)
	}
	if w.statEnd > w.statStart {
		out.statStart = shift(w.statStart)
		out.statEnd = shift(w.statEnd)
	}
	return out
}

func (w *wirePod) decodeObjectMeta() (metav1.ObjectMeta, error) {
	var out metav1.ObjectMeta
	if w.metaEnd > w.metaStart {
		if err := out.Unmarshal(w.envelope[w.metaStart:w.metaEnd]); err != nil {
			return metav1.ObjectMeta{}, err
		}
	}
	return out, nil
}

func (w *wirePod) decodeToCorePod() (*core.Pod, error) {
	var v1Pod v1.Pod
	if err := v1Pod.Unmarshal(w.podBytes()); err != nil {
		return nil, err
	}
	corePod := &core.Pod{}
	if err := corev1.Convert_v1_Pod_To_core_Pod(&v1Pod, corePod, nil); err != nil {
		return nil, err
	}
	return corePod, nil
}

// streamPodListProtobuf writes a core/v1 PodList Protobuf envelope directly to w
// using each item's pre-framed podBytes without copying into []core.Pod / []v1.Pod.
func streamPodListProtobuf(w io.Writer, list *PodList) error {
	listMetaSize := list.ListMeta.Size()
	totalRawSize := 1 + protowire.SizeVarint(uint64(listMetaSize)) + listMetaSize
	for i := range list.Items {
		n := len(list.Items[i].wire.podBytes())
		totalRawSize += 1 + protowire.SizeVarint(uint64(n)) + n
	}

	// Write the 4-byte Kubernetes Protobuf prefix first so deferredResponseWriter
	// preserves identical streaming gzip detection to protobuf/collections.go.
	if _, err := w.Write(protoEncodingPrefix); err != nil {
		return err
	}

	header := make([]byte, 0, len(podListEnvelopeHeader)+protowire.SizeVarint(uint64(totalRawSize))+1+protowire.SizeVarint(uint64(listMetaSize))+listMetaSize)
	header = append(header, podListEnvelopeHeader...)
	header = protowire.AppendVarint(header, uint64(totalRawSize))
	header = append(header, 0x0a)
	header = protowire.AppendVarint(header, uint64(listMetaSize))

	metaStart := len(header)
	header = header[:metaStart+listMetaSize]
	n, err := list.ListMeta.MarshalToSizedBuffer(header[metaStart:])
	if err != nil {
		return err
	}
	if n != listMetaSize {
		return fmt.Errorf("lazypod: ListMeta marshal size mismatch: wrote %d, want %d", n, listMetaSize)
	}
	if _, err := w.Write(header); err != nil {
		return err
	}

	itemHeader := make([]byte, 0, 1+protowire.SizeVarint(math.MaxUint64))
	for i := range list.Items {
		podRaw := list.Items[i].wire.podBytes()
		itemHeader = itemHeader[:0]
		itemHeader = append(itemHeader, 0x12)
		itemHeader = protowire.AppendVarint(itemHeader, uint64(len(podRaw)))
		if _, err := w.Write(itemHeader); err != nil {
			return err
		}
		if _, err := w.Write(podRaw); err != nil {
			return err
		}
	}

	_, err = w.Write(unknownEnvelopeTrailer)
	return err
}

// --- Envelope & top-level parsing ---

func parseUnknownV1Pod(b []byte) ([]byte, bool) {
	var raw []byte
	var seenRaw bool
	var apiVersion, kind []byte

	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return nil, false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return nil, false
		}
		switch num {
		case unknownFieldTypeMeta:
			if typ != protowire.BytesType {
				return nil, false
			}
			tm, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 {
				return nil, false
			}
			av, k, ok := parseTypeMeta(tm)
			if !ok {
				return nil, false
			}
			apiVersion, kind = av, k
		case unknownFieldRaw:
			if typ != protowire.BytesType || seenRaw {
				return nil, false
			}
			r, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 {
				return nil, false
			}
			raw = r
			seenRaw = true
		case unknownFieldContentEncoding, unknownFieldContentType:
			if typ != protowire.BytesType {
				return nil, false
			}
			v, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 || len(v) != 0 {
				return nil, false
			}
		default:
			return nil, false
		}
		offset += valLen
	}

	if !seenRaw || string(apiVersion) != "v1" || string(kind) != "Pod" {
		return nil, false
	}
	return raw, true
}

func parseTypeMeta(b []byte) (apiVersion, kind []byte, ok bool) {
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return nil, nil, false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return nil, nil, false
		}
		if typ == protowire.BytesType {
			v, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 {
				return nil, nil, false
			}
			switch num {
			case typeMetaFieldAPIVersion:
				apiVersion = v
			case typeMetaFieldKind:
				kind = v
			}
		}
		offset += valLen
	}
	return apiVersion, kind, true
}

func scanPodTopLevel(raw []byte) (metaTag, metaStart, metaEnd, specStart, specEnd, statStart, statEnd int, ok bool) {
	var seenMeta, seenSpec, seenStat bool
	offset := 0
	for offset < len(raw) {
		fieldStart := offset
		num, typ, tagLen := protowire.ConsumeTag(raw[offset:])
		if tagLen < 0 {
			return 0, 0, 0, 0, 0, 0, 0, false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, raw[offset:])
		if valLen < 0 || typ != protowire.BytesType {
			return 0, 0, 0, 0, 0, 0, 0, false
		}
		subLen, lenSize := protowire.ConsumeVarint(raw[offset : offset+valLen])
		if lenSize < 0 || lenSize+int(subLen) != valLen {
			return 0, 0, 0, 0, 0, 0, 0, false
		}
		payloadStart := offset + lenSize
		payloadEnd := offset + valLen
		switch num {
		case podFieldMetadata:
			if seenMeta {
				return 0, 0, 0, 0, 0, 0, 0, false
			}
			seenMeta = true
			metaTag = fieldStart
			metaStart = payloadStart
			metaEnd = payloadEnd
		case podFieldSpec:
			if seenSpec {
				return 0, 0, 0, 0, 0, 0, 0, false
			}
			seenSpec = true
			specStart = payloadStart
			specEnd = payloadEnd
		case podFieldStatus:
			if seenStat {
				return 0, 0, 0, 0, 0, 0, 0, false
			}
			seenStat = true
			statStart = payloadStart
			statEnd = payloadEnd
		default:
			return 0, 0, 0, 0, 0, 0, 0, false
		}
		offset += valLen
	}
	if !seenMeta {
		return 0, 0, 0, 0, 0, 0, 0, false
	}
	return metaTag, metaStart, metaEnd, specStart, specEnd, statStart, statEnd, true
}

func scanObjectMetaRV(metaBytes []byte) (rvStart, rvEnd int, rv string, ok bool) {
	rvStart = len(metaBytes)
	rvEnd = len(metaBytes)
	seenRV := false

	offset := 0
	for offset < len(metaBytes) {
		fieldStart := offset
		num, typ, tagLen := protowire.ConsumeTag(metaBytes[offset:])
		if tagLen < 0 {
			return 0, 0, "", false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, metaBytes[offset:])
		if valLen < 0 {
			return 0, 0, "", false
		}
		if num == metaFieldResourceVersion {
			if typ != protowire.BytesType || seenRV {
				return 0, 0, "", false
			}
			v, n := protowire.ConsumeBytes(metaBytes[offset : offset+valLen])
			if n < 0 {
				return 0, 0, "", false
			}
			seenRV = true
			rvStart = fieldStart
			rvEnd = offset + valLen
			if len(v) > 0 {
				rv = string(v)
			}
		}
		offset += valLen
	}
	return rvStart, rvEnd, rv, true
}

// --- Conservative SetObjectDefaults_Pod wire check ---

// isDefaultedPodWire reports whether wire-encoded PodSpec and PodStatus bytes are
// guaranteed to be unchanged by corev1.SetObjectDefaults_Pod. Returning false is
// always safe because StorageCodec falls back to standard decoding + defaulting.
func isDefaultedPodWire(specBytes, statusBytes []byte) bool {
	if !isDefaultedPodSpecWire(specBytes) {
		return false
	}
	return isDefaultedPodStatusWire(statusBytes)
}

func isDefaultedPodSpecWire(specBytes []byte) bool {
	if len(specBytes) == 0 {
		return false
	}
	plrLegacyDefault := utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) &&
		!utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourcesFixDefaulting)
	imageVolEnabled := utilfeature.DefaultFeatureGate.Enabled(features.ImageVolume)

	var (
		hasRestartPolicy  bool
		hasDNSPolicy      bool
		hasSchedulerName  bool
		hasTermGrace      bool
		hasSecCtx         bool
		hasEnableSvcLinks bool
		hostNetwork       bool
		saName            []byte
		deprecatedSA      []byte
		containersHavePrt bool
	)

	offset := 0
	for offset < len(specBytes) {
		num, typ, tagLen := protowire.ConsumeTag(specBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, specBytes[offset:])
		if valLen < 0 {
			return false
		}
		field := specBytes[offset : offset+valLen]
		switch num {
		case specFieldRestartPolicy:
			if typ != protowire.BytesType {
				return false
			}
			v, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			hasRestartPolicy = len(v) > 0
		case specFieldDNSPolicy:
			if typ != protowire.BytesType {
				return false
			}
			v, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			hasDNSPolicy = len(v) > 0
		case specFieldSchedulerName:
			if typ != protowire.BytesType {
				return false
			}
			v, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			hasSchedulerName = len(v) > 0
		case specFieldTerminationGracePeriodSeconds:
			if typ != protowire.VarintType {
				return false
			}
			v, n := protowire.ConsumeVarint(field)
			if n < 0 || int64(v) < 0 {
				return false
			}
			hasTermGrace = true
		case specFieldSecurityContext:
			if typ != protowire.BytesType {
				return false
			}
			hasSecCtx = true
		case specFieldEnableServiceLinks:
			if typ != protowire.VarintType {
				return false
			}
			hasEnableSvcLinks = true
		case specFieldHostNetwork:
			if typ != protowire.VarintType {
				return false
			}
			v, n := protowire.ConsumeVarint(field)
			if n < 0 {
				return false
			}
			hostNetwork = v != 0
		case specFieldServiceAccountName:
			if typ != protowire.BytesType {
				return false
			}
			v, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			saName = v
		case specFieldDeprecatedServiceAccount:
			if typ != protowire.BytesType {
				return false
			}
			v, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			deprecatedSA = v
		case specFieldVolumes:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 || !isDefaultedVolumeWire(sub, imageVolEnabled) {
				return false
			}
		case specFieldContainers, specFieldInitContainers:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			ok, hasPorts := isDefaultedContainerWire(sub)
			if !ok {
				return false
			}
			if hasPorts {
				containersHavePrt = true
			}
		case specFieldEphemeralContainers:
			// Rare: fall back to full defaulting.
			return false
		case specFieldOverhead:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 || !isDefaultedResourceEntryWire(sub) {
				return false
			}
		case specFieldResources:
			if plrLegacyDefault || typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 || !isDefaultedResourceReqWire(sub, false) {
				return false
			}
		}
		offset += valLen
	}

	if !hasRestartPolicy || !hasDNSPolicy || !hasSchedulerName || !hasTermGrace || !hasSecCtx || !hasEnableSvcLinks {
		return false
	}
	if !bytes.Equal(saName, deprecatedSA) {
		return false
	}
	if hostNetwork && containersHavePrt {
		return false
	}
	if plrLegacyDefault {
		return false
	}
	return true
}

func isDefaultedPodStatusWire(statusBytes []byte) bool {
	var (
		podIP      []byte
		firstPodIP []byte
		hasPodIPs  bool
	)
	offset := 0
	for offset < len(statusBytes) {
		num, typ, tagLen := protowire.ConsumeTag(statusBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, statusBytes[offset:])
		if valLen < 0 {
			return false
		}
		field := statusBytes[offset : offset+valLen]
		switch num {
		case statusFieldPodIP:
			if typ != protowire.BytesType {
				return false
			}
			v, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			podIP = v
		case statusFieldPodIPs:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false
			}
			if !hasPodIPs {
				ip, ok := scanSingleStringFieldBytes(sub, ipEntryFieldIP)
				if !ok {
					return false
				}
				firstPodIP = ip
				hasPodIPs = true
			}
		case statusFieldContainerStatuses, statusFieldInitContainerStatuses, statusFieldEphemeralContainerStatuses:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 || !isDefaultedContainerStatusWire(sub) {
				return false
			}
		case statusFieldAllocatedResources:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 || !isDefaultedResourceEntryWire(sub) {
				return false
			}
		case statusFieldResources:
			if typ != protowire.BytesType {
				return false
			}
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 || !isDefaultedResourceReqWire(sub, false) {
				return false
			}
		case statusFieldExtraNodeAllocatableResources:
			return false
		}
		offset += valLen
	}

	hasIP := len(podIP) > 0
	if hasIP != hasPodIPs {
		return false
	}
	if hasIP && !bytes.Equal(podIP, firstPodIP) {
		return false
	}
	return true
}

func isDefaultedContainerWire(cBytes []byte) (defaulted bool, hasPorts bool) {
	var hasPullPolicy, hasTermPath, hasTermPolicy bool
	offset := 0
	for offset < len(cBytes) {
		num, typ, tagLen := protowire.ConsumeTag(cBytes[offset:])
		if tagLen < 0 {
			return false, false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, cBytes[offset:])
		if valLen < 0 {
			return false, false
		}
		field := cBytes[offset : offset+valLen]
		if typ == protowire.BytesType {
			sub, n := protowire.ConsumeBytes(field)
			if n < 0 {
				return false, false
			}
			switch num {
			case containerFieldImagePullPolicy:
				hasPullPolicy = len(sub) > 0
			case containerFieldTerminationMessagePath:
				hasTermPath = len(sub) > 0
			case containerFieldTerminationMessagePolicy:
				hasTermPolicy = len(sub) > 0
			case containerFieldPorts:
				hasPorts = true
				protoStr, ok := scanSingleStringFieldBytes(sub, portFieldProtocol)
				if !ok || len(protoStr) == 0 {
					return false, false
				}
			case containerFieldEnv:
				if !isDefaultedEnvVarWire(sub) {
					return false, false
				}
			case containerFieldResources:
				if !isDefaultedResourceReqWire(sub, true) {
					return false, false
				}
			case containerFieldLivenessProbe, containerFieldReadinessProbe, containerFieldStartupProbe, containerFieldLifecycle:
				// Conservative fallback: probes/lifecycle have multiple defaulting branches and feature gates.
				return false, false
			}
		}
		offset += valLen
	}
	return hasPullPolicy && hasTermPath && hasTermPolicy, hasPorts
}

func isDefaultedEnvVarWire(envBytes []byte) bool {
	offset := 0
	for offset < len(envBytes) {
		num, typ, tagLen := protowire.ConsumeTag(envBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, envBytes[offset:])
		if valLen < 0 {
			return false
		}
		if num == envVarFieldValueFrom && typ == protowire.BytesType {
			vf, n := protowire.ConsumeBytes(envBytes[offset : offset+valLen])
			if n < 0 {
				return false
			}
			vOff := 0
			for vOff < len(vf) {
				vNum, vTyp, vTagLen := protowire.ConsumeTag(vf[vOff:])
				if vTagLen < 0 {
					return false
				}
				vOff += vTagLen
				vLen := protowire.ConsumeFieldValue(vNum, vTyp, vf[vOff:])
				if vLen < 0 {
					return false
				}
				if vTyp == protowire.BytesType {
					sub, sn := protowire.ConsumeBytes(vf[vOff : vOff+vLen])
					if sn < 0 {
						return false
					}
					switch vNum {
					case envVarSourceFieldFieldRef:
						apiVer, ok := scanSingleStringFieldBytes(sub, fieldSelectorFieldAPIVer)
						if !ok || len(apiVer) == 0 {
							return false
						}
					case envVarSourceFieldFileKey:
						if !hasField(sub, fileKeySelectorFieldOption) {
							return false
						}
					}
				}
				vOff += vLen
			}
		}
		offset += valLen
	}
	return true
}

func isDefaultedContainerStatusWire(csBytes []byte) bool {
	offset := 0
	for offset < len(csBytes) {
		num, typ, tagLen := protowire.ConsumeTag(csBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, csBytes[offset:])
		if valLen < 0 {
			return false
		}
		if typ == protowire.BytesType {
			sub, n := protowire.ConsumeBytes(csBytes[offset : offset+valLen])
			if n < 0 {
				return false
			}
			switch num {
			case containerStatusFieldAllocatedResources:
				if !isDefaultedResourceEntryWire(sub) {
					return false
				}
			case containerStatusFieldResources:
				if !isDefaultedResourceReqWire(sub, false) {
					return false
				}
			}
		}
		offset += valLen
	}
	return true
}

func isDefaultedResourceReqWire(resBytes []byte, requireLimitsInRequests bool) bool {
	hasLimits := false
	offset := 0
	for offset < len(resBytes) {
		num, typ, tagLen := protowire.ConsumeTag(resBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, resBytes[offset:])
		if valLen < 0 {
			return false
		}
		if typ == protowire.BytesType && (num == resReqFieldLimits || num == resReqFieldRequests) {
			entry, n := protowire.ConsumeBytes(resBytes[offset : offset+valLen])
			if n < 0 || !isDefaultedResourceEntryWire(entry) {
				return false
			}
			if num == resReqFieldLimits {
				hasLimits = true
			}
		}
		offset += valLen
	}
	if !requireLimitsInRequests || !hasLimits {
		return true
	}
	// Verify every key present in limits also exists in requests (SetDefaults_Pod copies missing requests from limits).
	offset = 0
	for offset < len(resBytes) {
		num, typ, tagLen := protowire.ConsumeTag(resBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, resBytes[offset:])
		if valLen < 0 {
			return false
		}
		if num == resReqFieldLimits && typ == protowire.BytesType {
			entry, _ := protowire.ConsumeBytes(resBytes[offset : offset+valLen])
			limitKey, ok := scanSingleStringFieldBytes(entry, mapEntryFieldKey)
			if !ok || !resourceListHasKey(resBytes, resReqFieldRequests, limitKey) {
				return false
			}
		}
		offset += valLen
	}
	return true
}

func resourceListHasKey(resBytes []byte, listTag protowire.Number, targetKey []byte) bool {
	offset := 0
	for offset < len(resBytes) {
		num, typ, tagLen := protowire.ConsumeTag(resBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, resBytes[offset:])
		if valLen < 0 {
			return false
		}
		if num == listTag && typ == protowire.BytesType {
			entry, n := protowire.ConsumeBytes(resBytes[offset : offset+valLen])
			if n >= 0 {
				k, ok := scanSingleStringFieldBytes(entry, mapEntryFieldKey)
				if ok && bytes.Equal(k, targetKey) {
					return true
				}
			}
		}
		offset += valLen
	}
	return false
}

func isDefaultedResourceEntryWire(entryBytes []byte) bool {
	offset := 0
	for offset < len(entryBytes) {
		num, typ, tagLen := protowire.ConsumeTag(entryBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, entryBytes[offset:])
		if valLen < 0 {
			return false
		}
		if num == mapEntryFieldValue && typ == protowire.BytesType {
			qtyMsg, n := protowire.ConsumeBytes(entryBytes[offset : offset+valLen])
			if n < 0 {
				return false
			}
			qtyStr, ok := scanSingleStringFieldBytes(qtyMsg, quantityFieldString)
			if !ok || quantityMayNeedRounding(qtyStr) {
				return false
			}
		}
		offset += valLen
	}
	return true
}

func quantityMayNeedRounding(qty []byte) bool {
	if len(qty) == 0 {
		return true
	}
	for _, c := range qty {
		switch c {
		case '.', 'n', 'u', 'e', 'E':
			return true
		}
	}
	return false
}

func isDefaultedVolumeWire(volBytes []byte, imageVolEnabled bool) bool {
	var vsBytes []byte
	var seenSource bool

	offset := 0
	for offset < len(volBytes) {
		num, typ, tagLen := protowire.ConsumeTag(volBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, volBytes[offset:])
		if valLen < 0 {
			return false
		}
		if num == volumeFieldSource && typ == protowire.BytesType {
			sub, n := protowire.ConsumeBytes(volBytes[offset : offset+valLen])
			if n < 0 {
				return false
			}
			vsBytes = sub
			seenSource = true
		}
		offset += valLen
	}
	if !seenSource || len(vsBytes) == 0 {
		return false
	}

	hasAnyField := false
	offset = 0
	for offset < len(vsBytes) {
		num, typ, tagLen := protowire.ConsumeTag(vsBytes[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, vsBytes[offset:])
		if valLen < 0 {
			return false
		}
		hasAnyField = true
		if typ == protowire.BytesType {
			sub, n := protowire.ConsumeBytes(vsBytes[offset : offset+valLen])
			if n < 0 {
				return false
			}
			switch num {
			case volSrcFieldHostPath:
				if !hasField(sub, hostPathFieldType) {
					return false
				}
			case volSrcFieldSecret:
				if !hasField(sub, secretVolFieldDefaultMode) {
					return false
				}
			case volSrcFieldConfigMap:
				if !hasField(sub, configMapVolFieldDefaultMode) {
					return false
				}
			case volSrcFieldDownwardAPI:
				if !hasField(sub, downwardVolFieldDefaultMode) || !isDefaultedDownwardItemsWire(sub) {
					return false
				}
			case volSrcFieldProjected:
				if !hasField(sub, projectedVolFieldDefaultMode) || !isDefaultedProjectedWire(sub) {
					return false
				}
			case volSrcFieldISCSI, volSrcFieldRBD, volSrcFieldAzureDisk, volSrcFieldScaleIO, volSrcFieldEphemeral:
				return false
			case volSrcFieldImage:
				if imageVolEnabled {
					return false
				}
			}
		}
		offset += valLen
	}
	return hasAnyField
}

func isDefaultedDownwardItemsWire(b []byte) bool {
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return false
		}
		if num == downwardVolFieldItems && typ == protowire.BytesType {
			item, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 {
				return false
			}
			if fr, ok := subMessageBytes(item, downwardFileFieldFieldRef); ok {
				apiVer, ok := scanSingleStringFieldBytes(fr, fieldSelectorFieldAPIVer)
				if !ok || len(apiVer) == 0 {
					return false
				}
			}
		}
		offset += valLen
	}
	return true
}

func isDefaultedProjectedWire(b []byte) bool {
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return false
		}
		if num == projectedVolFieldSources && typ == protowire.BytesType {
			src, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 {
				return false
			}
			if da, ok := subMessageBytes(src, volProjFieldDownwardAPI); ok {
				if !isDefaultedDownwardItemsWire(da) {
					return false
				}
			}
			if tok, ok := subMessageBytes(src, volProjFieldSAToken); ok {
				if !hasField(tok, saTokenProjFieldExpirationSecs) {
					return false
				}
			}
		}
		offset += valLen
	}
	return true
}

// --- Wire field readers ---

func hasField(b []byte, target protowire.Number) bool {
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return false
		}
		if num == target {
			return true
		}
		offset += valLen
	}
	return false
}

func subMessageBytes(b []byte, target protowire.Number) ([]byte, bool) {
	var found []byte
	var ok bool
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return nil, false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return nil, false
		}
		if num == target && typ == protowire.BytesType {
			sub, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n >= 0 {
				found = sub
				ok = true
			}
		}
		offset += valLen
	}
	return found, ok
}

func scanSingleStringFieldBytes(b []byte, target protowire.Number) ([]byte, bool) {
	var out []byte
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return nil, false
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return nil, false
		}
		if num == target {
			if typ != protowire.BytesType {
				return nil, false
			}
			v, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n < 0 {
				return nil, false
			}
			out = v
		}
		offset += valLen
	}
	return out, true
}

func extractStringField(b []byte, target protowire.Number) string {
	v, ok := scanSingleStringFieldBytes(b, target)
	if !ok || len(v) == 0 {
		return ""
	}
	return string(v)
}

func extractBoolField(b []byte, target protowire.Number) bool {
	var out bool
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return out
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return out
		}
		if num == target && typ == protowire.VarintType {
			if v, n := protowire.ConsumeVarint(b[offset : offset+valLen]); n >= 0 {
				out = v != 0
			}
		}
		offset += valLen
	}
	return out
}

func extractInt64Field(b []byte, target protowire.Number) int64 {
	v, _ := extractOptionalInt64Field(b, target)
	return v
}

func extractOptionalInt64Field(b []byte, target protowire.Number) (int64, bool) {
	var out int64
	var found bool
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return out, found
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return out, found
		}
		if num == target && typ == protowire.VarintType {
			if v, n := protowire.ConsumeVarint(b[offset : offset+valLen]); n >= 0 {
				out = int64(v)
				found = true
			}
		}
		offset += valLen
	}
	return out, found
}

func extractStringMapField(b []byte, target protowire.Number) map[string]string {
	var out map[string]string
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return out
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return out
		}
		if num == target && typ == protowire.BytesType {
			entry, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n >= 0 {
				k, v := parseStringMapEntry(entry)
				if out == nil {
					out = make(map[string]string)
				}
				out[k] = v
			}
		}
		offset += valLen
	}
	return out
}

func parseStringMapEntry(entry []byte) (key, val string) {
	offset := 0
	for offset < len(entry) {
		num, typ, tagLen := protowire.ConsumeTag(entry[offset:])
		if tagLen < 0 {
			return key, val
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, entry[offset:])
		if valLen < 0 {
			return key, val
		}
		if typ == protowire.BytesType {
			v, n := protowire.ConsumeBytes(entry[offset : offset+valLen])
			if n >= 0 {
				switch num {
				case mapEntryFieldKey:
					key = string(v)
				case mapEntryFieldValue:
					val = string(v)
				}
			}
		}
		offset += valLen
	}
	return key, val
}

func extractRepeatedStringField(b []byte, target protowire.Number) []string {
	var out []string
	offset := 0
	for offset < len(b) {
		num, typ, tagLen := protowire.ConsumeTag(b[offset:])
		if tagLen < 0 {
			return out
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, b[offset:])
		if valLen < 0 {
			return out
		}
		if num == target && typ == protowire.BytesType {
			v, n := protowire.ConsumeBytes(b[offset : offset+valLen])
			if n >= 0 {
				out = append(out, string(v))
			}
		}
		offset += valLen
	}
	return out
}

func extractPodIPs(statusBytes []byte) []core.PodIP {
	var out []core.PodIP
	offset := 0
	for offset < len(statusBytes) {
		num, typ, tagLen := protowire.ConsumeTag(statusBytes[offset:])
		if tagLen < 0 {
			return out
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, statusBytes[offset:])
		if valLen < 0 {
			return out
		}
		if num == statusFieldPodIPs && typ == protowire.BytesType {
			entry, n := protowire.ConsumeBytes(statusBytes[offset : offset+valLen])
			if n >= 0 {
				out = append(out, core.PodIP{IP: extractStringField(entry, ipEntryFieldIP)})
			}
		}
		offset += valLen
	}
	return out
}

func extractHostIPs(statusBytes []byte) []core.HostIP {
	var out []core.HostIP
	offset := 0
	for offset < len(statusBytes) {
		num, typ, tagLen := protowire.ConsumeTag(statusBytes[offset:])
		if tagLen < 0 {
			return out
		}
		offset += tagLen
		valLen := protowire.ConsumeFieldValue(num, typ, statusBytes[offset:])
		if valLen < 0 {
			return out
		}
		if num == statusFieldHostIPs && typ == protowire.BytesType {
			entry, n := protowire.ConsumeBytes(statusBytes[offset : offset+valLen])
			if n >= 0 {
				out = append(out, core.HostIP{IP: extractStringField(entry, ipEntryFieldIP)})
			}
		}
		offset += valLen
	}
	return out
}
