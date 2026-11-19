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
	"fmt"
	"io"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
	"k8s.io/apimachinery/pkg/types"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	core "k8s.io/kubernetes/pkg/apis/core"
)

var (
	v1PodGVK = v1.SchemeGroupVersion.WithKind("Pod")

	// v1ProtobufEncoderID matches the Identifier produced by versioning.NewCodec
	// wrapping protobuf.Serializer for core/v1 responses (GET, LIST, and WATCH).
	v1ProtobufEncoderID = versioning.NewCodec(
		protobuf.NewSerializer(nil, nil),
		nil, nil, nil, nil, nil,
		v1.SchemeGroupVersion,
		nil,
		"",
	).Identifier()
)

const watchEncoderIdentifierPrefix = `{"name":"watch",`

// Pod is a read-optimized Pod representation backed by immutable v1.Pod Protobuf
// wire bytes. It implements runtime.Object, metav1.Object, podutil.PodAccessor,
// and runtime.CacheableObject so GET, LIST, and WATCH responses in Protobuf format
// can write pre-framed wire bytes directly without decoding or re-encoding structs.
//
// Shallow struct copies (such as those made by Cacher.Get, Cacher.GetList, and
// cacheWatcher.getShallowCopyObject) remain race-free: wire is strictly immutable,
// and any mutation via metav1.Object setters detaches into a privately owned
// *core.Pod via owner self-pointer tracking.
type Pod struct {
	metav1.TypeMeta
	wire     *wirePod
	fallback *core.Pod
	owner    *Pod
}

var (
	_ runtime.Object          = (*Pod)(nil)
	_ metav1.Object           = (*Pod)(nil)
	_ podutil.PodAccessor     = (*Pod)(nil)
	_ runtime.CacheableObject = (*Pod)(nil)
)

// NewFromCorePod wraps a concrete *core.Pod in a Pod. Ownership of pod transfers
// to the returned Pod (callers must not mutate pod afterwards).
func NewFromCorePod(pod *core.Pod) *Pod {
	p := &Pod{fallback: pod}
	p.owner = p
	return p
}

// IsWireBacked reports whether p currently holds immutable Protobuf wire bytes
// instead of a decoded *core.Pod fallback.
func (p *Pod) IsWireBacked() bool {
	return p != nil && p.wire != nil && p.fallback == nil
}

// SetObject populates p from an existing *core.Pod (used by Cacher if the cache
// contains concrete *core.Pod values, e.g. when WatchList falls back to legacy LIST).
// The wrapped *core.Pod is treated as shared/read-only (owner is cleared) so any
// subsequent mutation copies before write.
func (p *Pod) SetObject(obj runtime.Object) bool {
	switch src := obj.(type) {
	case *Pod:
		*p = *src
		p.owner = nil
		return true
	case *core.Pod:
		*p = Pod{TypeMeta: src.TypeMeta, fallback: src}
		return true
	default:
		return false
	}
}

// DeepCopyObject implements runtime.Object. For wire-backed pods this only
// allocates a shallow Pod wrapper sharing the immutable wirePod buffer.
func (p *Pod) DeepCopyObject() runtime.Object {
	if p == nil {
		return nil
	}
	if p.fallback == nil && p.wire != nil {
		return &Pod{TypeMeta: p.TypeMeta, wire: p.wire}
	}
	if p.fallback != nil {
		cp := &Pod{TypeMeta: p.TypeMeta, fallback: p.fallback.DeepCopy()}
		cp.owner = cp
		return cp
	}
	return &Pod{TypeMeta: p.TypeMeta}
}

// GetObject implements runtime.CacheableObject, returning a freshly owned
// *core.Pod deep copy so callers that expect a concrete *core.Pod (writes,
// Table printers, JSON/CBOR/YAML encoders) remain decoupled from wire storage.
func (p *Pod) GetObject() runtime.Object {
	if p == nil {
		return nil
	}
	return p.ToCorePod()
}

// ToCorePod decodes p into a freshly owned *core.Pod.
func (p *Pod) ToCorePod() *core.Pod {
	if p == nil {
		return nil
	}
	if p.fallback != nil {
		return p.fallback.DeepCopy()
	}
	if p.wire == nil {
		return &core.Pod{TypeMeta: p.TypeMeta}
	}
	decoded, err := p.wire.decodeToCorePod()
	if err != nil {
		// wirePod was already validated during parseWirePod; an unmarshal error
		// here would indicate memory corruption or an invalid wire splice.
		panic(fmt.Sprintf("lazypod: failed to decode validated wire pod: %v", err))
	}
	decoded.TypeMeta = p.TypeMeta
	return decoded
}

// CacheEncode implements runtime.CacheableObject. For core/v1 Protobuf responses
// it writes or splices the pre-built envelope directly without struct conversion
// or serializer buffer allocation.
func (p *Pod) CacheEncode(id runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	if p.IsWireBacked() {
		if id == v1ProtobufEncoderID {
			if splicer, ok := w.(runtime.Splice); ok {
				splicer.Splice(p.wire.envelope)
				return nil
			}
			_, err := w.Write(p.wire.envelope)
			return err
		}
		if strings.HasPrefix(string(id), watchEncoderIdentifierPrefix) {
			// Pass p directly so watchEncoder.doEncode reaches watchEmbeddedEncoder,
			// which then invokes CacheEncode with the inner serializer identifier.
			return encode(p, w)
		}
	}
	return encode(p.GetObject(), w)
}

func (p *Pod) materializeMutable() *core.Pod {
	if p.fallback != nil {
		if p.owner != p {
			p.fallback = p.fallback.DeepCopy()
			p.owner = p
		}
		return p.fallback
	}
	p.fallback = p.ToCorePod()
	p.wire = nil
	p.owner = p
	return p.fallback
}

func (p *Pod) decodeMetaFallback() metav1.ObjectMeta {
	if p.wire == nil {
		return metav1.ObjectMeta{}
	}
	meta, err := p.wire.decodeObjectMeta()
	if err != nil {
		panic(fmt.Sprintf("lazypod: failed to decode validated ObjectMeta: %v", err))
	}
	return meta
}

// --- metav1.Object getters and setters ---

func (p *Pod) GetNamespace() string {
	if p.fallback != nil {
		return p.fallback.Namespace
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.metaBytes(), metaFieldNamespace)
}

func (p *Pod) SetNamespace(namespace string) {
	if p.fallback == nil && p.wire != nil && p.GetNamespace() == namespace {
		return
	}
	p.materializeMutable().Namespace = namespace
}

func (p *Pod) GetName() string {
	if p.fallback != nil {
		return p.fallback.Name
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.metaBytes(), metaFieldName)
}

func (p *Pod) SetName(name string) {
	if p.fallback == nil && p.wire != nil && p.GetName() == name {
		return
	}
	p.materializeMutable().Name = name
}

func (p *Pod) GetGenerateName() string {
	if p.fallback != nil {
		return p.fallback.GenerateName
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.metaBytes(), metaFieldGenerateName)
}

func (p *Pod) SetGenerateName(name string) {
	p.materializeMutable().GenerateName = name
}

func (p *Pod) GetUID() types.UID {
	if p.fallback != nil {
		return p.fallback.UID
	}
	if p.wire == nil {
		return ""
	}
	return types.UID(extractStringField(p.wire.metaBytes(), metaFieldUID))
}

func (p *Pod) SetUID(uid types.UID) {
	p.materializeMutable().UID = uid
}

func (p *Pod) GetResourceVersion() string {
	if p.fallback != nil {
		return p.fallback.ResourceVersion
	}
	if p.wire == nil {
		return ""
	}
	return p.wire.rv
}

func (p *Pod) SetResourceVersion(version string) {
	if p.fallback != nil {
		if p.fallback.ResourceVersion == version {
			return
		}
		p.materializeMutable().ResourceVersion = version
		return
	}
	if p.wire == nil {
		p.materializeMutable().ResourceVersion = version
		return
	}
	if p.wire.rv == version {
		return
	}
	p.wire = p.wire.withResourceVersion(version)
}

func (p *Pod) GetGeneration() int64 {
	if p.fallback != nil {
		return p.fallback.Generation
	}
	if p.wire == nil {
		return 0
	}
	return extractInt64Field(p.wire.metaBytes(), metaFieldGeneration)
}

func (p *Pod) SetGeneration(generation int64) {
	p.materializeMutable().Generation = generation
}

func (p *Pod) GetSelfLink() string {
	if p.fallback != nil {
		return p.fallback.SelfLink
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.metaBytes(), metaFieldSelfLink)
}

func (p *Pod) SetSelfLink(selfLink string) {
	p.materializeMutable().SelfLink = selfLink
}

func (p *Pod) GetCreationTimestamp() metav1.Time {
	if p.fallback != nil {
		return p.fallback.CreationTimestamp
	}
	return p.decodeMetaFallback().CreationTimestamp
}

func (p *Pod) SetCreationTimestamp(timestamp metav1.Time) {
	p.materializeMutable().CreationTimestamp = timestamp
}

func (p *Pod) GetDeletionTimestamp() *metav1.Time {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetDeletionTimestamp()
	}
	if p.wire == nil || !hasField(p.wire.metaBytes(), metaFieldDeletionTimestamp) {
		return nil
	}
	return p.decodeMetaFallback().DeletionTimestamp
}

func (p *Pod) SetDeletionTimestamp(timestamp *metav1.Time) {
	podutil.MustAccessor(p.materializeMutable()).SetDeletionTimestamp(timestamp)
}

func (p *Pod) GetDeletionGracePeriodSeconds() *int64 {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetDeletionGracePeriodSeconds()
	}
	if p.wire == nil {
		return nil
	}
	v, ok := extractOptionalInt64Field(p.wire.metaBytes(), metaFieldDeletionGracePeriodSeconds)
	if !ok {
		return nil
	}
	return &v
}

func (p *Pod) SetDeletionGracePeriodSeconds(gracePeriodSeconds *int64) {
	podutil.MustAccessor(p.materializeMutable()).SetDeletionGracePeriodSeconds(gracePeriodSeconds)
}

func (p *Pod) GetLabels() map[string]string {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetLabels()
	}
	if p.wire == nil {
		return nil
	}
	return extractStringMapField(p.wire.metaBytes(), metaFieldLabels)
}

func (p *Pod) SetLabels(labels map[string]string) {
	podutil.MustAccessor(p.materializeMutable()).SetLabels(labels)
}

func (p *Pod) GetAnnotations() map[string]string {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetAnnotations()
	}
	if p.wire == nil {
		return nil
	}
	return extractStringMapField(p.wire.metaBytes(), metaFieldAnnotations)
}

func (p *Pod) SetAnnotations(annotations map[string]string) {
	podutil.MustAccessor(p.materializeMutable()).SetAnnotations(annotations)
}

func (p *Pod) GetFinalizers() []string {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetFinalizers()
	}
	if p.wire == nil {
		return nil
	}
	return extractRepeatedStringField(p.wire.metaBytes(), metaFieldFinalizers)
}

func (p *Pod) SetFinalizers(finalizers []string) {
	podutil.MustAccessor(p.materializeMutable()).SetFinalizers(finalizers)
}

func (p *Pod) GetOwnerReferences() []metav1.OwnerReference {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetOwnerReferences()
	}
	if p.wire == nil || !hasField(p.wire.metaBytes(), metaFieldOwnerReferences) {
		return nil
	}
	return p.decodeMetaFallback().OwnerReferences
}

func (p *Pod) SetOwnerReferences(references []metav1.OwnerReference) {
	podutil.MustAccessor(p.materializeMutable()).SetOwnerReferences(references)
}

func (p *Pod) GetManagedFields() []metav1.ManagedFieldsEntry {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetManagedFields()
	}
	if p.wire == nil || !hasField(p.wire.metaBytes(), metaFieldManagedFields) {
		return nil
	}
	return p.decodeMetaFallback().ManagedFields
}

func (p *Pod) SetManagedFields(managedFields []metav1.ManagedFieldsEntry) {
	podutil.MustAccessor(p.materializeMutable()).SetManagedFields(managedFields)
}

// --- podutil.PodAccessor getters ---

func (p *Pod) GetSpecInitContainers() []core.Container {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetSpecInitContainers()
	}
	if p.wire == nil || !hasField(p.wire.specBytes(), specFieldInitContainers) {
		return nil
	}
	return p.ToCorePod().Spec.InitContainers
}

func (p *Pod) GetSpecContainers() []core.Container {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetSpecContainers()
	}
	if p.wire == nil || !hasField(p.wire.specBytes(), specFieldContainers) {
		return nil
	}
	return p.ToCorePod().Spec.Containers
}

func (p *Pod) GetSpecEphemeralContainers() []core.EphemeralContainer {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetSpecEphemeralContainers()
	}
	if p.wire == nil || !hasField(p.wire.specBytes(), specFieldEphemeralContainers) {
		return nil
	}
	return p.ToCorePod().Spec.EphemeralContainers
}

func (p *Pod) GetSpecRestartPolicy() core.RestartPolicy {
	if p.fallback != nil {
		return p.fallback.Spec.RestartPolicy
	}
	if p.wire == nil {
		return ""
	}
	return core.RestartPolicy(extractStringField(p.wire.specBytes(), specFieldRestartPolicy))
}

func (p *Pod) GetSpecTerminationGracePeriodSeconds() (int64, bool) {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetSpecTerminationGracePeriodSeconds()
	}
	if p.wire == nil {
		return 0, false
	}
	return extractOptionalInt64Field(p.wire.specBytes(), specFieldTerminationGracePeriodSeconds)
}

func (p *Pod) GetSpecServiceAccountName() string {
	if p.fallback != nil {
		return p.fallback.Spec.ServiceAccountName
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.specBytes(), specFieldServiceAccountName)
}

func (p *Pod) GetSpecNodeName() string {
	if p.fallback != nil {
		return p.fallback.Spec.NodeName
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.specBytes(), specFieldNodeName)
}

func (p *Pod) GetSpecHostNetwork() bool {
	if p.fallback != nil {
		return p.fallback.Spec.HostNetwork
	}
	if p.wire == nil {
		return false
	}
	return extractBoolField(p.wire.specBytes(), specFieldHostNetwork)
}

func (p *Pod) GetSpecAffinity() *core.Affinity {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetSpecAffinity()
	}
	if p.wire == nil || !hasField(p.wire.specBytes(), specFieldAffinity) {
		return nil
	}
	return p.ToCorePod().Spec.Affinity
}

func (p *Pod) GetSpecSchedulerName() string {
	if p.fallback != nil {
		return p.fallback.Spec.SchedulerName
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.specBytes(), specFieldSchedulerName)
}

func (p *Pod) GetStatusPhase() core.PodPhase {
	if p.fallback != nil {
		return p.fallback.Status.Phase
	}
	if p.wire == nil {
		return ""
	}
	return core.PodPhase(extractStringField(p.wire.statusBytes(), statusFieldPhase))
}

func (p *Pod) GetStatusNominatedNodeName() string {
	if p.fallback != nil {
		return p.fallback.Status.NominatedNodeName
	}
	if p.wire == nil {
		return ""
	}
	return extractStringField(p.wire.statusBytes(), statusFieldNominatedNodeName)
}

func (p *Pod) GetStatusHostIPs() []core.HostIP {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetStatusHostIPs()
	}
	if p.wire == nil {
		return nil
	}
	return extractHostIPs(p.wire.statusBytes())
}

func (p *Pod) GetStatusPodIPs() []core.PodIP {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetStatusPodIPs()
	}
	if p.wire == nil {
		return nil
	}
	return extractPodIPs(p.wire.statusBytes())
}

func (p *Pod) GetStatusContainerStatuses() []core.ContainerStatus {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetStatusContainerStatuses()
	}
	if p.wire == nil || !hasField(p.wire.statusBytes(), statusFieldContainerStatuses) {
		return nil
	}
	return p.ToCorePod().Status.ContainerStatuses
}

func (p *Pod) GetStatusResourceClaimStatuses() []core.PodResourceClaimStatus {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetStatusResourceClaimStatuses()
	}
	if p.wire == nil || !hasField(p.wire.statusBytes(), statusFieldResourceClaimStatuses) {
		return nil
	}
	return p.ToCorePod().Status.ResourceClaimStatuses
}

func (p *Pod) GetStatusExtendedResourceClaimStatus() *core.PodExtendedResourceClaimStatus {
	if p.fallback != nil {
		return podutil.MustAccessor(p.fallback).GetStatusExtendedResourceClaimStatus()
	}
	if p.wire == nil || !hasField(p.wire.statusBytes(), statusFieldExtendedResourceClaimStatus) {
		return nil
	}
	return p.ToCorePod().Status.ExtendedResourceClaimStatus
}

// PodList is a read-optimized list of lazy Pods implementing runtime.Object,
// metav1.ListInterface, and runtime.CacheableObject.
type PodList struct {
	metav1.TypeMeta
	metav1.ListMeta
	Items []Pod
}

var (
	_ runtime.Object          = (*PodList)(nil)
	_ metav1.ListInterface    = (*PodList)(nil)
	_ runtime.CacheableObject = (*PodList)(nil)
)

// DeepCopyObject implements runtime.Object.
func (l *PodList) DeepCopyObject() runtime.Object {
	if l == nil {
		return nil
	}
	out := &PodList{
		TypeMeta: l.TypeMeta,
	}
	l.ListMeta.DeepCopyInto(&out.ListMeta)
	if l.Items != nil {
		out.Items = make([]Pod, len(l.Items))
		for i := range l.Items {
			out.Items[i] = *l.Items[i].DeepCopyObject().(*Pod)
		}
	}
	return out
}

// GetObject implements runtime.CacheableObject, materializing a concrete *core.PodList.
func (l *PodList) GetObject() runtime.Object {
	if l == nil {
		return nil
	}
	out := &core.PodList{
		TypeMeta: l.TypeMeta,
	}
	l.ListMeta.DeepCopyInto(&out.ListMeta)
	if l.Items != nil {
		out.Items = make([]core.Pod, len(l.Items))
		for i := range l.Items {
			out.Items[i] = *l.Items[i].ToCorePod()
		}
	}
	return out
}

// CacheEncode implements runtime.CacheableObject. For core/v1 Protobuf responses
// where every item is wire-backed, it streams the v1.PodList Protobuf envelope
// directly to w without copying items into []core.Pod / []v1.Pod or marshaling structs.
func (l *PodList) CacheEncode(id runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	if id == v1ProtobufEncoderID && l.allItemsWireBacked() {
		return streamPodListProtobuf(w, l)
	}
	return encode(l.GetObject(), w)
}

func (l *PodList) allItemsWireBacked() bool {
	for i := range l.Items {
		if !l.Items[i].IsWireBacked() {
			return false
		}
	}
	return true
}

// StorageCodec wraps a Kubernetes storage runtime.Codec so etcd3 watch ingestion
// (into == nil) and lazy reads (into == *Pod) keep already-defaulted v1.Pod Protobuf
// payloads in wire form instead of decoding full Go structs. Any write path or caller
// passing *core.Pod delegates straight to the underlying codec unchanged.
type StorageCodec struct {
	base runtime.Codec
}

var (
	_ runtime.Codec                = (*StorageCodec)(nil)
	_ runtime.EncoderWithAllocator = (*StorageCodec)(nil)
)

// NewStorageCodec returns a runtime.Codec wrapping base with lazy Protobuf decoding for Pods.
func NewStorageCodec(base runtime.Codec) *StorageCodec {
	return &StorageCodec{base: base}
}

// Encode implements runtime.Encoder.
func (c *StorageCodec) Encode(obj runtime.Object, w io.Writer) error {
	if lp, ok := obj.(*Pod); ok {
		return c.base.Encode(lp.GetObject(), w)
	}
	return c.base.Encode(obj, w)
}

// EncodeWithAllocator implements runtime.EncoderWithAllocator.
func (c *StorageCodec) EncodeWithAllocator(obj runtime.Object, w io.Writer, memAlloc runtime.MemoryAllocator) error {
	if lp, ok := obj.(*Pod); ok {
		obj = lp.GetObject()
	}
	if ewa, ok := c.base.(runtime.EncoderWithAllocator); ok {
		return ewa.EncodeWithAllocator(obj, w, memAlloc)
	}
	return c.base.Encode(obj, w)
}

// Identifier implements runtime.Encoder.
func (c *StorageCodec) Identifier() runtime.Identifier {
	return c.base.Identifier()
}

// Decode implements runtime.Decoder.
func (c *StorageCodec) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	if into == nil {
		if wp, ok := parseWirePod(data); ok {
			gvk := v1PodGVK
			return &Pod{wire: wp}, &gvk, nil
		}
		obj, gvk, err := c.base.Decode(data, defaults, nil)
		if err != nil {
			return obj, gvk, err
		}
		if pod, ok := obj.(*core.Pod); ok {
			return NewFromCorePod(pod), gvk, nil
		}
		return obj, gvk, nil
	}

	if lp, ok := into.(*Pod); ok && lp != nil {
		if wp, ok := parseWirePod(data); ok {
			*lp = Pod{wire: wp}
			gvk := v1PodGVK
			return lp, &gvk, nil
		}
		obj, gvk, err := c.base.Decode(data, defaults, nil)
		if err != nil {
			return nil, gvk, err
		}
		pod, ok := obj.(*core.Pod)
		if !ok {
			return nil, gvk, fmt.Errorf("unexpected decoded type %T for lazypod.Pod", obj)
		}
		*lp = Pod{fallback: pod}
		lp.owner = lp
		return lp, gvk, nil
	}

	return c.base.Decode(data, defaults, into)
}
