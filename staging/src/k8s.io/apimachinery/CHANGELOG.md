This file documents Go API changes in apimachinery.

Breaking API changes *must* be documented here, together with instructions on
how to deal with them or why they are expected to have no impact.

Go API changes are typically not included in the Kubernetes release notes, so
non-breaking noteworthy Go API changes *may* be documented here if they are
useful to know about for developers.

### KEP-4222: Support CBOR encoding for non-resource endpoints.

See [PR #139632](https://github.com/kubernetes/kubernetes/pull/139632).

```
- ./pkg/runtime.UseNondeterministicEncoding: changed from func(Encoder) Encoder to func(Serializer) Encoder
```

### Introduce Deferred Gen concept to Validation-gen framework

See [PR #138205](https://github.com/kubernetes/kubernetes/pull/138205).

```
- ./pkg/api/validate.IfOption: changed from func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T, string, bool, func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T) k8s.io/apimachinery/pkg/util/validation/field.ErrorList) k8s.io/apimachinery/pkg/util/validation/field.ErrorList to func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, T, T, string, bool, func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, T, T) k8s.io/apimachinery/pkg/util/validation/field.ErrorList) k8s.io/apimachinery/pkg/util/validation/field.ErrorList
```

### Revert DV native error matcher and errors

See [PR #137017](https://github.com/kubernetes/kubernetes/pull/137017).

```
- ./pkg/util/validation/field.(*Error).MarkDeclarativeNative: removed
- ./pkg/util/validation/field.Error.DeclarativeNative: removed
- ./pkg/util/validation/field.ErrorList.MarkDeclarativeNative: removed
- ./pkg/util/validation/field.ErrorMatcher.ByDeclarativeNative: removed
```

### KEP-4671: Add Declarative Validation to Workload API

See [PR #135164](https://github.com/kubernetes/kubernetes/pull/135164).

```
- ./pkg/api/validation/path.NameMayNotBe: removed
- ./pkg/api/validation/path.NameMayNotContain: removed
```

### KEP-5589 - drop gogo runtime dependencies

See [PR #134256](https://github.com/kubernetes/kubernetes/pull/134256).

```
- ./pkg/api/resource.(*Quantity).Descriptor: removed
- ./pkg/api/resource.(*Quantity).ProtoMessage: removed
- ./pkg/api/resource.(*Quantity).XXX_DiscardUnknown: removed
- ./pkg/api/resource.(*Quantity).XXX_Marshal: removed
- ./pkg/api/resource.(*Quantity).XXX_Merge: removed
- ./pkg/api/resource.(*Quantity).XXX_Size: removed
- ./pkg/api/resource.(*Quantity).XXX_Unmarshal: removed
- ./pkg/api/resource.(*QuantityValue).Descriptor: removed
- ./pkg/api/resource.(*QuantityValue).ProtoMessage: removed
- ./pkg/api/resource.(*QuantityValue).XXX_DiscardUnknown: removed
- ./pkg/api/resource.(*QuantityValue).XXX_Marshal: removed
- ./pkg/api/resource.(*QuantityValue).XXX_Merge: removed
- ./pkg/api/resource.(*QuantityValue).XXX_Size: removed
- ./pkg/api/resource.(*QuantityValue).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*APIGroup).Descriptor: removed
- ./pkg/apis/meta/v1.(*APIGroup).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*APIGroup).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*APIGroup).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*APIGroup).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*APIGroup).XXX_Size: removed
- ./pkg/apis/meta/v1.(*APIGroup).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*APIGroupList).Descriptor: removed
- ./pkg/apis/meta/v1.(*APIGroupList).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*APIGroupList).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*APIGroupList).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*APIGroupList).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*APIGroupList).XXX_Size: removed
- ./pkg/apis/meta/v1.(*APIGroupList).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*APIResource).Descriptor: removed
- ./pkg/apis/meta/v1.(*APIResource).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*APIResource).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*APIResource).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*APIResource).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*APIResource).XXX_Size: removed
- ./pkg/apis/meta/v1.(*APIResource).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*APIResourceList).Descriptor: removed
- ./pkg/apis/meta/v1.(*APIResourceList).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*APIResourceList).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*APIResourceList).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*APIResourceList).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*APIResourceList).XXX_Size: removed
- ./pkg/apis/meta/v1.(*APIResourceList).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*APIVersions).Descriptor: removed
- ./pkg/apis/meta/v1.(*APIVersions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*APIVersions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*APIVersions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*APIVersions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*APIVersions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*APIVersions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*ApplyOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Condition).Descriptor: removed
- ./pkg/apis/meta/v1.(*Condition).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Condition).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Condition).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Condition).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Condition).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Condition).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*CreateOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*CreateOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*CreateOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*CreateOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*CreateOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*CreateOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*CreateOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*DeleteOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Duration).Descriptor: removed
- ./pkg/apis/meta/v1.(*Duration).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Duration).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Duration).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Duration).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Duration).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Duration).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).Descriptor: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).XXX_Size: removed
- ./pkg/apis/meta/v1.(*FieldSelectorRequirement).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*FieldsV1).Descriptor: removed
- ./pkg/apis/meta/v1.(*FieldsV1).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*FieldsV1).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*FieldsV1).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*FieldsV1).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*FieldsV1).XXX_Size: removed
- ./pkg/apis/meta/v1.(*FieldsV1).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GetOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*GetOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GetOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GetOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GetOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GetOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GetOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GroupKind).Descriptor: removed
- ./pkg/apis/meta/v1.(*GroupKind).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GroupKind).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GroupKind).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GroupKind).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GroupKind).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GroupKind).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GroupResource).Descriptor: removed
- ./pkg/apis/meta/v1.(*GroupResource).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GroupResource).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GroupResource).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GroupResource).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GroupResource).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GroupResource).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GroupVersion).Descriptor: removed
- ./pkg/apis/meta/v1.(*GroupVersion).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GroupVersion).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GroupVersion).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GroupVersion).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GroupVersion).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GroupVersion).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).Descriptor: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GroupVersionForDiscovery).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).Descriptor: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GroupVersionKind).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).Descriptor: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).XXX_Size: removed
- ./pkg/apis/meta/v1.(*GroupVersionResource).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*LabelSelector).Descriptor: removed
- ./pkg/apis/meta/v1.(*LabelSelector).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*LabelSelector).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*LabelSelector).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*LabelSelector).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*LabelSelector).XXX_Size: removed
- ./pkg/apis/meta/v1.(*LabelSelector).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).Descriptor: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).XXX_Size: removed
- ./pkg/apis/meta/v1.(*LabelSelectorRequirement).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*List).Descriptor: removed
- ./pkg/apis/meta/v1.(*List).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*List).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*List).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*List).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*List).XXX_Size: removed
- ./pkg/apis/meta/v1.(*List).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*ListMeta).Descriptor: removed
- ./pkg/apis/meta/v1.(*ListMeta).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*ListMeta).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*ListMeta).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*ListMeta).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*ListMeta).XXX_Size: removed
- ./pkg/apis/meta/v1.(*ListMeta).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*ListOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*ListOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*ListOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*ListOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*ListOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*ListOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*ListOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).Descriptor: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).XXX_Size: removed
- ./pkg/apis/meta/v1.(*ManagedFieldsEntry).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*MicroTime).Descriptor: removed
- ./pkg/apis/meta/v1.(*MicroTime).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*MicroTime).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*MicroTime).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*MicroTime).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*MicroTime).XXX_Size: removed
- ./pkg/apis/meta/v1.(*MicroTime).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).Descriptor: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).XXX_Size: removed
- ./pkg/apis/meta/v1.(*ObjectMeta).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*OwnerReference).Descriptor: removed
- ./pkg/apis/meta/v1.(*OwnerReference).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*OwnerReference).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*OwnerReference).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*OwnerReference).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*OwnerReference).XXX_Size: removed
- ./pkg/apis/meta/v1.(*OwnerReference).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).Descriptor: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).XXX_Size: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadata).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).Descriptor: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).XXX_Size: removed
- ./pkg/apis/meta/v1.(*PartialObjectMetadataList).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Patch).Descriptor: removed
- ./pkg/apis/meta/v1.(*Patch).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Patch).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Patch).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Patch).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Patch).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Patch).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*PatchOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*PatchOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*PatchOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*PatchOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*PatchOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*PatchOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*PatchOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Preconditions).Descriptor: removed
- ./pkg/apis/meta/v1.(*Preconditions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Preconditions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Preconditions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Preconditions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Preconditions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Preconditions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*RootPaths).Descriptor: removed
- ./pkg/apis/meta/v1.(*RootPaths).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*RootPaths).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*RootPaths).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*RootPaths).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*RootPaths).XXX_Size: removed
- ./pkg/apis/meta/v1.(*RootPaths).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).Descriptor: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).XXX_Size: removed
- ./pkg/apis/meta/v1.(*ServerAddressByClientCIDR).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Status).Descriptor: removed
- ./pkg/apis/meta/v1.(*Status).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Status).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Status).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Status).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Status).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Status).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*StatusCause).Descriptor: removed
- ./pkg/apis/meta/v1.(*StatusCause).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*StatusCause).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*StatusCause).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*StatusCause).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*StatusCause).XXX_Size: removed
- ./pkg/apis/meta/v1.(*StatusCause).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*StatusDetails).Descriptor: removed
- ./pkg/apis/meta/v1.(*StatusDetails).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*StatusDetails).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*StatusDetails).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*StatusDetails).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*StatusDetails).XXX_Size: removed
- ./pkg/apis/meta/v1.(*StatusDetails).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*TableOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*TableOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*TableOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*TableOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*TableOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*TableOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*TableOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Time).Descriptor: removed
- ./pkg/apis/meta/v1.(*Time).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Time).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Time).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Time).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Time).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Time).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Timestamp).Descriptor: removed
- ./pkg/apis/meta/v1.(*Timestamp).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Timestamp).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Timestamp).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Timestamp).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Timestamp).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Timestamp).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*TypeMeta).Descriptor, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).Descriptor: removed
- ./pkg/apis/meta/v1.(*TypeMeta).ProtoMessage, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_DiscardUnknown, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Marshal, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Merge, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Size, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Size: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Unmarshal, method set of *ListOptions: removed
- ./pkg/apis/meta/v1.(*TypeMeta).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).Descriptor: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).XXX_Size: removed
- ./pkg/apis/meta/v1.(*UpdateOptions).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*Verbs).Descriptor: removed
- ./pkg/apis/meta/v1.(*Verbs).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*Verbs).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*Verbs).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*Verbs).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*Verbs).XXX_Size: removed
- ./pkg/apis/meta/v1.(*Verbs).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1.(*WatchEvent).Descriptor: removed
- ./pkg/apis/meta/v1.(*WatchEvent).ProtoMessage: removed
- ./pkg/apis/meta/v1.(*WatchEvent).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1.(*WatchEvent).XXX_Marshal: removed
- ./pkg/apis/meta/v1.(*WatchEvent).XXX_Merge: removed
- ./pkg/apis/meta/v1.(*WatchEvent).XXX_Size: removed
- ./pkg/apis/meta/v1.(*WatchEvent).XXX_Unmarshal: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).Descriptor: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).ProtoMessage: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).XXX_DiscardUnknown: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).XXX_Marshal: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).XXX_Merge: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).XXX_Size: removed
- ./pkg/apis/meta/v1beta1.(*PartialObjectMetadataList).XXX_Unmarshal: removed
- ./pkg/apis/testapigroup/v1.(*Carp).Descriptor: removed
- ./pkg/apis/testapigroup/v1.(*Carp).ProtoMessage: removed
- ./pkg/apis/testapigroup/v1.(*Carp).XXX_DiscardUnknown: removed
- ./pkg/apis/testapigroup/v1.(*Carp).XXX_Marshal: removed
- ./pkg/apis/testapigroup/v1.(*Carp).XXX_Merge: removed
- ./pkg/apis/testapigroup/v1.(*Carp).XXX_Size: removed
- ./pkg/apis/testapigroup/v1.(*Carp).XXX_Unmarshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).Descriptor: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).ProtoMessage: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).XXX_DiscardUnknown: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).XXX_Marshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).XXX_Merge: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).XXX_Size: removed
- ./pkg/apis/testapigroup/v1.(*CarpCondition).XXX_Unmarshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).Descriptor: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).ProtoMessage: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).XXX_DiscardUnknown: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).XXX_Marshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).XXX_Merge: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).XXX_Size: removed
- ./pkg/apis/testapigroup/v1.(*CarpInfo).XXX_Unmarshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).Descriptor: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).ProtoMessage: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).XXX_DiscardUnknown: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).XXX_Marshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).XXX_Merge: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).XXX_Size: removed
- ./pkg/apis/testapigroup/v1.(*CarpList).XXX_Unmarshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).Descriptor: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).ProtoMessage: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).XXX_DiscardUnknown: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).XXX_Marshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).XXX_Merge: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).XXX_Size: removed
- ./pkg/apis/testapigroup/v1.(*CarpSpec).XXX_Unmarshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).Descriptor: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).ProtoMessage: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).XXX_DiscardUnknown: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).XXX_Marshal: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).XXX_Merge: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).XXX_Size: removed
- ./pkg/apis/testapigroup/v1.(*CarpStatus).XXX_Unmarshal: removed
- ./pkg/runtime.(*RawExtension).Descriptor: removed
- ./pkg/runtime.(*RawExtension).ProtoMessage: removed
- ./pkg/runtime.(*RawExtension).XXX_DiscardUnknown: removed
- ./pkg/runtime.(*RawExtension).XXX_Marshal: removed
- ./pkg/runtime.(*RawExtension).XXX_Merge: removed
- ./pkg/runtime.(*RawExtension).XXX_Size: removed
- ./pkg/runtime.(*RawExtension).XXX_Unmarshal: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).Descriptor, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).Descriptor: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).ProtoMessage: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_DiscardUnknown: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Marshal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Merge: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Size: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *EmbeddedTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *EmbeddedTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExtensionA: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExtensionB: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ExternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *InternalComplex: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *InternalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *InternalOptionalExtensionType: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *InternalSimple: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ObjectTest: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal, method set of *ObjectTestExternal: removed
- ./pkg/runtime.(*TypeMeta).XXX_Unmarshal: removed
- ./pkg/runtime.(*Unknown).Descriptor: removed
- ./pkg/runtime.(*Unknown).ProtoMessage: removed
- ./pkg/runtime.(*Unknown).XXX_DiscardUnknown: removed
- ./pkg/runtime.(*Unknown).XXX_Marshal: removed
- ./pkg/runtime.(*Unknown).XXX_Merge: removed
- ./pkg/runtime.(*Unknown).XXX_Size: removed
- ./pkg/runtime.(*Unknown).XXX_Unmarshal: removed
- ./pkg/util/intstr.(*IntOrString).Descriptor: removed
- ./pkg/util/intstr.(*IntOrString).ProtoMessage: removed
- ./pkg/util/intstr.(*IntOrString).XXX_DiscardUnknown: removed
- ./pkg/util/intstr.(*IntOrString).XXX_Marshal: removed
- ./pkg/util/intstr.(*IntOrString).XXX_Merge: removed
- ./pkg/util/intstr.(*IntOrString).XXX_Size: removed
- ./pkg/util/intstr.(*IntOrString).XXX_Unmarshal: removed
```

### feat(validation-gen): Add "cohorts" & Tighten and simplify test framework

See [PR #134347](https://github.com/kubernetes/kubernetes/pull/134347).

```
- ./pkg/util/validation/field.TestIntf.Logf: removed
```

### gogo protobuf dependency cleanup

See [PR #134228](https://github.com/kubernetes/kubernetes/pull/134228).

```
- ./pkg/runtime.ProtobufMarshaller.Size: added
- ./pkg/runtime.ProtobufReverseMarshaller.Size: added
```

### Add  +k8s:ifEnabled, +k8s:ifDisabled and +k8s:enumExclude tags

See [PR #133768](https://github.com/kubernetes/kubernetes/pull/133768).

```
- ./pkg/api/validate.Enum: changed from func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T, k8s.io/apimachinery/pkg/util/sets.Set[T]) k8s.io/apimachinery/pkg/util/validation/field.ErrorList to func(context.Context, k8s.io/apimachinery/pkg/api/operation.Operation, *k8s.io/apimachinery/pkg/util/validation/field.Path, *T, *T, k8s.io/apimachinery/pkg/util/sets.Set[T], []EnumExclusion[T]) k8s.io/apimachinery/pkg/util/validation/field.ErrorList
```

### Changes for Kubernetes <= 1.34

For older changes refer to the commit messages and PR descriptions.
