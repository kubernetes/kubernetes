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

package constobj

import (
	"fmt"
	"io"
	"sync"

	"github.com/gogo/protobuf/proto"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/types"
)

// ConstObject implements runtime.Object and friends, sufficient to allow its use in the watch cache.
// This lets us validate that object are not mutated
type ConstObject struct {
	inner    runtime.Object
	accessor metav1.Object

	mutex        sync.Mutex
	protobufData []byte
}

var _ runtime.Object = &ConstObject{}
var _ metav1.Object = &ConstObject{}
var _ meta.ConstListObject = &ConstObject{}
var _ runtime.Encodable = &ConstObject{}

// IsConstList implements meta.ConstListObject
func (c *ConstObject) IsConstList() bool {
	return meta.IsListType(c.inner)
}

// EachListItem implements meta.ConstListObject
func (c *ConstObject) EachListItem(fn func(runtime.Object) error) error {
	return meta.EachListItem(c.inner, func(o runtime.Object) error {
		// TODO: Cache the wrapped objects?
		return fn(Constify(o))
	})
}

// GetObjectKind implements runtime.Object
func (c *ConstObject) GetObjectKind() schema.ObjectKind {
	return c.inner.GetObjectKind()
}

// DeepCopyObject implements runtime.Object
func (c *ConstObject) DeepCopyObject() runtime.Object {
	return c
	//return c.inner.DeepCopyObject()
}

// DeconstDeepCopyObject returns a mutable object, by deep-copying the inner object.  Expensive!
func (c *ConstObject) DeconstDeepCopyObject() runtime.Object {
	return c.inner.DeepCopyObject()
}

// GetNamespace implements metav1.Object
func (c *ConstObject) GetNamespace() string {
	return c.accessor.GetNamespace()
}

// SetNamespace implements metav1.Object
func (c *ConstObject) SetNamespace(namespace string) {
	panic("not implemented")
}

// GetName implements metav1.Object
func (c *ConstObject) GetName() string {
	return c.accessor.GetName()
}

// SetName implements metav1.Object
func (c *ConstObject) SetName(name string) {
	panic("not implemented")
}

// GetGenerateName implements metav1.Object
func (c *ConstObject) GetGenerateName() string {
	panic("not implemented")
}

// SetGenerateName implements metav1.Object
func (c *ConstObject) SetGenerateName(name string) {
	panic("not implemented")
}

// GetUID implements metav1.Object
func (c *ConstObject) GetUID() types.UID {
	panic("not implemented")
}

// SetUID implements metav1.Object
func (c *ConstObject) SetUID(uid types.UID) {
	panic("not implemented")
}

// GetResourceVersion implements metav1.Object
func (c *ConstObject) GetResourceVersion() string {
	panic("not implemented")
}

// SetResourceVersion implements metav1.Object
func (c *ConstObject) SetResourceVersion(version string) {
	panic("not implemented")
}

// GetGeneration implements metav1.Object
func (c *ConstObject) GetGeneration() int64 {
	panic("not implemented")
}

// SetGeneration implements metav1.Object
func (c *ConstObject) SetGeneration(generation int64) {
	panic("SetGeneration not implemented")
}

// GetSelfLink implements metav1.Object
func (c *ConstObject) GetSelfLink() string {
	panic("GetSelfLink not implemented")
}

// todo fix this SetSelfLink implements metav1.Object
func (c *ConstObject) SetSelfLink(selfLink string) {
	current := c.accessor.GetSelfLink()
	glog.Warningf("SetSelfLink is incorrectly handled  (from %q -> %q)", current, selfLink)
	// TODO: Copying ConstObject itself is pretty cheap; the problem is
	// that this SelfLink appears in the serialized output.
	// Possibility: clone ConstObject, allow SelfLink to be set, when
	// encoding set self link in underlying object.  If shared, lock
	// underlying object first.
	c.accessor.SetSelfLink(selfLink)
}

// GetCreationTimestamp implements metav1.Object
func (c *ConstObject) GetCreationTimestamp() metav1.Time {
	panic("not implemented")
}

// SetCreationTimestamp implements metav1.Object
func (c *ConstObject) SetCreationTimestamp(timestamp metav1.Time) {
	panic("not implemented")
}

// GetDeletionTimestamp implements metav1.Object
func (c *ConstObject) GetDeletionTimestamp() *metav1.Time {
	panic("not implemented")
}

// SetDeletionTimestamp implements metav1.Object
func (c *ConstObject) SetDeletionTimestamp(timestamp *metav1.Time) {
	panic("not implemented")
}

// GetDeletionGracePeriodSeconds implements metav1.Object
func (c *ConstObject) GetDeletionGracePeriodSeconds() *int64 {
	panic("not implemented")
}

// SetDeletionGracePeriodSeconds implements metav1.Object
func (c *ConstObject) SetDeletionGracePeriodSeconds(*int64) {
	panic("not implemented")
}

// GetLabels implements metav1.Object
func (c *ConstObject) GetLabels() map[string]string {
	// TODO: used by DefaultClusterScopedAttr but we could switch that to labels.Labels
	src := c.accessor.GetLabels()
	if len(src) == 0 {
		return nil
	}
	m := make(map[string]string, len(src))
	for k, v := range src {
		m[k] = v
	}
	return m
}

// SetLabels implements metav1.Object
func (c *ConstObject) SetLabels(labels map[string]string) {
	panic("not implemented")
}

// GetAnnotations implements metav1.Object
func (c *ConstObject) GetAnnotations() map[string]string {
	panic("not implemented")
}

// SetAnnotations implements metav1.Object
func (c *ConstObject) SetAnnotations(annotations map[string]string) {
	panic("not implemented")
}

// GetInitializers implements metav1.Object
func (c *ConstObject) GetInitializers() *metav1.Initializers {
	i := c.accessor.GetInitializers()
	if i == nil {
		return nil
	}

	// TODO: Safer copy - or deal with DefaultClusterScopedAttr, called from processEvent
	copy := *i
	return &copy
}

// SetInitializers implements metav1.Object
func (c *ConstObject) SetInitializers(initializers *metav1.Initializers) {
	panic("not implemented")
}

// GetFinalizers implements metav1.Object
func (c *ConstObject) GetFinalizers() []string {
	panic("not implemented")
}

// SetFinalizers implements metav1.Object
func (c *ConstObject) SetFinalizers(finalizers []string) {
	panic("not implemented")
}

// GetOwnerReferences implements metav1.Object
func (c *ConstObject) GetOwnerReferences() []metav1.OwnerReference {
	panic("not implemented")
}

// SetOwnerReferences implements metav1.Object
func (c *ConstObject) SetOwnerReferences([]metav1.OwnerReference) {
	panic("not implemented")
}

// GetCLusterName implements metav1.Object
func (c *ConstObject) GetClusterName() string {
	panic("not implemented")
}

// SetClusterName implements metav1.Object
func (c *ConstObject) SetClusterName(clusterName string) {
	panic("not implemented")
}

// Constify wraps the specific object in a ConstObject, which does not allow mutation
// Note that the original object can still be mutated, so care should be taken that this does not happen!
func Constify(o runtime.Object) *ConstObject {
	if o == nil {
		return nil
	}

	accessor, err := meta.Accessor(o)
	if err != nil {
		panic(fmt.Sprintf("error getting Accessor for %T: %v", o, err))
	}

	return &ConstObject{inner: o, accessor: accessor}
}

// Encode implements runtime.Encodable
// TODO: Get our own encoder?
func (c *ConstObject) Encode(mediaType string, encoder runtime.Encoder, w io.Writer) error {
	// TODO: Caching
	return encoder.Encode(c.inner, w)
}

// DeconstifyForTest allows access to the inner object; this should only be used for tests as it breaks our contract
// TODO: DeepCopy?
func DeconstifyForTest(obj runtime.Object) runtime.Object {
	if co, ok := obj.(*ConstObject); ok {
		return co.inner
	}
	return obj
}

func (c *ConstObject) String() string {
	if stringer, ok := c.inner.(fmt.Stringer); ok {
		return "ConstObject[" + stringer.String() + "]"
	}
	return fmt.Sprintf("ConstObject[type %T]", c.inner)
}

// Support for protobuf marshalling

var _ proto.Sizer = &ConstObject{}
var _ runtime.ProtobufMarshaller = &ConstObject{}

// Size implements proto.Sizer
func (c *ConstObject) Size() int {
	if err := c.ensureProtobuf(); err != nil {
		return 0
	}

	return len(c.protobufData)
}

// MarshalTo implements runtime.ProtobufMarshaller
func (c *ConstObject) MarshalTo(data []byte) (int, error) {
	if err := c.ensureProtobuf(); err != nil {
		return 0, err
	}

	// TODO: What if data is too small?
	n := copy(data, c.protobufData)
	return n, nil
}

// ensureProtobuf encodes the object to protobuf
func (c *ConstObject) ensureProtobuf() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.protobufData != nil {
		return nil
	}

	data, err := protobuf.Marshal(c.inner)
	if err != nil {
		return err
	}

	c.protobufData = data
	return nil
}
