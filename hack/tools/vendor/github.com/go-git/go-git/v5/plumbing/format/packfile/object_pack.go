package packfile

import (
	"github.com/go-git/go-git/v5/plumbing"
)

// ObjectToPack is a representation of an object that is going to be into a
// pack file.
type ObjectToPack struct {
	// The main object to pack, it could be any object, including deltas
	Object plumbing.EncodedObject
	// Base is the object that a delta is based on (it could be also another delta).
	// If the main object is not a delta, Base will be null
	Base *ObjectToPack
	// Original is the object that we can generate applying the delta to
	// Base, or the same object as Object in the case of a non-delta
	// object.
	Original plumbing.EncodedObject
	// Depth is the amount of deltas needed to resolve to obtain Original
	// (delta based on delta based on ...)
	Depth int

	// offset in pack when object has been already written, or 0 if it
	// has not been written yet
	Offset int64

	// Information from the original object
	resolvedOriginal bool
	originalType     plumbing.ObjectType
	originalSize     int64
	originalHash     plumbing.Hash
}

// newObjectToPack creates a correct ObjectToPack based on a non-delta object
func newObjectToPack(o plumbing.EncodedObject) *ObjectToPack {
	return &ObjectToPack{
		Object:   o,
		Original: o,
	}
}

// newDeltaObjectToPack creates a correct ObjectToPack for a delta object, based on
// his base (could be another delta), the delta target (in this case called original),
// and the delta Object itself
func newDeltaObjectToPack(base *ObjectToPack, original, delta plumbing.EncodedObject) *ObjectToPack {
	return &ObjectToPack{
		Object:   delta,
		Base:     base,
		Original: original,
		Depth:    base.Depth + 1,
	}
}

// BackToOriginal converts that ObjectToPack to a non-deltified object if it was one
func (o *ObjectToPack) BackToOriginal() {
	if o.IsDelta() && o.Original != nil {
		o.Object = o.Original
		o.Base = nil
		o.Depth = 0
	}
}

// IsWritten returns if that ObjectToPack was
// already written into the packfile or not
func (o *ObjectToPack) IsWritten() bool {
	return o.Offset > 1
}

// MarkWantWrite marks this ObjectToPack as WantWrite
// to avoid delta chain loops
func (o *ObjectToPack) MarkWantWrite() {
	o.Offset = 1
}

// WantWrite checks if this ObjectToPack was marked as WantWrite before
func (o *ObjectToPack) WantWrite() bool {
	return o.Offset == 1
}

// SetOriginal sets both Original and saves size, type and hash. If object
// is nil Original is set but previous resolved values are kept
func (o *ObjectToPack) SetOriginal(obj plumbing.EncodedObject) {
	o.Original = obj
	o.SaveOriginalMetadata()
}

// SaveOriginalMetadata saves size, type and hash of Original object
func (o *ObjectToPack) SaveOriginalMetadata() {
	if o.Original != nil {
		o.originalSize = o.Original.Size()
		o.originalType = o.Original.Type()
		o.originalHash = o.Original.Hash()
		o.resolvedOriginal = true
	}
}

// CleanOriginal sets Original to nil
func (o *ObjectToPack) CleanOriginal() {
	o.Original = nil
}

func (o *ObjectToPack) Type() plumbing.ObjectType {
	if o.Original != nil {
		return o.Original.Type()
	}

	if o.resolvedOriginal {
		return o.originalType
	}

	if o.Base != nil {
		return o.Base.Type()
	}

	if o.Object != nil {
		return o.Object.Type()
	}

	panic("cannot get type")
}

func (o *ObjectToPack) Hash() plumbing.Hash {
	if o.Original != nil {
		return o.Original.Hash()
	}

	if o.resolvedOriginal {
		return o.originalHash
	}

	do, ok := o.Object.(plumbing.DeltaObject)
	if ok {
		return do.ActualHash()
	}

	panic("cannot get hash")
}

func (o *ObjectToPack) Size() int64 {
	if o.Original != nil {
		return o.Original.Size()
	}

	if o.resolvedOriginal {
		return o.originalSize
	}

	do, ok := o.Object.(plumbing.DeltaObject)
	if ok {
		return do.ActualSize()
	}

	panic("cannot get ObjectToPack size")
}

func (o *ObjectToPack) IsDelta() bool {
	return o.Base != nil
}

func (o *ObjectToPack) SetDelta(base *ObjectToPack, delta plumbing.EncodedObject) {
	o.Object = delta
	o.Base = base
	o.Depth = base.Depth + 1
}
