package packfile

import (
	"bytes"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/cache"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
)

// Format specifies if the packfile uses ref-deltas or ofs-deltas.
type Format int

// Possible values of the Format type.
const (
	UnknownFormat Format = iota
	OFSDeltaFormat
	REFDeltaFormat
)

var (
	// ErrMaxObjectsLimitReached is returned by Decode when the number
	// of objects in the packfile is higher than
	// Decoder.MaxObjectsLimit.
	ErrMaxObjectsLimitReached = NewError("max. objects limit reached")
	// ErrInvalidObject is returned by Decode when an invalid object is
	// found in the packfile.
	ErrInvalidObject = NewError("invalid git object")
	// ErrPackEntryNotFound is returned by Decode when a reference in
	// the packfile references and unknown object.
	ErrPackEntryNotFound = NewError("can't find a pack entry")
	// ErrZLib is returned by Decode when there was an error unzipping
	// the packfile contents.
	ErrZLib = NewError("zlib reading error")
	// ErrCannotRecall is returned by RecallByOffset or RecallByHash if the object
	// to recall cannot be returned.
	ErrCannotRecall = NewError("cannot recall object")
	// ErrResolveDeltasNotSupported is returned if a NewDecoder is used with a
	// non-seekable scanner and without a plumbing.ObjectStorage
	ErrResolveDeltasNotSupported = NewError("resolve delta is not supported")
	// ErrNonSeekable is returned if a ReadObjectAt method is called without a
	// seekable scanner
	ErrNonSeekable = NewError("non-seekable scanner")
	// ErrRollback error making Rollback over a transaction after an error
	ErrRollback = NewError("rollback error, during set error")
	// ErrAlreadyDecoded is returned if NewDecoder is called for a second time
	ErrAlreadyDecoded = NewError("packfile was already decoded")
)

// Decoder reads and decodes packfiles from an input Scanner, if an ObjectStorer
// was provided the decoded objects are store there. If not the decode object
// is destroyed. The Offsets and CRCs are calculated whether an
// ObjectStorer was provided or not.
type Decoder struct {
	DeltaBaseCache cache.Object

	s  *Scanner
	o  storer.EncodedObjectStorer
	tx storer.Transaction

	isDecoded bool

	// hasBuiltIndex indicates if the index is fully built or not. If it is not,
	// will be built incrementally while decoding.
	hasBuiltIndex bool
	idx           *Index

	offsetToType map[int64]plumbing.ObjectType
	decoderType  plumbing.ObjectType
}

// NewDecoder returns a new Decoder that decodes a Packfile using the given
// Scanner and stores the objects in the provided EncodedObjectStorer. ObjectStorer can be nil, in this
// If the passed EncodedObjectStorer is nil, objects are not stored, but
// offsets on the Packfile and CRCs are calculated.
//
// If EncodedObjectStorer is nil and the Scanner is not Seekable, ErrNonSeekable is
// returned.
//
// If the ObjectStorer implements storer.Transactioner, a transaction is created
// during the Decode execution. If anything fails, Rollback is called
func NewDecoder(s *Scanner, o storer.EncodedObjectStorer) (*Decoder, error) {
	return NewDecoderForType(s, o, plumbing.AnyObject)
}

// NewDecoderForType returns a new Decoder but in this case for a specific object type.
// When an object is read using this Decoder instance and it is not of the same type of
// the specified one, nil will be returned. This is intended to avoid the content
// deserialization of all the objects
func NewDecoderForType(s *Scanner, o storer.EncodedObjectStorer,
	t plumbing.ObjectType) (*Decoder, error) {

	if t == plumbing.OFSDeltaObject ||
		t == plumbing.REFDeltaObject ||
		t == plumbing.InvalidObject {
		return nil, plumbing.ErrInvalidType
	}

	if !canResolveDeltas(s, o) {
		return nil, ErrResolveDeltasNotSupported
	}

	return &Decoder{
		s: s,
		o: o,

		idx:          NewIndex(0),
		offsetToType: make(map[int64]plumbing.ObjectType, 0),
		decoderType:  t,
	}, nil
}

func canResolveDeltas(s *Scanner, o storer.EncodedObjectStorer) bool {
	return s.IsSeekable || o != nil
}

// Decode reads a packfile and stores it in the value pointed to by s. The
// offsets and the CRCs are calculated by this method
func (d *Decoder) Decode() (checksum plumbing.Hash, err error) {
	defer func() { d.isDecoded = true }()

	if d.isDecoded {
		return plumbing.ZeroHash, ErrAlreadyDecoded
	}

	if err := d.doDecode(); err != nil {
		return plumbing.ZeroHash, err
	}

	return d.s.Checksum()
}

func (d *Decoder) doDecode() error {
	_, count, err := d.s.Header()
	if err != nil {
		return err
	}

	if !d.hasBuiltIndex {
		d.idx = NewIndex(int(count))
	}
	defer func() { d.hasBuiltIndex = true }()

	_, isTxStorer := d.o.(storer.Transactioner)
	switch {
	case d.o == nil:
		return d.decodeObjects(int(count))
	case isTxStorer:
		return d.decodeObjectsWithObjectStorerTx(int(count))
	default:
		return d.decodeObjectsWithObjectStorer(int(count))
	}
}

func (d *Decoder) decodeObjects(count int) error {
	for i := 0; i < count; i++ {
		if _, err := d.DecodeObject(); err != nil {
			return err
		}
	}

	return nil
}

func (d *Decoder) decodeObjectsWithObjectStorer(count int) error {
	for i := 0; i < count; i++ {
		obj, err := d.DecodeObject()
		if err != nil {
			return err
		}

		if _, err := d.o.SetEncodedObject(obj); err != nil {
			return err
		}
	}

	return nil
}

func (d *Decoder) decodeObjectsWithObjectStorerTx(count int) error {
	d.tx = d.o.(storer.Transactioner).Begin()

	for i := 0; i < count; i++ {
		obj, err := d.DecodeObject()
		if err != nil {
			return err
		}

		if _, err := d.tx.SetEncodedObject(obj); err != nil {
			if rerr := d.tx.Rollback(); rerr != nil {
				return ErrRollback.AddDetails(
					"error: %s, during tx.Set error: %s", rerr, err,
				)
			}

			return err
		}

	}

	return d.tx.Commit()
}

// DecodeObject reads the next object from the scanner and returns it. This
// method can be used in replacement of the Decode method, to work in a
// interactive way. If you created a new decoder instance using NewDecoderForType
// constructor, if the object decoded is not equals to the specified one, nil will
// be returned
func (d *Decoder) DecodeObject() (plumbing.EncodedObject, error) {
	h, err := d.s.NextObjectHeader()
	if err != nil {
		return nil, err
	}

	if d.decoderType == plumbing.AnyObject {
		return d.decodeByHeader(h)
	}

	return d.decodeIfSpecificType(h)
}

func (d *Decoder) decodeIfSpecificType(h *ObjectHeader) (plumbing.EncodedObject, error) {
	var (
		obj      plumbing.EncodedObject
		realType plumbing.ObjectType
		err      error
	)
	switch h.Type {
	case plumbing.OFSDeltaObject:
		realType, err = d.ofsDeltaType(h.OffsetReference)
	case plumbing.REFDeltaObject:
		realType, err = d.refDeltaType(h.Reference)
		if err == plumbing.ErrObjectNotFound {
			obj, err = d.decodeByHeader(h)
			if err != nil {
				realType = obj.Type()
			}
		}
	default:
		realType = h.Type
	}

	if err != nil {
		return nil, err
	}

	d.offsetToType[h.Offset] = realType

	if d.decoderType == realType {
		if obj != nil {
			return obj, nil
		}

		return d.decodeByHeader(h)
	}

	return nil, nil
}

func (d *Decoder) ofsDeltaType(offset int64) (plumbing.ObjectType, error) {
	t, ok := d.offsetToType[offset]
	if !ok {
		return plumbing.InvalidObject, plumbing.ErrObjectNotFound
	}

	return t, nil
}

func (d *Decoder) refDeltaType(ref plumbing.Hash) (plumbing.ObjectType, error) {
	e, ok := d.idx.LookupHash(ref)
	if !ok {
		return plumbing.InvalidObject, plumbing.ErrObjectNotFound
	}

	return d.ofsDeltaType(int64(e.Offset))
}

func (d *Decoder) decodeByHeader(h *ObjectHeader) (plumbing.EncodedObject, error) {
	obj := d.newObject()
	obj.SetSize(h.Length)
	obj.SetType(h.Type)
	var crc uint32
	var err error
	switch h.Type {
	case plumbing.CommitObject, plumbing.TreeObject, plumbing.BlobObject, plumbing.TagObject:
		crc, err = d.fillRegularObjectContent(obj)
	case plumbing.REFDeltaObject:
		crc, err = d.fillREFDeltaObjectContent(obj, h.Reference)
	case plumbing.OFSDeltaObject:
		crc, err = d.fillOFSDeltaObjectContent(obj, h.OffsetReference)
	default:
		err = ErrInvalidObject.AddDetails("type %q", h.Type)
	}

	if err != nil {
		return obj, err
	}

	if !d.hasBuiltIndex {
		d.idx.Add(obj.Hash(), uint64(h.Offset), crc)
	}

	return obj, nil
}

func (d *Decoder) newObject() plumbing.EncodedObject {
	if d.o == nil {
		return &plumbing.MemoryObject{}
	}

	return d.o.NewEncodedObject()
}

// DecodeObjectAt reads an object at the given location. Every EncodedObject
// returned is added into a internal index. This is intended to be able to regenerate
// objects from deltas (offset deltas or reference deltas) without an package index
// (.idx file). If Decode wasn't called previously objects offset should provided
// using the SetOffsets method.
func (d *Decoder) DecodeObjectAt(offset int64) (plumbing.EncodedObject, error) {
	if !d.s.IsSeekable {
		return nil, ErrNonSeekable
	}

	beforeJump, err := d.s.SeekFromStart(offset)
	if err != nil {
		return nil, err
	}

	defer func() {
		_, seekErr := d.s.SeekFromStart(beforeJump)
		if err == nil {
			err = seekErr
		}
	}()

	return d.DecodeObject()
}

func (d *Decoder) fillRegularObjectContent(obj plumbing.EncodedObject) (uint32, error) {
	w, err := obj.Writer()
	if err != nil {
		return 0, err
	}

	_, crc, err := d.s.NextObject(w)
	return crc, err
}

func (d *Decoder) fillREFDeltaObjectContent(obj plumbing.EncodedObject, ref plumbing.Hash) (uint32, error) {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	_, crc, err := d.s.NextObject(buf)
	if err != nil {
		return 0, err
	}

	base, ok := d.cacheGet(ref)
	if !ok {
		base, err = d.recallByHash(ref)
		if err != nil {
			return 0, err
		}
	}

	obj.SetType(base.Type())
	err = ApplyDelta(obj, base, buf.Bytes())
	d.cachePut(obj)
	bufPool.Put(buf)

	return crc, err
}

func (d *Decoder) fillOFSDeltaObjectContent(obj plumbing.EncodedObject, offset int64) (uint32, error) {
	buf := bytes.NewBuffer(nil)
	_, crc, err := d.s.NextObject(buf)
	if err != nil {
		return 0, err
	}

	e, ok := d.idx.LookupOffset(uint64(offset))
	var base plumbing.EncodedObject
	if ok {
		base, ok = d.cacheGet(e.Hash)
	}

	if !ok {
		base, err = d.recallByOffset(offset)
		if err != nil {
			return 0, err
		}
	}

	obj.SetType(base.Type())
	err = ApplyDelta(obj, base, buf.Bytes())
	d.cachePut(obj)

	return crc, err
}

func (d *Decoder) cacheGet(h plumbing.Hash) (plumbing.EncodedObject, bool) {
	if d.DeltaBaseCache == nil {
		return nil, false
	}

	return d.DeltaBaseCache.Get(h)
}

func (d *Decoder) cachePut(obj plumbing.EncodedObject) {
	if d.DeltaBaseCache == nil {
		return
	}

	d.DeltaBaseCache.Put(obj)
}

func (d *Decoder) recallByOffset(o int64) (plumbing.EncodedObject, error) {
	if d.s.IsSeekable {
		return d.DecodeObjectAt(o)
	}

	if e, ok := d.idx.LookupOffset(uint64(o)); ok {
		return d.recallByHashNonSeekable(e.Hash)
	}

	return nil, plumbing.ErrObjectNotFound
}

func (d *Decoder) recallByHash(h plumbing.Hash) (plumbing.EncodedObject, error) {
	if d.s.IsSeekable {
		if e, ok := d.idx.LookupHash(h); ok {
			return d.DecodeObjectAt(int64(e.Offset))
		}
	}

	return d.recallByHashNonSeekable(h)
}

// recallByHashNonSeekable if we are in a transaction the objects are read from
// the transaction, if not are directly read from the ObjectStorer
func (d *Decoder) recallByHashNonSeekable(h plumbing.Hash) (obj plumbing.EncodedObject, err error) {
	if d.tx != nil {
		obj, err = d.tx.EncodedObject(plumbing.AnyObject, h)
	} else {
		obj, err = d.o.EncodedObject(plumbing.AnyObject, h)
	}

	if err != plumbing.ErrObjectNotFound {
		return obj, err
	}

	return nil, plumbing.ErrObjectNotFound
}

// SetIndex sets an index for the packfile. It is recommended to set this.
// The index might be read from a file or reused from a previous Decoder usage
// (see Index function).
func (d *Decoder) SetIndex(idx *Index) {
	d.hasBuiltIndex = true
	d.idx = idx
}

// Index returns the index for the packfile. If index was set with SetIndex,
// Index will return it. Otherwise, it will return an index that is built while
// decoding. If neither SetIndex was called with a full index or Decode called
// for the whole packfile, then the returned index will be incomplete.
func (d *Decoder) Index() *Index {
	return d.idx
}

// Close closes the Scanner. usually this mean that the whole reader is read and
// discarded
func (d *Decoder) Close() error {
	return d.s.Close()
}
