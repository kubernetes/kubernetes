package ebpf

import (
	"fmt"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"

	"github.com/pkg/errors"
)

// MapSpec defines a Map.
type MapSpec struct {
	// Name is passed to the kernel as a debug aid. Must only contain
	// alpha numeric and '_' characters.
	Name       string
	Type       MapType
	KeySize    uint32
	ValueSize  uint32
	MaxEntries uint32
	Flags      uint32
	// InnerMap is used as a template for ArrayOfMaps and HashOfMaps
	InnerMap *MapSpec
}

func (ms *MapSpec) String() string {
	return fmt.Sprintf("%s(keySize=%d, valueSize=%d, maxEntries=%d, flags=%d)", ms.Type, ms.KeySize, ms.ValueSize, ms.MaxEntries, ms.Flags)
}

// Copy returns a copy of the spec.
func (ms *MapSpec) Copy() *MapSpec {
	if ms == nil {
		return nil
	}

	cpy := *ms
	cpy.InnerMap = ms.InnerMap.Copy()
	return &cpy
}

// Map represents a Map file descriptor.
//
// It is not safe to close a map which is used by other goroutines.
//
// Methods which take interface{} arguments by default encode
// them using binary.Read/Write in the machine's native endianness.
//
// Implement encoding.BinaryMarshaler or encoding.BinaryUnmarshaler
// if you require custom encoding.
type Map struct {
	name string
	fd   *bpfFD
	abi  MapABI
	// Per CPU maps return values larger than the size in the spec
	fullValueSize int
}

// NewMapFromFD creates a map from a raw fd.
//
// You should not use fd after calling this function.
func NewMapFromFD(fd int) (*Map, error) {
	if fd < 0 {
		return nil, errors.New("invalid fd")
	}
	bpfFd := newBPFFD(uint32(fd))

	name, abi, err := newMapABIFromFd(bpfFd)
	if err != nil {
		bpfFd.forget()
		return nil, err
	}
	return newMap(bpfFd, name, abi)
}

// NewMap creates a new Map.
//
// Creating a map for the first time will perform feature detection
// by creating small, temporary maps.
func NewMap(spec *MapSpec) (*Map, error) {
	if spec.Type != ArrayOfMaps && spec.Type != HashOfMaps {
		return createMap(spec, nil)
	}

	if spec.InnerMap == nil {
		return nil, errors.Errorf("%s requires InnerMap", spec.Type)
	}

	template, err := createMap(spec.InnerMap, nil)
	if err != nil {
		return nil, err
	}
	defer template.Close()

	return createMap(spec, template.fd)
}

func createMap(spec *MapSpec, inner *bpfFD) (*Map, error) {
	spec = spec.Copy()

	switch spec.Type {
	case ArrayOfMaps:
		fallthrough
	case HashOfMaps:
		if spec.ValueSize != 0 && spec.ValueSize != 4 {
			return nil, errors.Errorf("ValueSize must be zero or four for map of map")
		}
		spec.ValueSize = 4

	case PerfEventArray:
		if spec.KeySize != 0 {
			return nil, errors.Errorf("KeySize must be zero for perf event array")
		}
		if spec.ValueSize != 0 {
			return nil, errors.Errorf("ValueSize must be zero for perf event array")
		}
		if spec.MaxEntries == 0 {
			n, err := internal.OnlineCPUs()
			if err != nil {
				return nil, errors.Wrap(err, "perf event array")
			}
			spec.MaxEntries = uint32(n)
		}

		spec.KeySize = 4
		spec.ValueSize = 4
	}

	attr := bpfMapCreateAttr{
		mapType:    spec.Type,
		keySize:    spec.KeySize,
		valueSize:  spec.ValueSize,
		maxEntries: spec.MaxEntries,
		flags:      spec.Flags,
	}

	if inner != nil {
		var err error
		attr.innerMapFd, err = inner.value()
		if err != nil {
			return nil, errors.Wrap(err, "map create")
		}
	}

	name, err := newBPFObjName(spec.Name)
	if err != nil {
		return nil, errors.Wrap(err, "map create")
	}

	if haveObjName.Result() {
		attr.mapName = name
	}

	fd, err := bpfMapCreate(&attr)
	if err != nil {
		return nil, errors.Wrap(err, "map create")
	}

	return newMap(fd, spec.Name, newMapABIFromSpec(spec))
}

func newMap(fd *bpfFD, name string, abi *MapABI) (*Map, error) {
	m := &Map{
		name,
		fd,
		*abi,
		int(abi.ValueSize),
	}

	if !abi.Type.hasPerCPUValue() {
		return m, nil
	}

	possibleCPUs, err := internal.PossibleCPUs()
	if err != nil {
		return nil, err
	}

	m.fullValueSize = align(int(abi.ValueSize), 8) * possibleCPUs
	return m, nil
}

func (m *Map) String() string {
	if m.name != "" {
		return fmt.Sprintf("%s(%s)#%v", m.abi.Type, m.name, m.fd)
	}
	return fmt.Sprintf("%s#%v", m.abi.Type, m.fd)
}

// ABI gets the ABI of the Map
func (m *Map) ABI() MapABI {
	return m.abi
}

// Lookup retrieves a value from a Map.
//
// Calls Close() on valueOut if it is of type **Map or **Program,
// and *valueOut is not nil.
//
// Returns an error if the key doesn't exist, see IsNotExist.
func (m *Map) Lookup(key, valueOut interface{}) error {
	valuePtr, valueBytes := makeBuffer(valueOut, m.fullValueSize)

	if err := m.lookup(key, valuePtr); err != nil {
		return err
	}

	if valueBytes == nil {
		return nil
	}

	if m.abi.Type.hasPerCPUValue() {
		return unmarshalPerCPUValue(valueOut, int(m.abi.ValueSize), valueBytes)
	}

	switch value := valueOut.(type) {
	case **Map:
		m, err := unmarshalMap(valueBytes)
		if err != nil {
			return err
		}

		(*value).Close()
		*value = m
		return nil
	case *Map:
		return errors.Errorf("can't unmarshal into %T, need %T", value, (**Map)(nil))
	case Map:
		return errors.Errorf("can't unmarshal into %T, need %T", value, (**Map)(nil))

	case **Program:
		p, err := unmarshalProgram(valueBytes)
		if err != nil {
			return err
		}

		(*value).Close()
		*value = p
		return nil
	case *Program:
		return errors.Errorf("can't unmarshal into %T, need %T", value, (**Program)(nil))
	case Program:
		return errors.Errorf("can't unmarshal into %T, need %T", value, (**Program)(nil))

	default:
		return unmarshalBytes(valueOut, valueBytes)
	}
}

// LookupBytes gets a value from Map.
//
// Returns a nil value if a key doesn't exist.
func (m *Map) LookupBytes(key interface{}) ([]byte, error) {
	valueBytes := make([]byte, m.fullValueSize)
	valuePtr := newPtr(unsafe.Pointer(&valueBytes[0]))

	err := m.lookup(key, valuePtr)
	if IsNotExist(err) {
		return nil, nil
	}

	return valueBytes, err
}

func (m *Map) lookup(key interface{}, valueOut syscallPtr) error {
	keyPtr, err := marshalPtr(key, int(m.abi.KeySize))
	if err != nil {
		return errors.WithMessage(err, "can't marshal key")
	}

	err = bpfMapLookupElem(m.fd, keyPtr, valueOut)
	return errors.WithMessage(err, "lookup failed")
}

// MapUpdateFlags controls the behaviour of the Map.Update call.
//
// The exact semantics depend on the specific MapType.
type MapUpdateFlags uint64

const (
	// UpdateAny creates a new element or update an existing one.
	UpdateAny MapUpdateFlags = iota
	// UpdateNoExist creates a new element.
	UpdateNoExist MapUpdateFlags = 1 << (iota - 1)
	// UpdateExist updates an existing element.
	UpdateExist
)

// Put replaces or creates a value in map.
//
// It is equivalent to calling Update with UpdateAny.
func (m *Map) Put(key, value interface{}) error {
	return m.Update(key, value, UpdateAny)
}

// Update changes the value of a key.
func (m *Map) Update(key, value interface{}, flags MapUpdateFlags) error {
	keyPtr, err := marshalPtr(key, int(m.abi.KeySize))
	if err != nil {
		return errors.WithMessage(err, "can't marshal key")
	}

	var valuePtr syscallPtr
	if m.abi.Type.hasPerCPUValue() {
		valuePtr, err = marshalPerCPUValue(value, int(m.abi.ValueSize))
	} else {
		valuePtr, err = marshalPtr(value, int(m.abi.ValueSize))
	}
	if err != nil {
		return errors.WithMessage(err, "can't marshal value")
	}

	return bpfMapUpdateElem(m.fd, keyPtr, valuePtr, uint64(flags))
}

// Delete removes a value.
//
// Returns an error if the key does not exist, see IsNotExist.
func (m *Map) Delete(key interface{}) error {
	keyPtr, err := marshalPtr(key, int(m.abi.KeySize))
	if err != nil {
		return errors.WithMessage(err, "can't marshal key")
	}

	err = bpfMapDeleteElem(m.fd, keyPtr)
	return errors.WithMessage(err, "can't delete key")
}

// NextKey finds the key following an initial key.
//
// See NextKeyBytes for details.
func (m *Map) NextKey(key, nextKeyOut interface{}) error {
	nextKeyPtr, nextKeyBytes := makeBuffer(nextKeyOut, int(m.abi.KeySize))

	if err := m.nextKey(key, nextKeyPtr); err != nil {
		return err
	}

	if nextKeyBytes == nil {
		return nil
	}

	err := unmarshalBytes(nextKeyOut, nextKeyBytes)
	return errors.WithMessage(err, "can't unmarshal next key")
}

// NextKeyBytes returns the key following an initial key as a byte slice.
//
// Passing nil will return the first key.
//
// Use Iterate if you want to traverse all entries in the map.
func (m *Map) NextKeyBytes(key interface{}) ([]byte, error) {
	nextKey := make([]byte, m.abi.KeySize)
	nextKeyPtr := newPtr(unsafe.Pointer(&nextKey[0]))

	err := m.nextKey(key, nextKeyPtr)
	if IsNotExist(err) {
		return nil, nil
	}

	return nextKey, err
}

func (m *Map) nextKey(key interface{}, nextKeyOut syscallPtr) error {
	var (
		keyPtr syscallPtr
		err    error
	)

	if key != nil {
		keyPtr, err = marshalPtr(key, int(m.abi.KeySize))
		if err != nil {
			return errors.WithMessage(err, "can't marshal key")
		}
	}

	err = bpfMapGetNextKey(m.fd, keyPtr, nextKeyOut)
	return errors.WithMessage(err, "can't get next key")
}

// Iterate traverses a map.
//
// It's safe to create multiple iterators at the same time.
//
// It's not possible to guarantee that all keys in a map will be
// returned if there are concurrent modifications to the map.
func (m *Map) Iterate() *MapIterator {
	return newMapIterator(m)
}

// Close removes a Map
func (m *Map) Close() error {
	if m == nil {
		// This makes it easier to clean up when iterating maps
		// of maps / programs.
		return nil
	}

	return m.fd.close()
}

// FD gets the file descriptor of the Map.
//
// Calling this function is invalid after Close has been called.
func (m *Map) FD() int {
	fd, err := m.fd.value()
	if err != nil {
		// Best effort: -1 is the number most likely to be an
		// invalid file descriptor.
		return -1
	}

	return int(fd)
}

// Clone creates a duplicate of the Map.
//
// Closing the duplicate does not affect the original, and vice versa.
// Changes made to the map are reflected by both instances however.
//
// Cloning a nil Map returns nil.
func (m *Map) Clone() (*Map, error) {
	if m == nil {
		return nil, nil
	}

	dup, err := m.fd.dup()
	if err != nil {
		return nil, errors.Wrap(err, "can't clone map")
	}

	return newMap(dup, m.name, &m.abi)
}

// Pin persists the map past the lifetime of the process that created it.
//
// This requires bpffs to be mounted above fileName. See http://cilium.readthedocs.io/en/doc-1.0/kubernetes/install/#mounting-the-bpf-fs-optional
func (m *Map) Pin(fileName string) error {
	return bpfPinObject(fileName, m.fd)
}

// LoadPinnedMap load a Map from a BPF file.
//
// The function is not compatible with nested maps.
// Use LoadPinnedMapExplicit in these situations.
func LoadPinnedMap(fileName string) (*Map, error) {
	fd, err := bpfGetObject(fileName)
	if err != nil {
		return nil, err
	}
	name, abi, err := newMapABIFromFd(fd)
	if err != nil {
		_ = fd.close()
		return nil, err
	}
	return newMap(fd, name, abi)
}

// LoadPinnedMapExplicit loads a map with explicit parameters.
func LoadPinnedMapExplicit(fileName string, abi *MapABI) (*Map, error) {
	fd, err := bpfGetObject(fileName)
	if err != nil {
		return nil, err
	}
	return newMap(fd, "", abi)
}

func unmarshalMap(buf []byte) (*Map, error) {
	if len(buf) != 4 {
		return nil, errors.New("map id requires 4 byte value")
	}

	// Looking up an entry in a nested map or prog array returns an id,
	// not an fd.
	id := internal.NativeEndian.Uint32(buf)
	fd, err := bpfGetMapFDByID(id)
	if err != nil {
		return nil, err
	}

	name, abi, err := newMapABIFromFd(fd)
	if err != nil {
		_ = fd.close()
		return nil, err
	}

	return newMap(fd, name, abi)
}

// MarshalBinary implements BinaryMarshaler.
func (m *Map) MarshalBinary() ([]byte, error) {
	fd, err := m.fd.value()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, 4)
	internal.NativeEndian.PutUint32(buf, fd)
	return buf, nil
}

// MapIterator iterates a Map.
//
// See Map.Iterate.
type MapIterator struct {
	target            *Map
	prevKey           interface{}
	prevBytes         []byte
	count, maxEntries uint32
	done              bool
	err               error
}

func newMapIterator(target *Map) *MapIterator {
	return &MapIterator{
		target:     target,
		maxEntries: target.abi.MaxEntries,
		prevBytes:  make([]byte, int(target.abi.KeySize)),
	}
}

var errIterationAborted = errors.New("iteration aborted")

// Next decodes the next key and value.
//
// Iterating a hash map from which keys are being deleted is not
// safe. You may see the same key multiple times. Iteration may
// also abort with an error, see IsIterationAborted.
//
// Returns false if there are no more entries. You must check
// the result of Err afterwards.
//
// See Map.Get for further caveats around valueOut.
func (mi *MapIterator) Next(keyOut, valueOut interface{}) bool {
	if mi.err != nil || mi.done {
		return false
	}

	for ; mi.count < mi.maxEntries; mi.count++ {
		var nextBytes []byte
		nextBytes, mi.err = mi.target.NextKeyBytes(mi.prevKey)
		if mi.err != nil {
			return false
		}

		if nextBytes == nil {
			mi.done = true
			return false
		}

		// The user can get access to nextBytes since unmarshalBytes
		// does not copy when unmarshaling into a []byte.
		// Make a copy to prevent accidental corruption of
		// iterator state.
		copy(mi.prevBytes, nextBytes)
		mi.prevKey = mi.prevBytes

		mi.err = mi.target.Lookup(nextBytes, valueOut)
		if IsNotExist(mi.err) {
			// Even though the key should be valid, we couldn't look up
			// its value. If we're iterating a hash map this is probably
			// because a concurrent delete removed the value before we
			// could get it. This means that the next call to NextKeyBytes
			// is very likely to restart iteration.
			// If we're iterating one of the fd maps like
			// ProgramArray it means that a given slot doesn't have
			// a valid fd associated. It's OK to continue to the next slot.
			continue
		}
		if mi.err != nil {
			return false
		}

		mi.err = unmarshalBytes(keyOut, nextBytes)
		return mi.err == nil
	}

	mi.err = errIterationAborted
	return false
}

// Err returns any encountered error.
//
// The method must be called after Next returns nil.
func (mi *MapIterator) Err() error {
	return mi.err
}

// IsNotExist returns true if the error indicates that a
// key doesn't exist.
func IsNotExist(err error) bool {
	return errors.Cause(err) == unix.ENOENT
}

// IsIterationAborted returns true if the iteration was aborted.
//
// This occurs when keys are deleted from a hash map during iteration.
func IsIterationAborted(err error) bool {
	return errors.Cause(err) == errIterationAborted
}
