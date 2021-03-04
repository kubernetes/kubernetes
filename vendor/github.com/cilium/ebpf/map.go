package ebpf

import (
	"errors"
	"fmt"
	"strings"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
	"github.com/cilium/ebpf/internal/unix"
)

// Errors returned by Map and MapIterator methods.
var (
	ErrKeyNotExist      = errors.New("key does not exist")
	ErrKeyExist         = errors.New("key already exists")
	ErrIterationAborted = errors.New("iteration aborted")
)

// MapID represents the unique ID of an eBPF map
type MapID uint32

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

	// The initial contents of the map. May be nil.
	Contents []MapKV

	// Whether to freeze a map after setting its initial contents.
	Freeze bool

	// InnerMap is used as a template for ArrayOfMaps and HashOfMaps
	InnerMap *MapSpec

	// The BTF associated with this map.
	BTF *btf.Map
}

func (ms *MapSpec) String() string {
	return fmt.Sprintf("%s(keySize=%d, valueSize=%d, maxEntries=%d, flags=%d)", ms.Type, ms.KeySize, ms.ValueSize, ms.MaxEntries, ms.Flags)
}

// Copy returns a copy of the spec.
//
// MapSpec.Contents is a shallow copy.
func (ms *MapSpec) Copy() *MapSpec {
	if ms == nil {
		return nil
	}

	cpy := *ms
	cpy.Contents = make([]MapKV, len(ms.Contents))
	copy(cpy.Contents, ms.Contents)
	cpy.InnerMap = ms.InnerMap.Copy()
	return &cpy
}

// MapKV is used to initialize the contents of a Map.
type MapKV struct {
	Key   interface{}
	Value interface{}
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
	fd   *internal.FD
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
	bpfFd := internal.NewFD(uint32(fd))

	name, abi, err := newMapABIFromFd(bpfFd)
	if err != nil {
		bpfFd.Forget()
		return nil, err
	}
	return newMap(bpfFd, name, abi)
}

// NewMap creates a new Map.
//
// Creating a map for the first time will perform feature detection
// by creating small, temporary maps.
//
// The caller is responsible for ensuring the process' rlimit is set
// sufficiently high for locking memory during map creation. This can be done
// by calling unix.Setrlimit with unix.RLIMIT_MEMLOCK prior to calling NewMap.
func NewMap(spec *MapSpec) (*Map, error) {
	if spec.BTF == nil {
		return newMapWithBTF(spec, nil)
	}

	handle, err := btf.NewHandle(btf.MapSpec(spec.BTF))
	if err != nil && !errors.Is(err, btf.ErrNotSupported) {
		return nil, fmt.Errorf("can't load BTF: %w", err)
	}

	return newMapWithBTF(spec, handle)
}

func newMapWithBTF(spec *MapSpec, handle *btf.Handle) (*Map, error) {
	if spec.Type != ArrayOfMaps && spec.Type != HashOfMaps {
		return createMap(spec, nil, handle)
	}

	if spec.InnerMap == nil {
		return nil, fmt.Errorf("%s requires InnerMap", spec.Type)
	}

	template, err := createMap(spec.InnerMap, nil, handle)
	if err != nil {
		return nil, err
	}
	defer template.Close()

	return createMap(spec, template.fd, handle)
}

func createMap(spec *MapSpec, inner *internal.FD, handle *btf.Handle) (*Map, error) {
	abi := newMapABIFromSpec(spec)

	switch spec.Type {
	case ArrayOfMaps:
		fallthrough
	case HashOfMaps:
		if err := haveNestedMaps(); err != nil {
			return nil, err
		}

		if abi.ValueSize != 0 && abi.ValueSize != 4 {
			return nil, errors.New("ValueSize must be zero or four for map of map")
		}
		abi.ValueSize = 4

	case PerfEventArray:
		if abi.KeySize != 0 && abi.KeySize != 4 {
			return nil, errors.New("KeySize must be zero or four for perf event array")
		}
		abi.KeySize = 4

		if abi.ValueSize != 0 && abi.ValueSize != 4 {
			return nil, errors.New("ValueSize must be zero or four for perf event array")
		}
		abi.ValueSize = 4

		if abi.MaxEntries == 0 {
			n, err := internal.PossibleCPUs()
			if err != nil {
				return nil, fmt.Errorf("perf event array: %w", err)
			}
			abi.MaxEntries = uint32(n)
		}
	}

	if abi.Flags&(unix.BPF_F_RDONLY_PROG|unix.BPF_F_WRONLY_PROG) > 0 || spec.Freeze {
		if err := haveMapMutabilityModifiers(); err != nil {
			return nil, fmt.Errorf("map create: %w", err)
		}
	}

	attr := bpfMapCreateAttr{
		mapType:    abi.Type,
		keySize:    abi.KeySize,
		valueSize:  abi.ValueSize,
		maxEntries: abi.MaxEntries,
		flags:      abi.Flags,
	}

	if inner != nil {
		var err error
		attr.innerMapFd, err = inner.Value()
		if err != nil {
			return nil, fmt.Errorf("map create: %w", err)
		}
	}

	if handle != nil && spec.BTF != nil {
		attr.btfFd = uint32(handle.FD())
		attr.btfKeyTypeID = btf.MapKey(spec.BTF).ID()
		attr.btfValueTypeID = btf.MapValue(spec.BTF).ID()
	}

	if haveObjName() == nil {
		attr.mapName = newBPFObjName(spec.Name)
	}

	fd, err := bpfMapCreate(&attr)
	if err != nil {
		return nil, fmt.Errorf("map create: %w", err)
	}

	m, err := newMap(fd, spec.Name, abi)
	if err != nil {
		return nil, err
	}

	if err := m.populate(spec.Contents); err != nil {
		m.Close()
		return nil, fmt.Errorf("map create: can't set initial contents: %w", err)
	}

	if spec.Freeze {
		if err := m.Freeze(); err != nil {
			m.Close()
			return nil, fmt.Errorf("can't freeze map: %w", err)
		}
	}

	return m, nil
}

func newMap(fd *internal.FD, name string, abi *MapABI) (*Map, error) {
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
		return fmt.Errorf("can't unmarshal into %T, need %T", value, (**Map)(nil))
	case Map:
		return fmt.Errorf("can't unmarshal into %T, need %T", value, (**Map)(nil))

	case **Program:
		p, err := unmarshalProgram(valueBytes)
		if err != nil {
			return err
		}

		(*value).Close()
		*value = p
		return nil
	case *Program:
		return fmt.Errorf("can't unmarshal into %T, need %T", value, (**Program)(nil))
	case Program:
		return fmt.Errorf("can't unmarshal into %T, need %T", value, (**Program)(nil))

	default:
		return unmarshalBytes(valueOut, valueBytes)
	}
}

// LookupAndDelete retrieves and deletes a value from a Map.
//
// Returns ErrKeyNotExist if the key doesn't exist.
func (m *Map) LookupAndDelete(key, valueOut interface{}) error {
	valuePtr, valueBytes := makeBuffer(valueOut, m.fullValueSize)

	keyPtr, err := marshalPtr(key, int(m.abi.KeySize))
	if err != nil {
		return fmt.Errorf("can't marshal key: %w", err)
	}

	if err := bpfMapLookupAndDelete(m.fd, keyPtr, valuePtr); err != nil {
		return fmt.Errorf("lookup and delete failed: %w", err)
	}

	return unmarshalBytes(valueOut, valueBytes)
}

// LookupBytes gets a value from Map.
//
// Returns a nil value if a key doesn't exist.
func (m *Map) LookupBytes(key interface{}) ([]byte, error) {
	valueBytes := make([]byte, m.fullValueSize)
	valuePtr := internal.NewSlicePointer(valueBytes)

	err := m.lookup(key, valuePtr)
	if errors.Is(err, ErrKeyNotExist) {
		return nil, nil
	}

	return valueBytes, err
}

func (m *Map) lookup(key interface{}, valueOut internal.Pointer) error {
	keyPtr, err := marshalPtr(key, int(m.abi.KeySize))
	if err != nil {
		return fmt.Errorf("can't marshal key: %w", err)
	}

	if err = bpfMapLookupElem(m.fd, keyPtr, valueOut); err != nil {
		return fmt.Errorf("lookup failed: %w", err)
	}
	return nil
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
		return fmt.Errorf("can't marshal key: %w", err)
	}

	var valuePtr internal.Pointer
	if m.abi.Type.hasPerCPUValue() {
		valuePtr, err = marshalPerCPUValue(value, int(m.abi.ValueSize))
	} else {
		valuePtr, err = marshalPtr(value, int(m.abi.ValueSize))
	}
	if err != nil {
		return fmt.Errorf("can't marshal value: %w", err)
	}

	if err = bpfMapUpdateElem(m.fd, keyPtr, valuePtr, uint64(flags)); err != nil {
		return fmt.Errorf("update failed: %w", err)
	}

	return nil
}

// Delete removes a value.
//
// Returns ErrKeyNotExist if the key does not exist.
func (m *Map) Delete(key interface{}) error {
	keyPtr, err := marshalPtr(key, int(m.abi.KeySize))
	if err != nil {
		return fmt.Errorf("can't marshal key: %w", err)
	}

	if err = bpfMapDeleteElem(m.fd, keyPtr); err != nil {
		return fmt.Errorf("delete failed: %w", err)
	}
	return nil
}

// NextKey finds the key following an initial key.
//
// See NextKeyBytes for details.
//
// Returns ErrKeyNotExist if there is no next key.
func (m *Map) NextKey(key, nextKeyOut interface{}) error {
	nextKeyPtr, nextKeyBytes := makeBuffer(nextKeyOut, int(m.abi.KeySize))

	if err := m.nextKey(key, nextKeyPtr); err != nil {
		return err
	}

	if nextKeyBytes == nil {
		return nil
	}

	if err := unmarshalBytes(nextKeyOut, nextKeyBytes); err != nil {
		return fmt.Errorf("can't unmarshal next key: %w", err)
	}
	return nil
}

// NextKeyBytes returns the key following an initial key as a byte slice.
//
// Passing nil will return the first key.
//
// Use Iterate if you want to traverse all entries in the map.
//
// Returns nil if there are no more keys.
func (m *Map) NextKeyBytes(key interface{}) ([]byte, error) {
	nextKey := make([]byte, m.abi.KeySize)
	nextKeyPtr := internal.NewSlicePointer(nextKey)

	err := m.nextKey(key, nextKeyPtr)
	if errors.Is(err, ErrKeyNotExist) {
		return nil, nil
	}

	return nextKey, err
}

func (m *Map) nextKey(key interface{}, nextKeyOut internal.Pointer) error {
	var (
		keyPtr internal.Pointer
		err    error
	)

	if key != nil {
		keyPtr, err = marshalPtr(key, int(m.abi.KeySize))
		if err != nil {
			return fmt.Errorf("can't marshal key: %w", err)
		}
	}

	if err = bpfMapGetNextKey(m.fd, keyPtr, nextKeyOut); err != nil {
		return fmt.Errorf("next key failed: %w", err)
	}
	return nil
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

	return m.fd.Close()
}

// FD gets the file descriptor of the Map.
//
// Calling this function is invalid after Close has been called.
func (m *Map) FD() int {
	fd, err := m.fd.Value()
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

	dup, err := m.fd.Dup()
	if err != nil {
		return nil, fmt.Errorf("can't clone map: %w", err)
	}

	return newMap(dup, m.name, &m.abi)
}

// Pin persists the map past the lifetime of the process that created it.
//
// This requires bpffs to be mounted above fileName. See http://cilium.readthedocs.io/en/doc-1.0/kubernetes/install/#mounting-the-bpf-fs-optional
func (m *Map) Pin(fileName string) error {
	return internal.BPFObjPin(fileName, m.fd)
}

// Freeze prevents a map to be modified from user space.
//
// It makes no changes to kernel-side restrictions.
func (m *Map) Freeze() error {
	if err := haveMapMutabilityModifiers(); err != nil {
		return fmt.Errorf("can't freeze map: %w", err)
	}

	if err := bpfMapFreeze(m.fd); err != nil {
		return fmt.Errorf("can't freeze map: %w", err)
	}
	return nil
}

func (m *Map) populate(contents []MapKV) error {
	for _, kv := range contents {
		if err := m.Put(kv.Key, kv.Value); err != nil {
			return fmt.Errorf("key %v: %w", kv.Key, err)
		}
	}
	return nil
}

// LoadPinnedMap load a Map from a BPF file.
//
// The function is not compatible with nested maps.
// Use LoadPinnedMapExplicit in these situations.
func LoadPinnedMap(fileName string) (*Map, error) {
	fd, err := internal.BPFObjGet(fileName)
	if err != nil {
		return nil, err
	}
	name, abi, err := newMapABIFromFd(fd)
	if err != nil {
		_ = fd.Close()
		return nil, err
	}
	return newMap(fd, name, abi)
}

// LoadPinnedMapExplicit loads a map with explicit parameters.
func LoadPinnedMapExplicit(fileName string, abi *MapABI) (*Map, error) {
	fd, err := internal.BPFObjGet(fileName)
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
	return NewMapFromID(MapID(id))
}

// MarshalBinary implements BinaryMarshaler.
func (m *Map) MarshalBinary() ([]byte, error) {
	fd, err := m.fd.Value()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, 4)
	internal.NativeEndian.PutUint32(buf, fd)
	return buf, nil
}

func patchValue(value []byte, typ btf.Type, replacements map[string]interface{}) error {
	replaced := make(map[string]bool)
	replace := func(name string, offset, size int, replacement interface{}) error {
		if offset+size > len(value) {
			return fmt.Errorf("%s: offset %d(+%d) is out of bounds", name, offset, size)
		}

		buf, err := marshalBytes(replacement, size)
		if err != nil {
			return fmt.Errorf("marshal %s: %w", name, err)
		}

		copy(value[offset:offset+size], buf)
		replaced[name] = true
		return nil
	}

	switch parent := typ.(type) {
	case *btf.Datasec:
		for _, secinfo := range parent.Vars {
			name := string(secinfo.Type.(*btf.Var).Name)
			replacement, ok := replacements[name]
			if !ok {
				continue
			}

			err := replace(name, int(secinfo.Offset), int(secinfo.Size), replacement)
			if err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("patching %T is not supported", typ)
	}

	if len(replaced) == len(replacements) {
		return nil
	}

	var missing []string
	for name := range replacements {
		if !replaced[name] {
			missing = append(missing, name)
		}
	}

	if len(missing) == 1 {
		return fmt.Errorf("unknown field: %s", missing[0])
	}

	return fmt.Errorf("unknown fields: %s", strings.Join(missing, ","))
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
		if errors.Is(mi.err, ErrKeyNotExist) {
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

	mi.err = fmt.Errorf("%w", ErrIterationAborted)
	return false
}

// Err returns any encountered error.
//
// The method must be called after Next returns nil.
//
// Returns ErrIterationAborted if it wasn't possible to do a full iteration.
func (mi *MapIterator) Err() error {
	return mi.err
}

// MapGetNextID returns the ID of the next eBPF map.
//
// Returns ErrNotExist, if there is no next eBPF map.
func MapGetNextID(startID MapID) (MapID, error) {
	id, err := objGetNextID(internal.BPF_MAP_GET_NEXT_ID, uint32(startID))
	return MapID(id), err
}

// NewMapFromID returns the map for a given id.
//
// Returns ErrNotExist, if there is no eBPF map with the given id.
func NewMapFromID(id MapID) (*Map, error) {
	fd, err := bpfObjGetFDByID(internal.BPF_MAP_GET_FD_BY_ID, uint32(id))
	if err != nil {
		return nil, err
	}

	name, abi, err := newMapABIFromFd(fd)
	if err != nil {
		_ = fd.Close()
		return nil, err
	}

	return newMap(fd, name, abi)
}

// ID returns the systemwide unique ID of the map.
func (m *Map) ID() (MapID, error) {
	info, err := bpfGetMapInfoByFD(m.fd)
	if err != nil {
		return MapID(0), err
	}
	return MapID(info.id), nil
}
