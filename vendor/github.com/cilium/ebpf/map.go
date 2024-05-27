package ebpf

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/sysenc"
	"github.com/cilium/ebpf/internal/unix"
)

// Errors returned by Map and MapIterator methods.
var (
	ErrKeyNotExist      = errors.New("key does not exist")
	ErrKeyExist         = errors.New("key already exists")
	ErrIterationAborted = errors.New("iteration aborted")
	ErrMapIncompatible  = errors.New("map spec is incompatible with existing map")
	errMapNoBTFValue    = errors.New("map spec does not contain a BTF Value")
)

// MapOptions control loading a map into the kernel.
type MapOptions struct {
	// The base path to pin maps in if requested via PinByName.
	// Existing maps will be re-used if they are compatible, otherwise an
	// error is returned.
	PinPath        string
	LoadPinOptions LoadPinOptions
}

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

	// Flags is passed to the kernel and specifies additional map
	// creation attributes.
	Flags uint32

	// Automatically pin and load a map from MapOptions.PinPath.
	// Generates an error if an existing pinned map is incompatible with the MapSpec.
	Pinning PinType

	// Specify numa node during map creation
	// (effective only if unix.BPF_F_NUMA_NODE flag is set,
	// which can be imported from golang.org/x/sys/unix)
	NumaNode uint32

	// The initial contents of the map. May be nil.
	Contents []MapKV

	// Whether to freeze a map after setting its initial contents.
	Freeze bool

	// InnerMap is used as a template for ArrayOfMaps and HashOfMaps
	InnerMap *MapSpec

	// Extra trailing bytes found in the ELF map definition when using structs
	// larger than libbpf's bpf_map_def. nil if no trailing bytes were present.
	// Must be nil or empty before instantiating the MapSpec into a Map.
	Extra *bytes.Reader

	// The key and value type of this map. May be nil.
	Key, Value btf.Type
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

// fixupMagicFields fills fields of MapSpec which are usually
// left empty in ELF or which depend on runtime information.
//
// The method doesn't modify Spec, instead returning a copy.
// The copy is only performed if fixups are necessary, so callers mustn't mutate
// the returned spec.
func (spec *MapSpec) fixupMagicFields() (*MapSpec, error) {
	switch spec.Type {
	case ArrayOfMaps, HashOfMaps:
		if spec.ValueSize != 0 && spec.ValueSize != 4 {
			return nil, errors.New("ValueSize must be zero or four for map of map")
		}

		spec = spec.Copy()
		spec.ValueSize = 4

	case PerfEventArray:
		if spec.KeySize != 0 && spec.KeySize != 4 {
			return nil, errors.New("KeySize must be zero or four for perf event array")
		}

		if spec.ValueSize != 0 && spec.ValueSize != 4 {
			return nil, errors.New("ValueSize must be zero or four for perf event array")
		}

		spec = spec.Copy()
		spec.KeySize = 4
		spec.ValueSize = 4

		n, err := PossibleCPU()
		if err != nil {
			return nil, fmt.Errorf("fixup perf event array: %w", err)
		}

		if n := uint32(n); spec.MaxEntries == 0 || spec.MaxEntries > n {
			// MaxEntries should be zero most of the time, but there is code
			// out there which hardcodes large constants. Clamp the number
			// of entries to the number of CPUs at most. Allow creating maps with
			// less than n items since some kernel selftests relied on this
			// behaviour in the past.
			spec.MaxEntries = n
		}
	}

	return spec, nil
}

// dataSection returns the contents and BTF Datasec descriptor of the spec.
func (ms *MapSpec) dataSection() ([]byte, *btf.Datasec, error) {
	if ms.Value == nil {
		return nil, nil, errMapNoBTFValue
	}

	ds, ok := ms.Value.(*btf.Datasec)
	if !ok {
		return nil, nil, fmt.Errorf("map value BTF is a %T, not a *btf.Datasec", ms.Value)
	}

	if n := len(ms.Contents); n != 1 {
		return nil, nil, fmt.Errorf("expected one key, found %d", n)
	}

	kv := ms.Contents[0]
	value, ok := kv.Value.([]byte)
	if !ok {
		return nil, nil, fmt.Errorf("value at first map key is %T, not []byte", kv.Value)
	}

	return value, ds, nil
}

// MapKV is used to initialize the contents of a Map.
type MapKV struct {
	Key   interface{}
	Value interface{}
}

// Compatible returns nil if an existing map may be used instead of creating
// one from the spec.
//
// Returns an error wrapping [ErrMapIncompatible] otherwise.
func (ms *MapSpec) Compatible(m *Map) error {
	ms, err := ms.fixupMagicFields()
	if err != nil {
		return err
	}

	diffs := []string{}
	if m.typ != ms.Type {
		diffs = append(diffs, fmt.Sprintf("Type: %s changed to %s", m.typ, ms.Type))
	}
	if m.keySize != ms.KeySize {
		diffs = append(diffs, fmt.Sprintf("KeySize: %d changed to %d", m.keySize, ms.KeySize))
	}
	if m.valueSize != ms.ValueSize {
		diffs = append(diffs, fmt.Sprintf("ValueSize: %d changed to %d", m.valueSize, ms.ValueSize))
	}
	if m.maxEntries != ms.MaxEntries {
		diffs = append(diffs, fmt.Sprintf("MaxEntries: %d changed to %d", m.maxEntries, ms.MaxEntries))
	}

	// BPF_F_RDONLY_PROG is set unconditionally for devmaps. Explicitly allow this
	// mismatch.
	if !((ms.Type == DevMap || ms.Type == DevMapHash) && m.flags^ms.Flags == unix.BPF_F_RDONLY_PROG) &&
		m.flags != ms.Flags {
		diffs = append(diffs, fmt.Sprintf("Flags: %d changed to %d", m.flags, ms.Flags))
	}

	if len(diffs) == 0 {
		return nil
	}

	return fmt.Errorf("%s: %w", strings.Join(diffs, ", "), ErrMapIncompatible)
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
	name       string
	fd         *sys.FD
	typ        MapType
	keySize    uint32
	valueSize  uint32
	maxEntries uint32
	flags      uint32
	pinnedPath string
	// Per CPU maps return values larger than the size in the spec
	fullValueSize int
}

// NewMapFromFD creates a map from a raw fd.
//
// You should not use fd after calling this function.
func NewMapFromFD(fd int) (*Map, error) {
	f, err := sys.NewFD(fd)
	if err != nil {
		return nil, err
	}

	return newMapFromFD(f)
}

func newMapFromFD(fd *sys.FD) (*Map, error) {
	info, err := newMapInfoFromFd(fd)
	if err != nil {
		fd.Close()
		return nil, fmt.Errorf("get map info: %w", err)
	}

	return newMap(fd, info.Name, info.Type, info.KeySize, info.ValueSize, info.MaxEntries, info.Flags)
}

// NewMap creates a new Map.
//
// It's equivalent to calling NewMapWithOptions with default options.
func NewMap(spec *MapSpec) (*Map, error) {
	return NewMapWithOptions(spec, MapOptions{})
}

// NewMapWithOptions creates a new Map.
//
// Creating a map for the first time will perform feature detection
// by creating small, temporary maps.
//
// The caller is responsible for ensuring the process' rlimit is set
// sufficiently high for locking memory during map creation. This can be done
// by calling rlimit.RemoveMemlock() prior to calling NewMapWithOptions.
//
// May return an error wrapping ErrMapIncompatible.
func NewMapWithOptions(spec *MapSpec, opts MapOptions) (*Map, error) {
	m, err := newMapWithOptions(spec, opts)
	if err != nil {
		return nil, fmt.Errorf("creating map: %w", err)
	}

	if err := m.finalize(spec); err != nil {
		m.Close()
		return nil, fmt.Errorf("populating map: %w", err)
	}

	return m, nil
}

func newMapWithOptions(spec *MapSpec, opts MapOptions) (_ *Map, err error) {
	closeOnError := func(c io.Closer) {
		if err != nil {
			c.Close()
		}
	}

	switch spec.Pinning {
	case PinByName:
		if spec.Name == "" {
			return nil, fmt.Errorf("pin by name: missing Name")
		}

		if opts.PinPath == "" {
			return nil, fmt.Errorf("pin by name: missing MapOptions.PinPath")
		}

		path := filepath.Join(opts.PinPath, spec.Name)
		m, err := LoadPinnedMap(path, &opts.LoadPinOptions)
		if errors.Is(err, unix.ENOENT) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("load pinned map: %w", err)
		}
		defer closeOnError(m)

		if err := spec.Compatible(m); err != nil {
			return nil, fmt.Errorf("use pinned map %s: %w", spec.Name, err)
		}

		return m, nil

	case PinNone:
		// Nothing to do here

	default:
		return nil, fmt.Errorf("pin type %d: %w", int(spec.Pinning), ErrNotSupported)
	}

	var innerFd *sys.FD
	if spec.Type == ArrayOfMaps || spec.Type == HashOfMaps {
		if spec.InnerMap == nil {
			return nil, fmt.Errorf("%s requires InnerMap", spec.Type)
		}

		if spec.InnerMap.Pinning != PinNone {
			return nil, errors.New("inner maps cannot be pinned")
		}

		template, err := spec.InnerMap.createMap(nil, opts)
		if err != nil {
			return nil, fmt.Errorf("inner map: %w", err)
		}
		defer template.Close()

		// Intentionally skip populating and freezing (finalizing)
		// the inner map template since it will be removed shortly.

		innerFd = template.fd
	}

	m, err := spec.createMap(innerFd, opts)
	if err != nil {
		return nil, err
	}
	defer closeOnError(m)

	if spec.Pinning == PinByName {
		path := filepath.Join(opts.PinPath, spec.Name)
		if err := m.Pin(path); err != nil {
			return nil, fmt.Errorf("pin map to %s: %w", path, err)
		}
	}

	return m, nil
}

// createMap validates the spec's properties and creates the map in the kernel
// using the given opts. It does not populate or freeze the map.
func (spec *MapSpec) createMap(inner *sys.FD, opts MapOptions) (_ *Map, err error) {
	closeOnError := func(closer io.Closer) {
		if err != nil {
			closer.Close()
		}
	}

	// Kernels 4.13 through 5.4 used a struct bpf_map_def that contained
	// additional 'inner_map_idx' and later 'numa_node' fields.
	// In order to support loading these definitions, tolerate the presence of
	// extra bytes, but require them to be zeroes.
	if spec.Extra != nil {
		if _, err := io.Copy(internal.DiscardZeroes{}, spec.Extra); err != nil {
			return nil, errors.New("extra contains unhandled non-zero bytes, drain before creating map")
		}
	}

	spec, err = spec.fixupMagicFields()
	if err != nil {
		return nil, err
	}

	attr := sys.MapCreateAttr{
		MapType:    sys.MapType(spec.Type),
		KeySize:    spec.KeySize,
		ValueSize:  spec.ValueSize,
		MaxEntries: spec.MaxEntries,
		MapFlags:   sys.MapFlags(spec.Flags),
		NumaNode:   spec.NumaNode,
	}

	if inner != nil {
		attr.InnerMapFd = inner.Uint()
	}

	if haveObjName() == nil {
		attr.MapName = sys.NewObjName(spec.Name)
	}

	if spec.Key != nil || spec.Value != nil {
		handle, keyTypeID, valueTypeID, err := btf.MarshalMapKV(spec.Key, spec.Value)
		if err != nil && !errors.Is(err, btf.ErrNotSupported) {
			return nil, fmt.Errorf("load BTF: %w", err)
		}

		if handle != nil {
			defer handle.Close()

			// Use BTF k/v during map creation.
			attr.BtfFd = uint32(handle.FD())
			attr.BtfKeyTypeId = keyTypeID
			attr.BtfValueTypeId = valueTypeID
		}
	}

	fd, err := sys.MapCreate(&attr)

	// Some map types don't support BTF k/v in earlier kernel versions.
	// Remove BTF metadata and retry map creation.
	if (errors.Is(err, sys.ENOTSUPP) || errors.Is(err, unix.EINVAL)) && attr.BtfFd != 0 {
		attr.BtfFd, attr.BtfKeyTypeId, attr.BtfValueTypeId = 0, 0, 0
		fd, err = sys.MapCreate(&attr)
	}
	if err != nil {
		return nil, handleMapCreateError(attr, spec, err)
	}

	defer closeOnError(fd)
	m, err := newMap(fd, spec.Name, spec.Type, spec.KeySize, spec.ValueSize, spec.MaxEntries, spec.Flags)
	if err != nil {
		return nil, fmt.Errorf("map create: %w", err)
	}
	return m, nil
}

func handleMapCreateError(attr sys.MapCreateAttr, spec *MapSpec, err error) error {
	if errors.Is(err, unix.EPERM) {
		return fmt.Errorf("map create: %w (MEMLOCK may be too low, consider rlimit.RemoveMemlock)", err)
	}
	if errors.Is(err, unix.EINVAL) && spec.MaxEntries == 0 {
		return fmt.Errorf("map create: %w (MaxEntries may be incorrectly set to zero)", err)
	}
	if errors.Is(err, unix.EINVAL) && spec.Type == UnspecifiedMap {
		return fmt.Errorf("map create: cannot use type %s", UnspecifiedMap)
	}
	if errors.Is(err, unix.EINVAL) && spec.Flags&unix.BPF_F_NO_PREALLOC > 0 {
		return fmt.Errorf("map create: %w (noPrealloc flag may be incompatible with map type %s)", err, spec.Type)
	}

	switch spec.Type {
	case ArrayOfMaps, HashOfMaps:
		if haveFeatErr := haveNestedMaps(); haveFeatErr != nil {
			return fmt.Errorf("map create: %w", haveFeatErr)
		}
	}
	if spec.Flags&(unix.BPF_F_RDONLY_PROG|unix.BPF_F_WRONLY_PROG) > 0 || spec.Freeze {
		if haveFeatErr := haveMapMutabilityModifiers(); haveFeatErr != nil {
			return fmt.Errorf("map create: %w", haveFeatErr)
		}
	}
	if spec.Flags&unix.BPF_F_MMAPABLE > 0 {
		if haveFeatErr := haveMmapableMaps(); haveFeatErr != nil {
			return fmt.Errorf("map create: %w", haveFeatErr)
		}
	}
	if spec.Flags&unix.BPF_F_INNER_MAP > 0 {
		if haveFeatErr := haveInnerMaps(); haveFeatErr != nil {
			return fmt.Errorf("map create: %w", haveFeatErr)
		}
	}
	if spec.Flags&unix.BPF_F_NO_PREALLOC > 0 {
		if haveFeatErr := haveNoPreallocMaps(); haveFeatErr != nil {
			return fmt.Errorf("map create: %w", haveFeatErr)
		}
	}
	// BPF_MAP_TYPE_RINGBUF's max_entries must be a power-of-2 multiple of kernel's page size.
	if errors.Is(err, unix.EINVAL) &&
		(attr.MapType == sys.BPF_MAP_TYPE_RINGBUF || attr.MapType == sys.BPF_MAP_TYPE_USER_RINGBUF) {
		pageSize := uint32(os.Getpagesize())
		maxEntries := attr.MaxEntries
		if maxEntries%pageSize != 0 || !internal.IsPow(maxEntries) {
			return fmt.Errorf("map create: %w (ring map size %d not a multiple of page size %d)", err, maxEntries, pageSize)
		}
	}
	if attr.BtfFd == 0 {
		return fmt.Errorf("map create: %w (without BTF k/v)", err)
	}

	return fmt.Errorf("map create: %w", err)
}

// newMap allocates and returns a new Map structure.
// Sets the fullValueSize on per-CPU maps.
func newMap(fd *sys.FD, name string, typ MapType, keySize, valueSize, maxEntries, flags uint32) (*Map, error) {
	m := &Map{
		name,
		fd,
		typ,
		keySize,
		valueSize,
		maxEntries,
		flags,
		"",
		int(valueSize),
	}

	if !typ.hasPerCPUValue() {
		return m, nil
	}

	possibleCPUs, err := PossibleCPU()
	if err != nil {
		return nil, err
	}

	m.fullValueSize = int(internal.Align(valueSize, 8)) * possibleCPUs
	return m, nil
}

func (m *Map) String() string {
	if m.name != "" {
		return fmt.Sprintf("%s(%s)#%v", m.typ, m.name, m.fd)
	}
	return fmt.Sprintf("%s#%v", m.typ, m.fd)
}

// Type returns the underlying type of the map.
func (m *Map) Type() MapType {
	return m.typ
}

// KeySize returns the size of the map key in bytes.
func (m *Map) KeySize() uint32 {
	return m.keySize
}

// ValueSize returns the size of the map value in bytes.
func (m *Map) ValueSize() uint32 {
	return m.valueSize
}

// MaxEntries returns the maximum number of elements the map can hold.
func (m *Map) MaxEntries() uint32 {
	return m.maxEntries
}

// Flags returns the flags of the map.
func (m *Map) Flags() uint32 {
	return m.flags
}

// Info returns metadata about the map.
func (m *Map) Info() (*MapInfo, error) {
	return newMapInfoFromFd(m.fd)
}

// MapLookupFlags controls the behaviour of the map lookup calls.
type MapLookupFlags uint64

// LookupLock look up the value of a spin-locked map.
const LookupLock MapLookupFlags = unix.BPF_F_LOCK

// Lookup retrieves a value from a Map.
//
// Calls Close() on valueOut if it is of type **Map or **Program,
// and *valueOut is not nil.
//
// Returns an error if the key doesn't exist, see ErrKeyNotExist.
func (m *Map) Lookup(key, valueOut interface{}) error {
	return m.LookupWithFlags(key, valueOut, 0)
}

// LookupWithFlags retrieves a value from a Map with flags.
//
// Passing LookupLock flag will look up the value of a spin-locked
// map without returning the lock. This must be specified if the
// elements contain a spinlock.
//
// Calls Close() on valueOut if it is of type **Map or **Program,
// and *valueOut is not nil.
//
// Returns an error if the key doesn't exist, see ErrKeyNotExist.
func (m *Map) LookupWithFlags(key, valueOut interface{}, flags MapLookupFlags) error {
	if m.typ.hasPerCPUValue() {
		return m.lookupPerCPU(key, valueOut, flags)
	}

	valueBytes := makeMapSyscallOutput(valueOut, m.fullValueSize)
	if err := m.lookup(key, valueBytes.Pointer(), flags); err != nil {
		return err
	}

	return m.unmarshalValue(valueOut, valueBytes)
}

// LookupAndDelete retrieves and deletes a value from a Map.
//
// Returns ErrKeyNotExist if the key doesn't exist.
func (m *Map) LookupAndDelete(key, valueOut interface{}) error {
	return m.LookupAndDeleteWithFlags(key, valueOut, 0)
}

// LookupAndDeleteWithFlags retrieves and deletes a value from a Map.
//
// Passing LookupLock flag will look up and delete the value of a spin-locked
// map without returning the lock. This must be specified if the elements
// contain a spinlock.
//
// Returns ErrKeyNotExist if the key doesn't exist.
func (m *Map) LookupAndDeleteWithFlags(key, valueOut interface{}, flags MapLookupFlags) error {
	if m.typ.hasPerCPUValue() {
		return m.lookupAndDeletePerCPU(key, valueOut, flags)
	}

	valueBytes := makeMapSyscallOutput(valueOut, m.fullValueSize)
	if err := m.lookupAndDelete(key, valueBytes.Pointer(), flags); err != nil {
		return err
	}
	return m.unmarshalValue(valueOut, valueBytes)
}

// LookupBytes gets a value from Map.
//
// Returns a nil value if a key doesn't exist.
func (m *Map) LookupBytes(key interface{}) ([]byte, error) {
	valueBytes := make([]byte, m.fullValueSize)
	valuePtr := sys.NewSlicePointer(valueBytes)

	err := m.lookup(key, valuePtr, 0)
	if errors.Is(err, ErrKeyNotExist) {
		return nil, nil
	}

	return valueBytes, err
}

func (m *Map) lookupPerCPU(key, valueOut any, flags MapLookupFlags) error {
	slice, err := ensurePerCPUSlice(valueOut, int(m.valueSize))
	if err != nil {
		return err
	}
	valueBytes := make([]byte, m.fullValueSize)
	if err := m.lookup(key, sys.NewSlicePointer(valueBytes), flags); err != nil {
		return err
	}
	return unmarshalPerCPUValue(slice, int(m.valueSize), valueBytes)
}

func (m *Map) lookup(key interface{}, valueOut sys.Pointer, flags MapLookupFlags) error {
	keyPtr, err := m.marshalKey(key)
	if err != nil {
		return fmt.Errorf("can't marshal key: %w", err)
	}

	attr := sys.MapLookupElemAttr{
		MapFd: m.fd.Uint(),
		Key:   keyPtr,
		Value: valueOut,
		Flags: uint64(flags),
	}

	if err = sys.MapLookupElem(&attr); err != nil {
		return fmt.Errorf("lookup: %w", wrapMapError(err))
	}
	return nil
}

func (m *Map) lookupAndDeletePerCPU(key, valueOut any, flags MapLookupFlags) error {
	slice, err := ensurePerCPUSlice(valueOut, int(m.valueSize))
	if err != nil {
		return err
	}
	valueBytes := make([]byte, m.fullValueSize)
	if err := m.lookupAndDelete(key, sys.NewSlicePointer(valueBytes), flags); err != nil {
		return err
	}
	return unmarshalPerCPUValue(slice, int(m.valueSize), valueBytes)
}

// ensurePerCPUSlice allocates a slice for a per-CPU value if necessary.
func ensurePerCPUSlice(sliceOrPtr any, elemLength int) (any, error) {
	sliceOrPtrType := reflect.TypeOf(sliceOrPtr)
	if sliceOrPtrType.Kind() == reflect.Slice {
		// The target is a slice, the caller is responsible for ensuring that
		// size is correct.
		return sliceOrPtr, nil
	}

	slicePtrType := sliceOrPtrType
	if slicePtrType.Kind() != reflect.Ptr || slicePtrType.Elem().Kind() != reflect.Slice {
		return nil, fmt.Errorf("per-cpu value requires a slice or a pointer to slice")
	}

	possibleCPUs, err := PossibleCPU()
	if err != nil {
		return nil, err
	}

	sliceType := slicePtrType.Elem()
	slice := reflect.MakeSlice(sliceType, possibleCPUs, possibleCPUs)

	sliceElemType := sliceType.Elem()
	sliceElemIsPointer := sliceElemType.Kind() == reflect.Ptr
	reflect.ValueOf(sliceOrPtr).Elem().Set(slice)
	if !sliceElemIsPointer {
		return slice.Interface(), nil
	}
	sliceElemType = sliceElemType.Elem()

	for i := 0; i < possibleCPUs; i++ {
		newElem := reflect.New(sliceElemType)
		slice.Index(i).Set(newElem)
	}

	return slice.Interface(), nil
}

func (m *Map) lookupAndDelete(key any, valuePtr sys.Pointer, flags MapLookupFlags) error {
	keyPtr, err := m.marshalKey(key)
	if err != nil {
		return fmt.Errorf("can't marshal key: %w", err)
	}

	attr := sys.MapLookupAndDeleteElemAttr{
		MapFd: m.fd.Uint(),
		Key:   keyPtr,
		Value: valuePtr,
		Flags: uint64(flags),
	}

	if err := sys.MapLookupAndDeleteElem(&attr); err != nil {
		return fmt.Errorf("lookup and delete: %w", wrapMapError(err))
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
	// UpdateLock updates elements under bpf_spin_lock.
	UpdateLock
)

// Put replaces or creates a value in map.
//
// It is equivalent to calling Update with UpdateAny.
func (m *Map) Put(key, value interface{}) error {
	return m.Update(key, value, UpdateAny)
}

// Update changes the value of a key.
func (m *Map) Update(key, value any, flags MapUpdateFlags) error {
	if m.typ.hasPerCPUValue() {
		return m.updatePerCPU(key, value, flags)
	}

	valuePtr, err := m.marshalValue(value)
	if err != nil {
		return fmt.Errorf("marshal value: %w", err)
	}

	return m.update(key, valuePtr, flags)
}

func (m *Map) updatePerCPU(key, value any, flags MapUpdateFlags) error {
	valuePtr, err := marshalPerCPUValue(value, int(m.valueSize))
	if err != nil {
		return fmt.Errorf("marshal value: %w", err)
	}

	return m.update(key, valuePtr, flags)
}

func (m *Map) update(key any, valuePtr sys.Pointer, flags MapUpdateFlags) error {
	keyPtr, err := m.marshalKey(key)
	if err != nil {
		return fmt.Errorf("marshal key: %w", err)
	}

	attr := sys.MapUpdateElemAttr{
		MapFd: m.fd.Uint(),
		Key:   keyPtr,
		Value: valuePtr,
		Flags: uint64(flags),
	}

	if err = sys.MapUpdateElem(&attr); err != nil {
		return fmt.Errorf("update: %w", wrapMapError(err))
	}

	return nil
}

// Delete removes a value.
//
// Returns ErrKeyNotExist if the key does not exist.
func (m *Map) Delete(key interface{}) error {
	keyPtr, err := m.marshalKey(key)
	if err != nil {
		return fmt.Errorf("can't marshal key: %w", err)
	}

	attr := sys.MapDeleteElemAttr{
		MapFd: m.fd.Uint(),
		Key:   keyPtr,
	}

	if err = sys.MapDeleteElem(&attr); err != nil {
		return fmt.Errorf("delete: %w", wrapMapError(err))
	}
	return nil
}

// NextKey finds the key following an initial key.
//
// See NextKeyBytes for details.
//
// Returns ErrKeyNotExist if there is no next key.
func (m *Map) NextKey(key, nextKeyOut interface{}) error {
	nextKeyBytes := makeMapSyscallOutput(nextKeyOut, int(m.keySize))

	if err := m.nextKey(key, nextKeyBytes.Pointer()); err != nil {
		return err
	}

	if err := nextKeyBytes.Unmarshal(nextKeyOut); err != nil {
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
	nextKey := make([]byte, m.keySize)
	nextKeyPtr := sys.NewSlicePointer(nextKey)

	err := m.nextKey(key, nextKeyPtr)
	if errors.Is(err, ErrKeyNotExist) {
		return nil, nil
	}

	return nextKey, err
}

func (m *Map) nextKey(key interface{}, nextKeyOut sys.Pointer) error {
	var (
		keyPtr sys.Pointer
		err    error
	)

	if key != nil {
		keyPtr, err = m.marshalKey(key)
		if err != nil {
			return fmt.Errorf("can't marshal key: %w", err)
		}
	}

	attr := sys.MapGetNextKeyAttr{
		MapFd:   m.fd.Uint(),
		Key:     keyPtr,
		NextKey: nextKeyOut,
	}

	if err = sys.MapGetNextKey(&attr); err != nil {
		// Kernels 4.4.131 and earlier return EFAULT instead of a pointer to the
		// first map element when a nil key pointer is specified.
		if key == nil && errors.Is(err, unix.EFAULT) {
			var guessKey []byte
			guessKey, err = m.guessNonExistentKey()
			if err != nil {
				return err
			}

			// Retry the syscall with a valid non-existing key.
			attr.Key = sys.NewSlicePointer(guessKey)
			if err = sys.MapGetNextKey(&attr); err == nil {
				return nil
			}
		}

		return fmt.Errorf("next key: %w", wrapMapError(err))
	}

	return nil
}

var mmapProtectedPage = sync.OnceValues(func() ([]byte, error) {
	return unix.Mmap(-1, 0, os.Getpagesize(), unix.PROT_NONE, unix.MAP_ANON|unix.MAP_SHARED)
})

// guessNonExistentKey attempts to perform a map lookup that returns ENOENT.
// This is necessary on kernels before 4.4.132, since those don't support
// iterating maps from the start by providing an invalid key pointer.
func (m *Map) guessNonExistentKey() ([]byte, error) {
	// Map a protected page and use that as the value pointer. This saves some
	// work copying out the value, which we're not interested in.
	page, err := mmapProtectedPage()
	if err != nil {
		return nil, err
	}
	valuePtr := sys.NewSlicePointer(page)

	randKey := make([]byte, int(m.keySize))

	for i := 0; i < 4; i++ {
		switch i {
		// For hash maps, the 0 key is less likely to be occupied. They're often
		// used for storing data related to pointers, and their access pattern is
		// generally scattered across the keyspace.
		case 0:
		// An all-0xff key is guaranteed to be out of bounds of any array, since
		// those have a fixed key size of 4 bytes. The only corner case being
		// arrays with 2^32 max entries, but those are prohibitively expensive
		// in many environments.
		case 1:
			for r := range randKey {
				randKey[r] = 0xff
			}
		// Inspired by BCC, 0x55 is an alternating binary pattern (0101), so
		// is unlikely to be taken.
		case 2:
			for r := range randKey {
				randKey[r] = 0x55
			}
		// Last ditch effort, generate a random key.
		case 3:
			rand.New(rand.NewSource(time.Now().UnixNano())).Read(randKey)
		}

		err := m.lookup(randKey, valuePtr, 0)
		if errors.Is(err, ErrKeyNotExist) {
			return randKey, nil
		}
	}

	return nil, errors.New("couldn't find non-existing key")
}

// BatchLookup looks up many elements in a map at once.
//
// "keysOut" and "valuesOut" must be of type slice, a pointer
// to a slice or buffer will not work.
// "cursor" is an pointer to an opaque handle. It must be non-nil. Pass
// "cursor" to subsequent calls of this function to continue the batching
// operation in the case of chunking.
//
// Warning: This API is not very safe to use as the kernel implementation for
// batching relies on the user to be aware of subtle details with regarding to
// different map type implementations.
//
// ErrKeyNotExist is returned when the batch lookup has reached
// the end of all possible results, even when partial results
// are returned. It should be used to evaluate when lookup is "done".
func (m *Map) BatchLookup(cursor *MapBatchCursor, keysOut, valuesOut interface{}, opts *BatchOptions) (int, error) {
	return m.batchLookup(sys.BPF_MAP_LOOKUP_BATCH, cursor, keysOut, valuesOut, opts)
}

// BatchLookupAndDelete looks up many elements in a map at once,
//
// It then deletes all those elements.
// "keysOut" and "valuesOut" must be of type slice, a pointer
// to a slice or buffer will not work.
// "cursor" is an pointer to an opaque handle. It must be non-nil. Pass
// "cursor" to subsequent calls of this function to continue the batching
// operation in the case of chunking.
//
// Warning: This API is not very safe to use as the kernel implementation for
// batching relies on the user to be aware of subtle details with regarding to
// different map type implementations.
//
// ErrKeyNotExist is returned when the batch lookup has reached
// the end of all possible results, even when partial results
// are returned. It should be used to evaluate when lookup is "done".
func (m *Map) BatchLookupAndDelete(cursor *MapBatchCursor, keysOut, valuesOut interface{}, opts *BatchOptions) (int, error) {
	return m.batchLookup(sys.BPF_MAP_LOOKUP_AND_DELETE_BATCH, cursor, keysOut, valuesOut, opts)
}

// MapBatchCursor represents a starting point for a batch operation.
type MapBatchCursor struct {
	m      *Map
	opaque []byte
}

func (m *Map) batchLookup(cmd sys.Cmd, cursor *MapBatchCursor, keysOut, valuesOut interface{}, opts *BatchOptions) (int, error) {
	if m.typ.hasPerCPUValue() {
		return m.batchLookupPerCPU(cmd, cursor, keysOut, valuesOut, opts)
	}

	count, err := batchCount(keysOut, valuesOut)
	if err != nil {
		return 0, err
	}

	valueBuf := sysenc.SyscallOutput(valuesOut, count*int(m.fullValueSize))

	n, err := m.batchLookupCmd(cmd, cursor, count, keysOut, valueBuf.Pointer(), opts)
	if err != nil {
		return n, err
	}

	err = valueBuf.Unmarshal(valuesOut)
	if err != nil {
		return 0, err
	}

	return n, nil
}

func (m *Map) batchLookupPerCPU(cmd sys.Cmd, cursor *MapBatchCursor, keysOut, valuesOut interface{}, opts *BatchOptions) (int, error) {
	count, err := sliceLen(keysOut)
	if err != nil {
		return 0, fmt.Errorf("keys: %w", err)
	}

	valueBuf := make([]byte, count*int(m.fullValueSize))
	valuePtr := sys.NewSlicePointer(valueBuf)

	n, sysErr := m.batchLookupCmd(cmd, cursor, count, keysOut, valuePtr, opts)
	if sysErr != nil && !errors.Is(sysErr, unix.ENOENT) {
		return 0, err
	}

	err = unmarshalBatchPerCPUValue(valuesOut, count, int(m.valueSize), valueBuf)
	if err != nil {
		return 0, err
	}

	return n, sysErr
}

func (m *Map) batchLookupCmd(cmd sys.Cmd, cursor *MapBatchCursor, count int, keysOut any, valuePtr sys.Pointer, opts *BatchOptions) (int, error) {
	cursorLen := int(m.keySize)
	if cursorLen < 4 {
		// * generic_map_lookup_batch requires that batch_out is key_size bytes.
		//   This is used by array and LPM maps.
		//
		// * __htab_map_lookup_and_delete_batch requires u32. This is used by the
		//   various hash maps.
		//
		// Use a minimum of 4 bytes to avoid having to distinguish between the two.
		cursorLen = 4
	}

	inBatch := cursor.opaque
	if inBatch == nil {
		// This is the first lookup, allocate a buffer to hold the cursor.
		cursor.opaque = make([]byte, cursorLen)
		cursor.m = m
	} else if cursor.m != m {
		// Prevent reuse of a cursor across maps. First, it's unlikely to work.
		// Second, the maps may require different cursorLen and cursor.opaque
		// may therefore be too short. This could lead to the kernel clobbering
		// user space memory.
		return 0, errors.New("a cursor may not be reused across maps")
	}

	if err := haveBatchAPI(); err != nil {
		return 0, err
	}

	keyBuf := sysenc.SyscallOutput(keysOut, count*int(m.keySize))

	attr := sys.MapLookupBatchAttr{
		MapFd:    m.fd.Uint(),
		Keys:     keyBuf.Pointer(),
		Values:   valuePtr,
		Count:    uint32(count),
		InBatch:  sys.NewSlicePointer(inBatch),
		OutBatch: sys.NewSlicePointer(cursor.opaque),
	}

	if opts != nil {
		attr.ElemFlags = opts.ElemFlags
		attr.Flags = opts.Flags
	}

	_, sysErr := sys.BPF(cmd, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	sysErr = wrapMapError(sysErr)
	if sysErr != nil && !errors.Is(sysErr, unix.ENOENT) {
		return 0, sysErr
	}

	if err := keyBuf.Unmarshal(keysOut); err != nil {
		return 0, err
	}

	return int(attr.Count), sysErr
}

// BatchUpdate updates the map with multiple keys and values
// simultaneously.
// "keys" and "values" must be of type slice, a pointer
// to a slice or buffer will not work.
func (m *Map) BatchUpdate(keys, values interface{}, opts *BatchOptions) (int, error) {
	if m.typ.hasPerCPUValue() {
		return m.batchUpdatePerCPU(keys, values, opts)
	}

	count, err := batchCount(keys, values)
	if err != nil {
		return 0, err
	}

	valuePtr, err := marshalMapSyscallInput(values, count*int(m.valueSize))
	if err != nil {
		return 0, err
	}

	return m.batchUpdate(count, keys, valuePtr, opts)
}

func (m *Map) batchUpdate(count int, keys any, valuePtr sys.Pointer, opts *BatchOptions) (int, error) {
	keyPtr, err := marshalMapSyscallInput(keys, count*int(m.keySize))
	if err != nil {
		return 0, err
	}

	attr := sys.MapUpdateBatchAttr{
		MapFd:  m.fd.Uint(),
		Keys:   keyPtr,
		Values: valuePtr,
		Count:  uint32(count),
	}
	if opts != nil {
		attr.ElemFlags = opts.ElemFlags
		attr.Flags = opts.Flags
	}

	err = sys.MapUpdateBatch(&attr)
	if err != nil {
		if haveFeatErr := haveBatchAPI(); haveFeatErr != nil {
			return 0, haveFeatErr
		}
		return int(attr.Count), fmt.Errorf("batch update: %w", wrapMapError(err))
	}

	return int(attr.Count), nil
}

func (m *Map) batchUpdatePerCPU(keys, values any, opts *BatchOptions) (int, error) {
	count, err := sliceLen(keys)
	if err != nil {
		return 0, fmt.Errorf("keys: %w", err)
	}

	valueBuf, err := marshalBatchPerCPUValue(values, count, int(m.valueSize))
	if err != nil {
		return 0, err
	}

	return m.batchUpdate(count, keys, sys.NewSlicePointer(valueBuf), opts)
}

// BatchDelete batch deletes entries in the map by keys.
// "keys" must be of type slice, a pointer to a slice or buffer will not work.
func (m *Map) BatchDelete(keys interface{}, opts *BatchOptions) (int, error) {
	count, err := sliceLen(keys)
	if err != nil {
		return 0, fmt.Errorf("keys: %w", err)
	}

	keyPtr, err := marshalMapSyscallInput(keys, count*int(m.keySize))
	if err != nil {
		return 0, fmt.Errorf("cannot marshal keys: %v", err)
	}

	attr := sys.MapDeleteBatchAttr{
		MapFd: m.fd.Uint(),
		Keys:  keyPtr,
		Count: uint32(count),
	}

	if opts != nil {
		attr.ElemFlags = opts.ElemFlags
		attr.Flags = opts.Flags
	}

	if err = sys.MapDeleteBatch(&attr); err != nil {
		if haveFeatErr := haveBatchAPI(); haveFeatErr != nil {
			return 0, haveFeatErr
		}
		return int(attr.Count), fmt.Errorf("batch delete: %w", wrapMapError(err))
	}

	return int(attr.Count), nil
}

func batchCount(keys, values any) (int, error) {
	keysLen, err := sliceLen(keys)
	if err != nil {
		return 0, fmt.Errorf("keys: %w", err)
	}

	valuesLen, err := sliceLen(values)
	if err != nil {
		return 0, fmt.Errorf("values: %w", err)
	}

	if keysLen != valuesLen {
		return 0, fmt.Errorf("keys and values must have the same length")
	}

	return keysLen, nil
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

// Close the Map's underlying file descriptor, which could unload the
// Map from the kernel if it is not pinned or in use by a loaded Program.
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
	return m.fd.Int()
}

// Clone creates a duplicate of the Map.
//
// Closing the duplicate does not affect the original, and vice versa.
// Changes made to the map are reflected by both instances however.
// If the original map was pinned, the cloned map will not be pinned by default.
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

	return &Map{
		m.name,
		dup,
		m.typ,
		m.keySize,
		m.valueSize,
		m.maxEntries,
		m.flags,
		"",
		m.fullValueSize,
	}, nil
}

// Pin persists the map on the BPF virtual file system past the lifetime of
// the process that created it .
//
// Calling Pin on a previously pinned map will overwrite the path, except when
// the new path already exists. Re-pinning across filesystems is not supported.
// You can Clone a map to pin it to a different path.
//
// This requires bpffs to be mounted above fileName.
// See https://docs.cilium.io/en/stable/network/kubernetes/configuration/#mounting-bpffs-with-systemd
func (m *Map) Pin(fileName string) error {
	if err := internal.Pin(m.pinnedPath, fileName, m.fd); err != nil {
		return err
	}
	m.pinnedPath = fileName
	return nil
}

// Unpin removes the persisted state for the map from the BPF virtual filesystem.
//
// Failed calls to Unpin will not alter the state returned by IsPinned.
//
// Unpinning an unpinned Map returns nil.
func (m *Map) Unpin() error {
	if err := internal.Unpin(m.pinnedPath); err != nil {
		return err
	}
	m.pinnedPath = ""
	return nil
}

// IsPinned returns true if the map has a non-empty pinned path.
func (m *Map) IsPinned() bool {
	return m.pinnedPath != ""
}

// Freeze prevents a map to be modified from user space.
//
// It makes no changes to kernel-side restrictions.
func (m *Map) Freeze() error {
	attr := sys.MapFreezeAttr{
		MapFd: m.fd.Uint(),
	}

	if err := sys.MapFreeze(&attr); err != nil {
		if haveFeatErr := haveMapMutabilityModifiers(); haveFeatErr != nil {
			return fmt.Errorf("can't freeze map: %w", haveFeatErr)
		}
		return fmt.Errorf("can't freeze map: %w", err)
	}
	return nil
}

// finalize populates the Map according to the Contents specified
// in spec and freezes the Map if requested by spec.
func (m *Map) finalize(spec *MapSpec) error {
	for _, kv := range spec.Contents {
		if err := m.Put(kv.Key, kv.Value); err != nil {
			return fmt.Errorf("putting value: key %v: %w", kv.Key, err)
		}
	}

	if spec.Freeze {
		if err := m.Freeze(); err != nil {
			return fmt.Errorf("freezing map: %w", err)
		}
	}

	return nil
}

func (m *Map) marshalKey(data interface{}) (sys.Pointer, error) {
	if data == nil {
		if m.keySize == 0 {
			// Queues have a key length of zero, so passing nil here is valid.
			return sys.NewPointer(nil), nil
		}
		return sys.Pointer{}, errors.New("can't use nil as key of map")
	}

	return marshalMapSyscallInput(data, int(m.keySize))
}

func (m *Map) marshalValue(data interface{}) (sys.Pointer, error) {
	var (
		buf []byte
		err error
	)

	switch value := data.(type) {
	case *Map:
		if !m.typ.canStoreMap() {
			return sys.Pointer{}, fmt.Errorf("can't store map in %s", m.typ)
		}
		buf, err = marshalMap(value, int(m.valueSize))

	case *Program:
		if !m.typ.canStoreProgram() {
			return sys.Pointer{}, fmt.Errorf("can't store program in %s", m.typ)
		}
		buf, err = marshalProgram(value, int(m.valueSize))

	default:
		return marshalMapSyscallInput(data, int(m.valueSize))
	}

	if err != nil {
		return sys.Pointer{}, err
	}

	return sys.NewSlicePointer(buf), nil
}

func (m *Map) unmarshalValue(value any, buf sysenc.Buffer) error {
	switch value := value.(type) {
	case **Map:
		if !m.typ.canStoreMap() {
			return fmt.Errorf("can't read a map from %s", m.typ)
		}

		other, err := unmarshalMap(buf)
		if err != nil {
			return err
		}

		// The caller might close the map externally, so ignore errors.
		_ = (*value).Close()

		*value = other
		return nil

	case *Map:
		if !m.typ.canStoreMap() {
			return fmt.Errorf("can't read a map from %s", m.typ)
		}
		return errors.New("require pointer to *Map")

	case **Program:
		if !m.typ.canStoreProgram() {
			return fmt.Errorf("can't read a program from %s", m.typ)
		}

		other, err := unmarshalProgram(buf)
		if err != nil {
			return err
		}

		// The caller might close the program externally, so ignore errors.
		_ = (*value).Close()

		*value = other
		return nil

	case *Program:
		if !m.typ.canStoreProgram() {
			return fmt.Errorf("can't read a program from %s", m.typ)
		}
		return errors.New("require pointer to *Program")
	}

	return buf.Unmarshal(value)
}

// LoadPinnedMap loads a Map from a BPF file.
func LoadPinnedMap(fileName string, opts *LoadPinOptions) (*Map, error) {
	fd, err := sys.ObjGet(&sys.ObjGetAttr{
		Pathname:  sys.NewStringPointer(fileName),
		FileFlags: opts.Marshal(),
	})
	if err != nil {
		return nil, err
	}

	m, err := newMapFromFD(fd)
	if err == nil {
		m.pinnedPath = fileName
	}

	return m, err
}

// unmarshalMap creates a map from a map ID encoded in host endianness.
func unmarshalMap(buf sysenc.Buffer) (*Map, error) {
	var id uint32
	if err := buf.Unmarshal(&id); err != nil {
		return nil, err
	}
	return NewMapFromID(MapID(id))
}

// marshalMap marshals the fd of a map into a buffer in host endianness.
func marshalMap(m *Map, length int) ([]byte, error) {
	if length != 4 {
		return nil, fmt.Errorf("can't marshal map to %d bytes", length)
	}

	buf := make([]byte, 4)
	internal.NativeEndian.PutUint32(buf, m.fd.Uint())
	return buf, nil
}

// MapIterator iterates a Map.
//
// See Map.Iterate.
type MapIterator struct {
	target *Map
	// Temporary storage to avoid allocations in Next(). This is any instead
	// of []byte to avoid allocations.
	cursor            any
	count, maxEntries uint32
	done              bool
	err               error
}

func newMapIterator(target *Map) *MapIterator {
	return &MapIterator{
		target:     target,
		maxEntries: target.maxEntries,
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

	// For array-like maps NextKey returns nil only after maxEntries
	// iterations.
	for mi.count <= mi.maxEntries {
		if mi.cursor == nil {
			// Pass nil interface to NextKey to make sure the Map's first key
			// is returned. If we pass an uninitialized []byte instead, it'll see a
			// non-nil interface and try to marshal it.
			mi.cursor = make([]byte, mi.target.keySize)
			mi.err = mi.target.NextKey(nil, mi.cursor)
		} else {
			mi.err = mi.target.NextKey(mi.cursor, mi.cursor)
		}

		if errors.Is(mi.err, ErrKeyNotExist) {
			mi.done = true
			mi.err = nil
			return false
		} else if mi.err != nil {
			mi.err = fmt.Errorf("get next key: %w", mi.err)
			return false
		}

		mi.count++
		mi.err = mi.target.Lookup(mi.cursor, valueOut)
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
			mi.err = fmt.Errorf("look up next key: %w", mi.err)
			return false
		}

		buf := mi.cursor.([]byte)
		if ptr, ok := keyOut.(unsafe.Pointer); ok {
			copy(unsafe.Slice((*byte)(ptr), len(buf)), buf)
		} else {
			mi.err = sysenc.Unmarshal(keyOut, buf)
		}

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
	attr := &sys.MapGetNextIdAttr{Id: uint32(startID)}
	return MapID(attr.NextId), sys.MapGetNextId(attr)
}

// NewMapFromID returns the map for a given id.
//
// Returns ErrNotExist, if there is no eBPF map with the given id.
func NewMapFromID(id MapID) (*Map, error) {
	fd, err := sys.MapGetFdById(&sys.MapGetFdByIdAttr{
		Id: uint32(id),
	})
	if err != nil {
		return nil, err
	}

	return newMapFromFD(fd)
}

// sliceLen returns the length if the value is a slice or an error otherwise.
func sliceLen(slice any) (int, error) {
	sliceValue := reflect.ValueOf(slice)
	if sliceValue.Kind() != reflect.Slice {
		return 0, fmt.Errorf("%T is not a slice", slice)
	}
	return sliceValue.Len(), nil
}
