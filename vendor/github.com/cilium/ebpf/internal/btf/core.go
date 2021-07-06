package btf

import (
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// Code in this file is derived from libbpf, which is available under a BSD
// 2-Clause license.

// Relocation describes a CO-RE relocation.
type Relocation struct {
	Current uint32
	New     uint32
}

func (r Relocation) equal(other Relocation) bool {
	return r.Current == other.Current && r.New == other.New
}

// coreReloKind is the type of CO-RE relocation
type coreReloKind uint32

const (
	reloFieldByteOffset coreReloKind = iota /* field byte offset */
	reloFieldByteSize                       /* field size in bytes */
	reloFieldExists                         /* field existence in target kernel */
	reloFieldSigned                         /* field signedness (0 - unsigned, 1 - signed) */
	reloFieldLShiftU64                      /* bitfield-specific left bitshift */
	reloFieldRShiftU64                      /* bitfield-specific right bitshift */
	reloTypeIDLocal                         /* type ID in local BPF object */
	reloTypeIDTarget                        /* type ID in target kernel */
	reloTypeExists                          /* type existence in target kernel */
	reloTypeSize                            /* type size in bytes */
	reloEnumvalExists                       /* enum value existence in target kernel */
	reloEnumvalValue                        /* enum value integer value */
)

func (k coreReloKind) String() string {
	switch k {
	case reloFieldByteOffset:
		return "byte_off"
	case reloFieldByteSize:
		return "byte_sz"
	case reloFieldExists:
		return "field_exists"
	case reloFieldSigned:
		return "signed"
	case reloFieldLShiftU64:
		return "lshift_u64"
	case reloFieldRShiftU64:
		return "rshift_u64"
	case reloTypeIDLocal:
		return "local_type_id"
	case reloTypeIDTarget:
		return "target_type_id"
	case reloTypeExists:
		return "type_exists"
	case reloTypeSize:
		return "type_size"
	case reloEnumvalExists:
		return "enumval_exists"
	case reloEnumvalValue:
		return "enumval_value"
	default:
		return "unknown"
	}
}

func coreRelocate(local, target *Spec, coreRelos bpfCoreRelos) (map[uint64]Relocation, error) {
	if target == nil {
		var err error
		target, err = loadKernelSpec()
		if err != nil {
			return nil, err
		}
	}

	if local.byteOrder != target.byteOrder {
		return nil, fmt.Errorf("can't relocate %s against %s", local.byteOrder, target.byteOrder)
	}

	relocations := make(map[uint64]Relocation, len(coreRelos))
	for _, relo := range coreRelos {
		accessorStr, err := local.strings.Lookup(relo.AccessStrOff)
		if err != nil {
			return nil, err
		}

		accessor, err := parseCoreAccessor(accessorStr)
		if err != nil {
			return nil, fmt.Errorf("accessor %q: %s", accessorStr, err)
		}

		if int(relo.TypeID) >= len(local.types) {
			return nil, fmt.Errorf("invalid type id %d", relo.TypeID)
		}

		typ := local.types[relo.TypeID]

		if relo.ReloKind == reloTypeIDLocal {
			relocations[uint64(relo.InsnOff)] = Relocation{
				uint32(typ.ID()),
				uint32(typ.ID()),
			}
			continue
		}

		named, ok := typ.(namedType)
		if !ok || named.name() == "" {
			return nil, fmt.Errorf("relocate anonymous type %s: %w", typ.String(), ErrNotSupported)
		}

		name := essentialName(named.name())
		res, err := coreCalculateRelocation(typ, target.namedTypes[name], relo.ReloKind, accessor)
		if err != nil {
			return nil, fmt.Errorf("relocate %s: %w", name, err)
		}

		relocations[uint64(relo.InsnOff)] = res
	}

	return relocations, nil
}

var errAmbiguousRelocation = errors.New("ambiguous relocation")

func coreCalculateRelocation(local Type, targets []namedType, kind coreReloKind, localAccessor coreAccessor) (Relocation, error) {
	var relos []Relocation
	var matches []Type
	for _, target := range targets {
		switch kind {
		case reloTypeIDTarget:
			if localAccessor[0] != 0 {
				return Relocation{}, fmt.Errorf("%s: unexpected non-zero accessor", kind)
			}

			if compat, err := coreAreTypesCompatible(local, target); err != nil {
				return Relocation{}, fmt.Errorf("%s: %s", kind, err)
			} else if !compat {
				continue
			}

			relos = append(relos, Relocation{uint32(target.ID()), uint32(target.ID())})

		default:
			return Relocation{}, fmt.Errorf("relocation %s: %w", kind, ErrNotSupported)
		}
		matches = append(matches, target)
	}

	if len(relos) == 0 {
		// TODO: Add switch for existence checks like reloEnumvalExists here.

		// TODO: This might have to be poisoned.
		return Relocation{}, fmt.Errorf("no relocation found, tried %v", targets)
	}

	relo := relos[0]
	for _, altRelo := range relos[1:] {
		if !altRelo.equal(relo) {
			return Relocation{}, fmt.Errorf("multiple types %v match: %w", matches, errAmbiguousRelocation)
		}
	}

	return relo, nil
}

/* coreAccessor contains a path through a struct. It contains at least one index.
 *
 * The interpretation depends on the kind of the relocation. The following is
 * taken from struct bpf_core_relo in libbpf_internal.h:
 *
 * - for field-based relocations, string encodes an accessed field using
 *   a sequence of field and array indices, separated by colon (:). It's
 *   conceptually very close to LLVM's getelementptr ([0]) instruction's
 *   arguments for identifying offset to a field.
 * - for type-based relocations, strings is expected to be just "0";
 * - for enum value-based relocations, string contains an index of enum
 *   value within its enum type;
 *
 * Example to provide a better feel.
 *
 *   struct sample {
 *       int a;
 *       struct {
 *           int b[10];
 *       };
 *   };
 *
 *   struct sample s = ...;
 *   int x = &s->a;     // encoded as "0:0" (a is field #0)
 *   int y = &s->b[5];  // encoded as "0:1:0:5" (anon struct is field #1,
 *                      // b is field #0 inside anon struct, accessing elem #5)
 *   int z = &s[10]->b; // encoded as "10:1" (ptr is used as an array)
 */
type coreAccessor []int

func parseCoreAccessor(accessor string) (coreAccessor, error) {
	if accessor == "" {
		return nil, fmt.Errorf("empty accessor")
	}

	var result coreAccessor
	parts := strings.Split(accessor, ":")
	for _, part := range parts {
		// 31 bits to avoid overflowing int on 32 bit platforms.
		index, err := strconv.ParseUint(part, 10, 31)
		if err != nil {
			return nil, fmt.Errorf("accessor index %q: %s", part, err)
		}

		result = append(result, int(index))
	}

	return result, nil
}

/* The comment below is from bpf_core_types_are_compat in libbpf.c:
 *
 * Check local and target types for compatibility. This check is used for
 * type-based CO-RE relocations and follow slightly different rules than
 * field-based relocations. This function assumes that root types were already
 * checked for name match. Beyond that initial root-level name check, names
 * are completely ignored. Compatibility rules are as follows:
 *   - any two STRUCTs/UNIONs/FWDs/ENUMs/INTs are considered compatible, but
 *     kind should match for local and target types (i.e., STRUCT is not
 *     compatible with UNION);
 *   - for ENUMs, the size is ignored;
 *   - for INT, size and signedness are ignored;
 *   - for ARRAY, dimensionality is ignored, element types are checked for
 *     compatibility recursively;
 *   - CONST/VOLATILE/RESTRICT modifiers are ignored;
 *   - TYPEDEFs/PTRs are compatible if types they pointing to are compatible;
 *   - FUNC_PROTOs are compatible if they have compatible signature: same
 *     number of input args and compatible return and argument types.
 * These rules are not set in stone and probably will be adjusted as we get
 * more experience with using BPF CO-RE relocations.
 */
func coreAreTypesCompatible(localType Type, targetType Type) (bool, error) {
	var (
		localTs, targetTs typeDeque
		l, t              = &localType, &targetType
		depth             = 0
	)

	for ; l != nil && t != nil; l, t = localTs.shift(), targetTs.shift() {
		if depth >= maxTypeDepth {
			return false, errors.New("types are nested too deep")
		}

		localType = skipQualifierAndTypedef(*l)
		targetType = skipQualifierAndTypedef(*t)

		if reflect.TypeOf(localType) != reflect.TypeOf(targetType) {
			return false, nil
		}

		switch lv := (localType).(type) {
		case *Void, *Struct, *Union, *Enum, *Fwd:
			// Nothing to do here

		case *Int:
			tv := targetType.(*Int)
			if lv.isBitfield() || tv.isBitfield() {
				return false, nil
			}

		case *Pointer, *Array:
			depth++
			localType.walk(&localTs)
			targetType.walk(&targetTs)

		case *FuncProto:
			tv := targetType.(*FuncProto)
			if len(lv.Params) != len(tv.Params) {
				return false, nil
			}

			depth++
			localType.walk(&localTs)
			targetType.walk(&targetTs)

		default:
			return false, fmt.Errorf("unsupported type %T", localType)
		}
	}

	if l != nil {
		return false, fmt.Errorf("dangling local type %T", *l)
	}

	if t != nil {
		return false, fmt.Errorf("dangling target type %T", *t)
	}

	return true, nil
}

/* The comment below is from bpf_core_fields_are_compat in libbpf.c:
 *
 * Check two types for compatibility for the purpose of field access
 * relocation. const/volatile/restrict and typedefs are skipped to ensure we
 * are relocating semantically compatible entities:
 *   - any two STRUCTs/UNIONs are compatible and can be mixed;
 *   - any two FWDs are compatible, if their names match (modulo flavor suffix);
 *   - any two PTRs are always compatible;
 *   - for ENUMs, names should be the same (ignoring flavor suffix) or at
 *     least one of enums should be anonymous;
 *   - for ENUMs, check sizes, names are ignored;
 *   - for INT, size and signedness are ignored;
 *   - for ARRAY, dimensionality is ignored, element types are checked for
 *     compatibility recursively;
 *   - everything else shouldn't be ever a target of relocation.
 * These rules are not set in stone and probably will be adjusted as we get
 * more experience with using BPF CO-RE relocations.
 */
func coreAreMembersCompatible(localType Type, targetType Type) (bool, error) {
	doNamesMatch := func(a, b string) bool {
		if a == "" || b == "" {
			// allow anonymous and named type to match
			return true
		}

		return essentialName(a) == essentialName(b)
	}

	for depth := 0; depth <= maxTypeDepth; depth++ {
		localType = skipQualifierAndTypedef(localType)
		targetType = skipQualifierAndTypedef(targetType)

		_, lok := localType.(composite)
		_, tok := targetType.(composite)
		if lok && tok {
			return true, nil
		}

		if reflect.TypeOf(localType) != reflect.TypeOf(targetType) {
			return false, nil
		}

		switch lv := localType.(type) {
		case *Pointer:
			return true, nil

		case *Enum:
			tv := targetType.(*Enum)
			return doNamesMatch(lv.name(), tv.name()), nil

		case *Fwd:
			tv := targetType.(*Fwd)
			return doNamesMatch(lv.name(), tv.name()), nil

		case *Int:
			tv := targetType.(*Int)
			return !lv.isBitfield() && !tv.isBitfield(), nil

		case *Array:
			tv := targetType.(*Array)

			localType = lv.Type
			targetType = tv.Type

		default:
			return false, fmt.Errorf("unsupported type %T", localType)
		}
	}

	return false, errors.New("types are nested too deep")
}

func skipQualifierAndTypedef(typ Type) Type {
	result := typ
	for depth := 0; depth <= maxTypeDepth; depth++ {
		switch v := (result).(type) {
		case qualifier:
			result = v.qualify()
		case *Typedef:
			result = v.Type
		default:
			return result
		}
	}
	return typ
}
