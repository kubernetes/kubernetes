package btf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/cilium/ebpf/asm"
)

// Code in this file is derived from libbpf, which is available under a BSD
// 2-Clause license.

// A constant used when CO-RE relocation has to remove instructions.
//
// Taken from libbpf.
const COREBadRelocationSentinel = 0xbad2310

// COREFixup is the result of computing a CO-RE relocation for a target.
type COREFixup struct {
	kind   coreKind
	local  uint64
	target uint64
	// True if there is no valid fixup. The instruction is replaced with an
	// invalid dummy.
	poison bool
	// True if the validation of the local value should be skipped. Used by
	// some kinds of bitfield relocations.
	skipLocalValidation bool
}

func (f *COREFixup) equal(other COREFixup) bool {
	return f.local == other.local && f.target == other.target
}

func (f *COREFixup) String() string {
	if f.poison {
		return fmt.Sprintf("%s=poison", f.kind)
	}
	return fmt.Sprintf("%s=%d->%d", f.kind, f.local, f.target)
}

func (f *COREFixup) Apply(ins *asm.Instruction) error {
	if f.poison {
		// Relocation is poisoned, replace the instruction with an invalid one.
		if ins.OpCode.IsDWordLoad() {
			// Replace a dword load with a invalid dword load to preserve instruction size.
			*ins = asm.LoadImm(asm.R10, COREBadRelocationSentinel, asm.DWord)
		} else {
			// Replace all single size instruction with a invalid call instruction.
			*ins = asm.BuiltinFunc(COREBadRelocationSentinel).Call()
		}

		// Add context to the kernel verifier output.
		if source := ins.Source(); source != nil {
			*ins = ins.WithSource(asm.Comment(fmt.Sprintf("instruction poisoned by CO-RE: %s", source)))
		} else {
			*ins = ins.WithSource(asm.Comment("instruction poisoned by CO-RE"))
		}

		return nil
	}

	switch class := ins.OpCode.Class(); class {
	case asm.LdXClass, asm.StClass, asm.StXClass:
		if want := int16(f.local); !f.skipLocalValidation && want != ins.Offset {
			return fmt.Errorf("invalid offset %d, expected %d", ins.Offset, f.local)
		}

		if f.target > math.MaxInt16 {
			return fmt.Errorf("offset %d exceeds MaxInt16", f.target)
		}

		ins.Offset = int16(f.target)

	case asm.LdClass:
		if !ins.IsConstantLoad(asm.DWord) {
			return fmt.Errorf("not a dword-sized immediate load")
		}

		if want := int64(f.local); !f.skipLocalValidation && want != ins.Constant {
			return fmt.Errorf("invalid immediate %d, expected %d (fixup: %v)", ins.Constant, want, f)
		}

		ins.Constant = int64(f.target)

	case asm.ALUClass:
		if ins.OpCode.ALUOp() == asm.Swap {
			return fmt.Errorf("relocation against swap")
		}

		fallthrough

	case asm.ALU64Class:
		if src := ins.OpCode.Source(); src != asm.ImmSource {
			return fmt.Errorf("invalid source %s", src)
		}

		if want := int64(f.local); !f.skipLocalValidation && want != ins.Constant {
			return fmt.Errorf("invalid immediate %d, expected %d (fixup: %v, kind: %v, ins: %v)", ins.Constant, want, f, f.kind, ins)
		}

		if f.target > math.MaxInt32 {
			return fmt.Errorf("immediate %d exceeds MaxInt32", f.target)
		}

		ins.Constant = int64(f.target)

	default:
		return fmt.Errorf("invalid class %s", class)
	}

	return nil
}

func (f COREFixup) isNonExistant() bool {
	return f.kind.checksForExistence() && f.target == 0
}

// coreKind is the type of CO-RE relocation as specified in BPF source code.
type coreKind uint32

const (
	reloFieldByteOffset coreKind = iota /* field byte offset */
	reloFieldByteSize                   /* field size in bytes */
	reloFieldExists                     /* field existence in target kernel */
	reloFieldSigned                     /* field signedness (0 - unsigned, 1 - signed) */
	reloFieldLShiftU64                  /* bitfield-specific left bitshift */
	reloFieldRShiftU64                  /* bitfield-specific right bitshift */
	reloTypeIDLocal                     /* type ID in local BPF object */
	reloTypeIDTarget                    /* type ID in target kernel */
	reloTypeExists                      /* type existence in target kernel */
	reloTypeSize                        /* type size in bytes */
	reloEnumvalExists                   /* enum value existence in target kernel */
	reloEnumvalValue                    /* enum value integer value */
	reloTypeMatches                     /* type matches kernel type */
)

func (k coreKind) checksForExistence() bool {
	return k == reloEnumvalExists || k == reloTypeExists || k == reloFieldExists || k == reloTypeMatches
}

func (k coreKind) String() string {
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
	case reloTypeMatches:
		return "type_matches"
	default:
		return fmt.Sprintf("unknown (%d)", k)
	}
}

// CORERelocate calculates changes needed to adjust eBPF instructions for differences
// in types.
//
// targets forms the set of types to relocate against. The first element has to be
// BTF for vmlinux, the following must be types for kernel modules.
//
// resolveLocalTypeID is called for each local type which requires a stable TypeID.
// Calling the function with the same type multiple times must produce the same
// result. It is the callers responsibility to ensure that the relocated instructions
// are loaded with matching BTF.
//
// Returns a list of fixups which can be applied to instructions to make them
// match the target type(s).
//
// Fixups are returned in the order of relos, e.g. fixup[i] is the solution
// for relos[i].
func CORERelocate(relos []*CORERelocation, targets []*Spec, bo binary.ByteOrder, resolveLocalTypeID func(Type) (TypeID, error)) ([]COREFixup, error) {
	if len(targets) == 0 {
		// Explicitly check for nil here since the argument used to be optional.
		return nil, fmt.Errorf("targets must be provided")
	}

	// We can't encode type IDs that aren't for vmlinux into instructions at the
	// moment.
	resolveTargetTypeID := targets[0].TypeID

	for _, target := range targets {
		if bo != target.imm.byteOrder {
			return nil, fmt.Errorf("can't relocate %s against %s", bo, target.imm.byteOrder)
		}
	}

	type reloGroup struct {
		relos []*CORERelocation
		// Position of each relocation in relos.
		indices []int
	}

	// Split relocations into per Type lists.
	relosByType := make(map[Type]*reloGroup)
	result := make([]COREFixup, len(relos))
	for i, relo := range relos {
		if relo.kind == reloTypeIDLocal {
			// Filtering out reloTypeIDLocal here makes our lives a lot easier
			// down the line, since it doesn't have a target at all.
			if len(relo.accessor) > 1 || relo.accessor[0] != 0 {
				return nil, fmt.Errorf("%s: unexpected accessor %v", relo.kind, relo.accessor)
			}

			id, err := resolveLocalTypeID(relo.typ)
			if err != nil {
				return nil, fmt.Errorf("%s: get type id: %w", relo.kind, err)
			}

			result[i] = COREFixup{
				kind:   relo.kind,
				local:  uint64(relo.id),
				target: uint64(id),
			}
			continue
		}

		group, ok := relosByType[relo.typ]
		if !ok {
			group = &reloGroup{}
			relosByType[relo.typ] = group
		}
		group.relos = append(group.relos, relo)
		group.indices = append(group.indices, i)
	}

	for localType, group := range relosByType {
		localTypeName := localType.TypeName()
		if localTypeName == "" {
			return nil, fmt.Errorf("relocate unnamed or anonymous type %s: %w", localType, ErrNotSupported)
		}

		essentialName := newEssentialName(localTypeName)

		var targetTypes []Type
		for _, target := range targets {
			namedTypeIDs := target.imm.namedTypes[essentialName]
			targetTypes = slices.Grow(targetTypes, len(namedTypeIDs))
			for _, id := range namedTypeIDs {
				typ, err := target.TypeByID(id)
				if err != nil {
					return nil, err
				}

				targetTypes = append(targetTypes, typ)
			}
		}

		fixups, err := coreCalculateFixups(group.relos, targetTypes, bo, resolveTargetTypeID)
		if err != nil {
			return nil, fmt.Errorf("relocate %s: %w", localType, err)
		}

		for j, index := range group.indices {
			result[index] = fixups[j]
		}
	}

	return result, nil
}

var errAmbiguousRelocation = errors.New("ambiguous relocation")
var errImpossibleRelocation = errors.New("impossible relocation")
var errIncompatibleTypes = errors.New("incompatible types")

// coreCalculateFixups finds the target type that best matches all relocations.
//
// All relos must target the same type.
//
// The best target is determined by scoring: the less poisoning we have to do
// the better the target is.
func coreCalculateFixups(relos []*CORERelocation, targets []Type, bo binary.ByteOrder, resolveTargetTypeID func(Type) (TypeID, error)) ([]COREFixup, error) {
	bestScore := len(relos)
	var bestFixups []COREFixup
	for _, target := range targets {
		score := 0 // lower is better
		fixups := make([]COREFixup, 0, len(relos))
		for _, relo := range relos {
			fixup, err := coreCalculateFixup(relo, target, bo, resolveTargetTypeID)
			if err != nil {
				return nil, fmt.Errorf("target %s: %s: %w", target, relo.kind, err)
			}
			if fixup.poison || fixup.isNonExistant() {
				score++
			}
			fixups = append(fixups, fixup)
		}

		if score > bestScore {
			// We have a better target already, ignore this one.
			continue
		}

		if score < bestScore {
			// This is the best target yet, use it.
			bestScore = score
			bestFixups = fixups
			continue
		}

		// Some other target has the same score as the current one. Make sure
		// the fixups agree with each other.
		for i, fixup := range bestFixups {
			if !fixup.equal(fixups[i]) {
				return nil, fmt.Errorf("%s: multiple types match: %w", fixup.kind, errAmbiguousRelocation)
			}
		}
	}

	if bestFixups == nil {
		// Nothing at all matched, probably because there are no suitable
		// targets at all.
		//
		// Poison everything except checksForExistence.
		bestFixups = make([]COREFixup, len(relos))
		for i, relo := range relos {
			if relo.kind.checksForExistence() {
				bestFixups[i] = COREFixup{kind: relo.kind, local: 1, target: 0}
			} else {
				bestFixups[i] = COREFixup{kind: relo.kind, poison: true}
			}
		}
	}

	return bestFixups, nil
}

var errNoSignedness = errors.New("no signedness")

// coreCalculateFixup calculates the fixup given a relocation and a target type.
func coreCalculateFixup(relo *CORERelocation, target Type, bo binary.ByteOrder, resolveTargetTypeID func(Type) (TypeID, error)) (COREFixup, error) {
	fixup := func(local, target uint64) (COREFixup, error) {
		return COREFixup{kind: relo.kind, local: local, target: target}, nil
	}
	fixupWithoutValidation := func(local, target uint64) (COREFixup, error) {
		return COREFixup{kind: relo.kind, local: local, target: target, skipLocalValidation: true}, nil
	}
	poison := func() (COREFixup, error) {
		if relo.kind.checksForExistence() {
			return fixup(1, 0)
		}
		return COREFixup{kind: relo.kind, poison: true}, nil
	}
	zero := COREFixup{}

	local := relo.typ

	switch relo.kind {
	case reloTypeMatches:
		if len(relo.accessor) > 1 || relo.accessor[0] != 0 {
			return zero, fmt.Errorf("unexpected accessor %v", relo.accessor)
		}

		err := coreTypesMatch(local, target, nil)
		if errors.Is(err, errIncompatibleTypes) {
			return poison()
		}
		if err != nil {
			return zero, err
		}

		return fixup(1, 1)

	case reloTypeIDTarget, reloTypeSize, reloTypeExists:
		if len(relo.accessor) > 1 || relo.accessor[0] != 0 {
			return zero, fmt.Errorf("unexpected accessor %v", relo.accessor)
		}

		err := CheckTypeCompatibility(local, target)
		if errors.Is(err, errIncompatibleTypes) {
			return poison()
		}
		if err != nil {
			return zero, err
		}

		switch relo.kind {
		case reloTypeExists:
			return fixup(1, 1)

		case reloTypeIDTarget:
			targetID, err := resolveTargetTypeID(target)
			if errors.Is(err, ErrNotFound) {
				// Probably a relocation trying to get the ID
				// of a type from a kmod.
				return poison()
			}
			if err != nil {
				return zero, err
			}
			return fixup(uint64(relo.id), uint64(targetID))

		case reloTypeSize:
			localSize, err := Sizeof(local)
			if err != nil {
				return zero, err
			}

			targetSize, err := Sizeof(target)
			if err != nil {
				return zero, err
			}

			return fixup(uint64(localSize), uint64(targetSize))
		}

	case reloEnumvalValue, reloEnumvalExists:
		localValue, targetValue, err := coreFindEnumValue(local, relo.accessor, target)
		if errors.Is(err, errImpossibleRelocation) {
			return poison()
		}
		if err != nil {
			return zero, err
		}

		switch relo.kind {
		case reloEnumvalExists:
			return fixup(1, 1)

		case reloEnumvalValue:
			return fixup(localValue.Value, targetValue.Value)
		}

	case reloFieldByteOffset, reloFieldByteSize, reloFieldExists, reloFieldLShiftU64, reloFieldRShiftU64, reloFieldSigned:
		if _, ok := As[*Fwd](target); ok {
			// We can't relocate fields using a forward declaration, so
			// skip it. If a non-forward declaration is present in the BTF
			// we'll find it in one of the other iterations.
			return poison()
		}

		localField, targetField, err := coreFindField(local, relo.accessor, target)
		if errors.Is(err, errImpossibleRelocation) {
			return poison()
		}
		if err != nil {
			return zero, err
		}

		maybeSkipValidation := func(f COREFixup, err error) (COREFixup, error) {
			f.skipLocalValidation = localField.bitfieldSize > 0
			return f, err
		}

		switch relo.kind {
		case reloFieldExists:
			return fixup(1, 1)

		case reloFieldByteOffset:
			return maybeSkipValidation(fixup(uint64(localField.offset), uint64(targetField.offset)))

		case reloFieldByteSize:
			localSize, err := Sizeof(localField.Type)
			if err != nil {
				return zero, err
			}

			targetSize, err := Sizeof(targetField.Type)
			if err != nil {
				return zero, err
			}
			return maybeSkipValidation(fixup(uint64(localSize), uint64(targetSize)))

		case reloFieldLShiftU64:
			var target uint64
			if bo == binary.LittleEndian {
				targetSize, err := targetField.sizeBits()
				if err != nil {
					return zero, err
				}

				target = uint64(64 - targetField.bitfieldOffset - targetSize)
			} else {
				loadWidth, err := Sizeof(targetField.Type)
				if err != nil {
					return zero, err
				}

				target = uint64(64 - Bits(loadWidth*8) + targetField.bitfieldOffset)
			}
			return fixupWithoutValidation(0, target)

		case reloFieldRShiftU64:
			targetSize, err := targetField.sizeBits()
			if err != nil {
				return zero, err
			}

			return fixupWithoutValidation(0, uint64(64-targetSize))

		case reloFieldSigned:
			switch local := UnderlyingType(localField.Type).(type) {
			case *Enum:
				target, ok := As[*Enum](targetField.Type)
				if !ok {
					return zero, fmt.Errorf("target isn't *Enum but %T", targetField.Type)
				}

				return fixup(boolToUint64(local.Signed), boolToUint64(target.Signed))
			case *Int:
				target, ok := As[*Int](targetField.Type)
				if !ok {
					return zero, fmt.Errorf("target isn't *Int but %T", targetField.Type)
				}

				return fixup(
					uint64(local.Encoding&Signed),
					uint64(target.Encoding&Signed),
				)
			default:
				return zero, fmt.Errorf("type %T: %w", local, errNoSignedness)
			}
		}
	}

	return zero, ErrNotSupported
}

func boolToUint64(val bool) uint64 {
	if val {
		return 1
	}
	return 0
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

func parseCOREAccessor(accessor string) (coreAccessor, error) {
	if accessor == "" {
		return nil, fmt.Errorf("empty accessor")
	}

	parts := strings.Split(accessor, ":")
	result := make(coreAccessor, 0, len(parts))
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

func (ca coreAccessor) String() string {
	strs := make([]string, 0, len(ca))
	for _, i := range ca {
		strs = append(strs, strconv.Itoa(i))
	}
	return strings.Join(strs, ":")
}

func (ca coreAccessor) enumValue(t Type) (*EnumValue, error) {
	e, ok := As[*Enum](t)
	if !ok {
		return nil, fmt.Errorf("not an enum: %s", t)
	}

	if len(ca) > 1 {
		return nil, fmt.Errorf("invalid accessor %s for enum", ca)
	}

	i := ca[0]
	if i >= len(e.Values) {
		return nil, fmt.Errorf("invalid index %d for %s", i, e)
	}

	return &e.Values[i], nil
}

// coreField represents the position of a "child" of a composite type from the
// start of that type.
//
//	/- start of composite
//	| offset * 8 | bitfieldOffset | bitfieldSize | ... |
//	             \- start of field       end of field -/
type coreField struct {
	Type Type

	// The position of the field from the start of the composite type in bytes.
	offset uint32

	// The offset of the bitfield in bits from the start of the field.
	bitfieldOffset Bits

	// The size of the bitfield in bits.
	//
	// Zero if the field is not a bitfield.
	bitfieldSize Bits
}

func (cf *coreField) adjustOffsetToNthElement(n int) error {
	if n == 0 {
		return nil
	}

	size, err := Sizeof(cf.Type)
	if err != nil {
		return err
	}

	cf.offset += uint32(n) * uint32(size)
	return nil
}

func (cf *coreField) adjustOffsetBits(offset Bits) error {
	align, err := alignof(cf.Type)
	if err != nil {
		return err
	}

	// We can compute the load offset by:
	// 1) converting the bit offset to bytes with a flooring division.
	// 2) dividing and multiplying that offset by the alignment, yielding the
	//    load size aligned offset.
	offsetBytes := uint32(offset/8) / uint32(align) * uint32(align)

	// The number of bits remaining is the bit offset less the number of bits
	// we can "skip" with the aligned offset.
	cf.bitfieldOffset = offset - Bits(offsetBytes*8)

	// We know that cf.offset is aligned at to at least align since we get it
	// from the compiler via BTF. Adding an aligned offsetBytes preserves the
	// alignment.
	cf.offset += offsetBytes
	return nil
}

func (cf *coreField) sizeBits() (Bits, error) {
	if cf.bitfieldSize > 0 {
		return cf.bitfieldSize, nil
	}

	// Someone is trying to access a non-bitfield via a bit shift relocation.
	// This happens when a field changes from a bitfield to a regular field
	// between kernel versions. Synthesise the size to make the shifts work.
	size, err := Sizeof(cf.Type)
	if err != nil {
		return 0, err
	}
	return Bits(size * 8), nil
}

// coreFindField descends into the local type using the accessor and tries to
// find an equivalent field in target at each step.
//
// Returns the field and the offset of the field from the start of
// target in bits.
func coreFindField(localT Type, localAcc coreAccessor, targetT Type) (coreField, coreField, error) {
	local := coreField{Type: localT}
	target := coreField{Type: targetT}

	if err := coreAreMembersCompatible(local.Type, target.Type); err != nil {
		return coreField{}, coreField{}, fmt.Errorf("fields: %w", err)
	}

	// The first index is used to offset a pointer of the base type like
	// when accessing an array.
	if err := local.adjustOffsetToNthElement(localAcc[0]); err != nil {
		return coreField{}, coreField{}, err
	}

	if err := target.adjustOffsetToNthElement(localAcc[0]); err != nil {
		return coreField{}, coreField{}, err
	}

	var localMaybeFlex, targetMaybeFlex bool
	for i, acc := range localAcc[1:] {
		switch localType := UnderlyingType(local.Type).(type) {
		case composite:
			// For composite types acc is used to find the field in the local type,
			// and then we try to find a field in target with the same name.
			localMembers := localType.members()
			if acc >= len(localMembers) {
				return coreField{}, coreField{}, fmt.Errorf("invalid accessor %d for %s", acc, localType)
			}

			localMember := localMembers[acc]
			if localMember.Name == "" {
				localMemberType, ok := As[composite](localMember.Type)
				if !ok {
					return coreField{}, coreField{}, fmt.Errorf("unnamed field with type %s: %s", localMember.Type, ErrNotSupported)
				}

				// This is an anonymous struct or union, ignore it.
				local = coreField{
					Type:   localMemberType,
					offset: local.offset + localMember.Offset.Bytes(),
				}
				localMaybeFlex = false
				continue
			}

			targetType, ok := As[composite](target.Type)
			if !ok {
				return coreField{}, coreField{}, fmt.Errorf("target not composite: %w", errImpossibleRelocation)
			}

			targetMember, last, err := coreFindMember(targetType, localMember.Name)
			if err != nil {
				return coreField{}, coreField{}, err
			}

			local = coreField{
				Type:         localMember.Type,
				offset:       local.offset,
				bitfieldSize: localMember.BitfieldSize,
			}
			localMaybeFlex = acc == len(localMembers)-1

			target = coreField{
				Type:         targetMember.Type,
				offset:       target.offset,
				bitfieldSize: targetMember.BitfieldSize,
			}
			targetMaybeFlex = last

			if local.bitfieldSize == 0 && target.bitfieldSize == 0 {
				local.offset += localMember.Offset.Bytes()
				target.offset += targetMember.Offset.Bytes()
				break
			}

			// Either of the members is a bitfield. Make sure we're at the
			// end of the accessor.
			if next := i + 1; next < len(localAcc[1:]) {
				return coreField{}, coreField{}, fmt.Errorf("can't descend into bitfield")
			}

			if err := local.adjustOffsetBits(localMember.Offset); err != nil {
				return coreField{}, coreField{}, err
			}

			if err := target.adjustOffsetBits(targetMember.Offset); err != nil {
				return coreField{}, coreField{}, err
			}

		case *Array:
			// For arrays, acc is the index in the target.
			targetType, ok := As[*Array](target.Type)
			if !ok {
				return coreField{}, coreField{}, fmt.Errorf("target not array: %w", errImpossibleRelocation)
			}

			if localType.Nelems == 0 && !localMaybeFlex {
				return coreField{}, coreField{}, fmt.Errorf("local type has invalid flexible array")
			}
			if targetType.Nelems == 0 && !targetMaybeFlex {
				return coreField{}, coreField{}, fmt.Errorf("target type has invalid flexible array")
			}

			if localType.Nelems > 0 && acc >= int(localType.Nelems) {
				return coreField{}, coreField{}, fmt.Errorf("invalid access of %s at index %d", localType, acc)
			}
			if targetType.Nelems > 0 && acc >= int(targetType.Nelems) {
				return coreField{}, coreField{}, fmt.Errorf("out of bounds access of target: %w", errImpossibleRelocation)
			}

			local = coreField{
				Type:   localType.Type,
				offset: local.offset,
			}
			localMaybeFlex = false

			if err := local.adjustOffsetToNthElement(acc); err != nil {
				return coreField{}, coreField{}, err
			}

			target = coreField{
				Type:   targetType.Type,
				offset: target.offset,
			}
			targetMaybeFlex = false

			if err := target.adjustOffsetToNthElement(acc); err != nil {
				return coreField{}, coreField{}, err
			}

		default:
			return coreField{}, coreField{}, fmt.Errorf("relocate field of %T: %w", localType, ErrNotSupported)
		}

		if err := coreAreMembersCompatible(local.Type, target.Type); err != nil {
			return coreField{}, coreField{}, err
		}
	}

	return local, target, nil
}

// coreFindMember finds a member in a composite type while handling anonymous
// structs and unions.
func coreFindMember(typ composite, name string) (Member, bool, error) {
	if name == "" {
		return Member{}, false, errors.New("can't search for anonymous member")
	}

	type offsetTarget struct {
		composite
		offset Bits
	}

	targets := []offsetTarget{{typ, 0}}
	visited := make(map[composite]bool)

	for i := 0; i < len(targets); i++ {
		target := targets[i]

		// Only visit targets once to prevent infinite recursion.
		if visited[target] {
			continue
		}
		if len(visited) >= maxResolveDepth {
			// This check is different than libbpf, which restricts the entire
			// path to BPF_CORE_SPEC_MAX_LEN items.
			return Member{}, false, fmt.Errorf("type is nested too deep")
		}
		visited[target] = true

		members := target.members()
		for j, member := range members {
			if member.Name == name {
				// NB: This is safe because member is a copy.
				member.Offset += target.offset
				return member, j == len(members)-1, nil
			}

			// The names don't match, but this member could be an anonymous struct
			// or union.
			if member.Name != "" {
				continue
			}

			comp, ok := As[composite](member.Type)
			if !ok {
				return Member{}, false, fmt.Errorf("anonymous non-composite type %T not allowed", member.Type)
			}

			targets = append(targets, offsetTarget{comp, target.offset + member.Offset})
		}
	}

	return Member{}, false, fmt.Errorf("no matching member: %w", errImpossibleRelocation)
}

// coreFindEnumValue follows localAcc to find the equivalent enum value in target.
func coreFindEnumValue(local Type, localAcc coreAccessor, target Type) (localValue, targetValue *EnumValue, _ error) {
	localValue, err := localAcc.enumValue(local)
	if err != nil {
		return nil, nil, err
	}

	targetEnum, ok := As[*Enum](target)
	if !ok {
		return nil, nil, errImpossibleRelocation
	}

	localName := newEssentialName(localValue.Name)
	for i, targetValue := range targetEnum.Values {
		if newEssentialName(targetValue.Name) != localName {
			continue
		}

		return localValue, &targetEnum.Values[i], nil
	}

	return nil, nil, errImpossibleRelocation
}

// CheckTypeCompatibility checks local and target types for Compatibility according to CO-RE rules.
//
// Only layout compatibility is checked, ignoring names of the root type.
func CheckTypeCompatibility(localType Type, targetType Type) error {
	return coreAreTypesCompatible(localType, targetType, nil)
}

type pair struct {
	A, B Type
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
 *
 * Returns errIncompatibleTypes if types are not compatible.
 */
func coreAreTypesCompatible(localType Type, targetType Type, visited map[pair]struct{}) error {
	localType = UnderlyingType(localType)
	targetType = UnderlyingType(targetType)

	if reflect.TypeOf(localType) != reflect.TypeOf(targetType) {
		return fmt.Errorf("type mismatch between %v and %v: %w", localType, targetType, errIncompatibleTypes)
	}

	if _, ok := visited[pair{localType, targetType}]; ok {
		return nil
	}
	if visited == nil {
		visited = make(map[pair]struct{})
	}
	visited[pair{localType, targetType}] = struct{}{}

	switch lv := localType.(type) {
	case *Void, *Struct, *Union, *Enum, *Fwd, *Int:
		return nil

	case *Pointer:
		tv := targetType.(*Pointer)
		return coreAreTypesCompatible(lv.Target, tv.Target, visited)

	case *Array:
		tv := targetType.(*Array)
		if err := coreAreTypesCompatible(lv.Index, tv.Index, visited); err != nil {
			return err
		}

		return coreAreTypesCompatible(lv.Type, tv.Type, visited)

	case *FuncProto:
		tv := targetType.(*FuncProto)
		if err := coreAreTypesCompatible(lv.Return, tv.Return, visited); err != nil {
			return err
		}

		if len(lv.Params) != len(tv.Params) {
			return fmt.Errorf("function param mismatch: %w", errIncompatibleTypes)
		}

		for i, localParam := range lv.Params {
			targetParam := tv.Params[i]
			if err := coreAreTypesCompatible(localParam.Type, targetParam.Type, visited); err != nil {
				return err
			}
		}

		return nil

	default:
		return fmt.Errorf("unsupported type %T", localType)
	}
}

/* coreAreMembersCompatible checks two types for field-based relocation compatibility.
 *
 * The comment below is from bpf_core_fields_are_compat in libbpf.c:
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
 *   - any two FLOATs are always compatible;
 *   - for ARRAY, dimensionality is ignored, element types are checked for
 *     compatibility recursively;
 *     [ NB: coreAreMembersCompatible doesn't recurse, this check is done
 *       by coreFindField. ]
 *   - everything else shouldn't be ever a target of relocation.
 * These rules are not set in stone and probably will be adjusted as we get
 * more experience with using BPF CO-RE relocations.
 *
 * Returns errImpossibleRelocation if the members are not compatible.
 */
func coreAreMembersCompatible(localType Type, targetType Type) error {
	localType = UnderlyingType(localType)
	targetType = UnderlyingType(targetType)

	_, lok := localType.(composite)
	_, tok := targetType.(composite)
	if lok && tok {
		return nil
	}

	if reflect.TypeOf(localType) != reflect.TypeOf(targetType) {
		return fmt.Errorf("type mismatch: %w", errImpossibleRelocation)
	}

	switch lv := localType.(type) {
	case *Array, *Pointer, *Float, *Int:
		return nil

	case *Enum:
		tv := targetType.(*Enum)
		if !coreEssentialNamesMatch(lv.Name, tv.Name) {
			return fmt.Errorf("names %q and %q don't match: %w", lv.Name, tv.Name, errImpossibleRelocation)
		}

		return nil

	case *Fwd:
		tv := targetType.(*Fwd)
		if !coreEssentialNamesMatch(lv.Name, tv.Name) {
			return fmt.Errorf("names %q and %q don't match: %w", lv.Name, tv.Name, errImpossibleRelocation)
		}

		return nil

	default:
		return fmt.Errorf("type %s: %w", localType, ErrNotSupported)
	}
}

// coreEssentialNamesMatch compares two names while ignoring their flavour suffix.
//
// This should only be used on names which are in the global scope, like struct
// names, typedefs or enum values.
func coreEssentialNamesMatch(a, b string) bool {
	if a == "" || b == "" {
		// allow anonymous and named type to match
		return true
	}

	return newEssentialName(a) == newEssentialName(b)
}

/* The comment below is from __bpf_core_types_match in relo_core.c:
 *
 * Check that two types "match". This function assumes that root types were
 * already checked for name match.
 *
 * The matching relation is defined as follows:
 * - modifiers and typedefs are stripped (and, hence, effectively ignored)
 * - generally speaking types need to be of same kind (struct vs. struct, union
 *   vs. union, etc.)
 *   - exceptions are struct/union behind a pointer which could also match a
 *     forward declaration of a struct or union, respectively, and enum vs.
 *     enum64 (see below)
 * Then, depending on type:
 * - integers:
 *   - match if size and signedness match
 * - arrays & pointers:
 *   - target types are recursively matched
 * - structs & unions:
 *   - local members need to exist in target with the same name
 *   - for each member we recursively check match unless it is already behind a
 *     pointer, in which case we only check matching names and compatible kind
 * - enums:
 *   - local variants have to have a match in target by symbolic name (but not
 *     numeric value)
 *   - size has to match (but enum may match enum64 and vice versa)
 * - function pointers:
 *   - number and position of arguments in local type has to match target
 *   - for each argument and the return value we recursively check match
 */
func coreTypesMatch(localType Type, targetType Type, visited map[pair]struct{}) error {
	localType = UnderlyingType(localType)
	targetType = UnderlyingType(targetType)

	if !coreEssentialNamesMatch(localType.TypeName(), targetType.TypeName()) {
		return fmt.Errorf("type name %q don't match %q: %w", localType.TypeName(), targetType.TypeName(), errIncompatibleTypes)
	}

	if reflect.TypeOf(localType) != reflect.TypeOf(targetType) {
		return fmt.Errorf("type mismatch between %v and %v: %w", localType, targetType, errIncompatibleTypes)
	}

	if _, ok := visited[pair{localType, targetType}]; ok {
		return nil
	}
	if visited == nil {
		visited = make(map[pair]struct{})
	}
	visited[pair{localType, targetType}] = struct{}{}

	switch lv := (localType).(type) {
	case *Void:

	case *Fwd:
		if targetType.(*Fwd).Kind != lv.Kind {
			return fmt.Errorf("fwd kind mismatch between %v and %v: %w", localType, targetType, errIncompatibleTypes)
		}

	case *Enum:
		return coreEnumsMatch(lv, targetType.(*Enum))

	case composite:
		tv := targetType.(composite)

		if len(lv.members()) > len(tv.members()) {
			return errIncompatibleTypes
		}

		localMembers := lv.members()
		targetMembers := map[string]Member{}
		for _, member := range tv.members() {
			targetMembers[member.Name] = member
		}

		for _, localMember := range localMembers {
			targetMember, found := targetMembers[localMember.Name]
			if !found {
				return fmt.Errorf("no field %q in %v: %w", localMember.Name, targetType, errIncompatibleTypes)
			}

			err := coreTypesMatch(localMember.Type, targetMember.Type, visited)
			if err != nil {
				return err
			}
		}

	case *Int:
		if !coreEncodingMatches(lv, targetType.(*Int)) {
			return fmt.Errorf("int mismatch between %v and %v: %w", localType, targetType, errIncompatibleTypes)
		}

	case *Pointer:
		tv := targetType.(*Pointer)

		// Allow a pointer to a forward declaration to match a struct
		// or union.
		if fwd, ok := As[*Fwd](lv.Target); ok && fwd.matches(tv.Target) {
			return nil
		}

		if fwd, ok := As[*Fwd](tv.Target); ok && fwd.matches(lv.Target) {
			return nil
		}

		return coreTypesMatch(lv.Target, tv.Target, visited)

	case *Array:
		tv := targetType.(*Array)

		if lv.Nelems != tv.Nelems {
			return fmt.Errorf("array mismatch between %v and %v: %w", localType, targetType, errIncompatibleTypes)
		}

		return coreTypesMatch(lv.Type, tv.Type, visited)

	case *FuncProto:
		tv := targetType.(*FuncProto)

		if len(lv.Params) != len(tv.Params) {
			return fmt.Errorf("function param mismatch: %w", errIncompatibleTypes)
		}

		for i, lparam := range lv.Params {
			if err := coreTypesMatch(lparam.Type, tv.Params[i].Type, visited); err != nil {
				return err
			}
		}

		return coreTypesMatch(lv.Return, tv.Return, visited)

	default:
		return fmt.Errorf("unsupported type %T", localType)
	}

	return nil
}

// coreEncodingMatches returns true if both ints have the same size and signedness.
// All encodings other than `Signed` are considered unsigned.
func coreEncodingMatches(local, target *Int) bool {
	return local.Size == target.Size && (local.Encoding == Signed) == (target.Encoding == Signed)
}

// coreEnumsMatch checks two enums match, which is considered to be the case if the following is true:
// - size has to match (but enum may match enum64 and vice versa)
// - local variants have to have a match in target by symbolic name (but not numeric value)
func coreEnumsMatch(local *Enum, target *Enum) error {
	if local.Size != target.Size {
		return fmt.Errorf("size mismatch between %v and %v: %w", local, target, errIncompatibleTypes)
	}

	// If there are more values in the local than the target, there must be at least one value in the local
	// that isn't in the target, and therefor the types are incompatible.
	if len(local.Values) > len(target.Values) {
		return fmt.Errorf("local has more values than target: %w", errIncompatibleTypes)
	}

outer:
	for _, lv := range local.Values {
		for _, rv := range target.Values {
			if coreEssentialNamesMatch(lv.Name, rv.Name) {
				continue outer
			}
		}

		return fmt.Errorf("no match for %v in %v: %w", lv, target, errIncompatibleTypes)
	}

	return nil
}
