package btf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/cilium/ebpf/asm"
)

// Code in this file is derived from libbpf, which is available under a BSD
// 2-Clause license.

// COREFixup is the result of computing a CO-RE relocation for a target.
type COREFixup struct {
	kind   coreKind
	local  uint32
	target uint32
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
		const badRelo = 0xbad2310

		*ins = asm.BuiltinFunc(badRelo).Call()
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
)

func (k coreKind) checksForExistence() bool {
	return k == reloEnumvalExists || k == reloTypeExists || k == reloFieldExists
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
	default:
		return "unknown"
	}
}

// CORERelocate calculates changes needed to adjust eBPF instructions for differences
// in types.
//
// Returns a list of fixups which can be applied to instructions to make them
// match the target type(s).
//
// Fixups are returned in the order of relos, e.g. fixup[i] is the solution
// for relos[i].
func CORERelocate(relos []*CORERelocation, target *Spec, bo binary.ByteOrder) ([]COREFixup, error) {
	if bo != target.byteOrder {
		return nil, fmt.Errorf("can't relocate %s against %s", bo, target.byteOrder)
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

			result[i] = COREFixup{
				kind:  relo.kind,
				local: uint32(relo.id),
				// NB: Using relo.id as the target here is incorrect, since
				// it doesn't match the BTF we generate on the fly. This isn't
				// too bad for now since there are no uses of the local type ID
				// in the kernel, yet.
				target: uint32(relo.id),
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

		targets := target.namedTypes[newEssentialName(localTypeName)]
		fixups, err := coreCalculateFixups(group.relos, target, targets, bo)
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

// coreCalculateFixups finds the target type that best matches all relocations.
//
// All relos must target the same type.
//
// The best target is determined by scoring: the less poisoning we have to do
// the better the target is.
func coreCalculateFixups(relos []*CORERelocation, targetSpec *Spec, targets []Type, bo binary.ByteOrder) ([]COREFixup, error) {
	bestScore := len(relos)
	var bestFixups []COREFixup
	for i := range targets {
		targetID, err := targetSpec.TypeID(targets[i])
		if err != nil {
			return nil, fmt.Errorf("target type ID: %w", err)
		}
		target := Copy(targets[i], UnderlyingType)

		score := 0 // lower is better
		fixups := make([]COREFixup, 0, len(relos))
		for _, relo := range relos {
			fixup, err := coreCalculateFixup(relo, target, targetID, bo)
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

// coreCalculateFixup calculates the fixup for a single local type, target type
// and relocation.
func coreCalculateFixup(relo *CORERelocation, target Type, targetID TypeID, bo binary.ByteOrder) (COREFixup, error) {
	fixup := func(local, target uint32) (COREFixup, error) {
		return COREFixup{kind: relo.kind, local: local, target: target}, nil
	}
	fixupWithoutValidation := func(local, target uint32) (COREFixup, error) {
		return COREFixup{kind: relo.kind, local: local, target: target, skipLocalValidation: true}, nil
	}
	poison := func() (COREFixup, error) {
		if relo.kind.checksForExistence() {
			return fixup(1, 0)
		}
		return COREFixup{kind: relo.kind, poison: true}, nil
	}
	zero := COREFixup{}

	local := Copy(relo.typ, UnderlyingType)

	switch relo.kind {
	case reloTypeIDTarget, reloTypeSize, reloTypeExists:
		if len(relo.accessor) > 1 || relo.accessor[0] != 0 {
			return zero, fmt.Errorf("unexpected accessor %v", relo.accessor)
		}

		err := coreAreTypesCompatible(local, target)
		if errors.Is(err, errImpossibleRelocation) {
			return poison()
		}
		if err != nil {
			return zero, err
		}

		switch relo.kind {
		case reloTypeExists:
			return fixup(1, 1)

		case reloTypeIDTarget:
			return fixup(uint32(relo.id), uint32(targetID))

		case reloTypeSize:
			localSize, err := Sizeof(local)
			if err != nil {
				return zero, err
			}

			targetSize, err := Sizeof(target)
			if err != nil {
				return zero, err
			}

			return fixup(uint32(localSize), uint32(targetSize))
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
			return fixup(uint32(localValue.Value), uint32(targetValue.Value))
		}

	case reloFieldSigned:
		switch local.(type) {
		case *Enum:
			return fixup(1, 1)
		case *Int:
			return fixup(
				uint32(local.(*Int).Encoding&Signed),
				uint32(target.(*Int).Encoding&Signed),
			)
		default:
			return fixupWithoutValidation(0, 0)
		}

	case reloFieldByteOffset, reloFieldByteSize, reloFieldExists, reloFieldLShiftU64, reloFieldRShiftU64:
		if _, ok := target.(*Fwd); ok {
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
			return maybeSkipValidation(fixup(localField.offset, targetField.offset))

		case reloFieldByteSize:
			localSize, err := Sizeof(localField.Type)
			if err != nil {
				return zero, err
			}

			targetSize, err := Sizeof(targetField.Type)
			if err != nil {
				return zero, err
			}
			return maybeSkipValidation(fixup(uint32(localSize), uint32(targetSize)))

		case reloFieldLShiftU64:
			var target uint32
			if bo == binary.LittleEndian {
				targetSize, err := targetField.sizeBits()
				if err != nil {
					return zero, err
				}

				target = uint32(64 - targetField.bitfieldOffset - targetSize)
			} else {
				loadWidth, err := Sizeof(targetField.Type)
				if err != nil {
					return zero, err
				}

				target = uint32(64 - Bits(loadWidth*8) + targetField.bitfieldOffset)
			}
			return fixupWithoutValidation(0, target)

		case reloFieldRShiftU64:
			targetSize, err := targetField.sizeBits()
			if err != nil {
				return zero, err
			}

			return fixupWithoutValidation(0, uint32(64-targetSize))
		}
	}

	return zero, ErrNotSupported
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
	e, ok := t.(*Enum)
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
		return 0, nil
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
		switch localType := local.Type.(type) {
		case composite:
			// For composite types acc is used to find the field in the local type,
			// and then we try to find a field in target with the same name.
			localMembers := localType.members()
			if acc >= len(localMembers) {
				return coreField{}, coreField{}, fmt.Errorf("invalid accessor %d for %s", acc, localType)
			}

			localMember := localMembers[acc]
			if localMember.Name == "" {
				_, ok := localMember.Type.(composite)
				if !ok {
					return coreField{}, coreField{}, fmt.Errorf("unnamed field with type %s: %s", localMember.Type, ErrNotSupported)
				}

				// This is an anonymous struct or union, ignore it.
				local = coreField{
					Type:   localMember.Type,
					offset: local.offset + localMember.Offset.Bytes(),
				}
				localMaybeFlex = false
				continue
			}

			targetType, ok := target.Type.(composite)
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
			targetType, ok := target.Type.(*Array)
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
		if len(visited) >= maxTypeDepth {
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

			comp, ok := member.Type.(composite)
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

	targetEnum, ok := target.(*Enum)
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
 * Returns errImpossibleRelocation if types are not compatible.
 */
func coreAreTypesCompatible(localType Type, targetType Type) error {
	var (
		localTs, targetTs typeDeque
		l, t              = &localType, &targetType
		depth             = 0
	)

	for ; l != nil && t != nil; l, t = localTs.Shift(), targetTs.Shift() {
		if depth >= maxTypeDepth {
			return errors.New("types are nested too deep")
		}

		localType = *l
		targetType = *t

		if reflect.TypeOf(localType) != reflect.TypeOf(targetType) {
			return fmt.Errorf("type mismatch: %w", errImpossibleRelocation)
		}

		switch lv := (localType).(type) {
		case *Void, *Struct, *Union, *Enum, *Fwd, *Int:
			// Nothing to do here

		case *Pointer, *Array:
			depth++
			walkType(localType, localTs.Push)
			walkType(targetType, targetTs.Push)

		case *FuncProto:
			tv := targetType.(*FuncProto)
			if len(lv.Params) != len(tv.Params) {
				return fmt.Errorf("function param mismatch: %w", errImpossibleRelocation)
			}

			depth++
			walkType(localType, localTs.Push)
			walkType(targetType, targetTs.Push)

		default:
			return fmt.Errorf("unsupported type %T", localType)
		}
	}

	if l != nil {
		return fmt.Errorf("dangling local type %T", *l)
	}

	if t != nil {
		return fmt.Errorf("dangling target type %T", *t)
	}

	return nil
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
	doNamesMatch := func(a, b string) error {
		if a == "" || b == "" {
			// allow anonymous and named type to match
			return nil
		}

		if newEssentialName(a) == newEssentialName(b) {
			return nil
		}

		return fmt.Errorf("names don't match: %w", errImpossibleRelocation)
	}

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
		return doNamesMatch(lv.Name, tv.Name)

	case *Fwd:
		tv := targetType.(*Fwd)
		return doNamesMatch(lv.Name, tv.Name)

	default:
		return fmt.Errorf("type %s: %w", localType, ErrNotSupported)
	}
}
