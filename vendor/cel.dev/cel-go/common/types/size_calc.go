// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"math"
	"reflect"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"

	structpb "google.golang.org/protobuf/types/known/structpb"
)

const (
	defaultSizeCalculatorMaxDepth         = 5
	defaultSizeCalculatorMaxTraversal     = 10000
	defaultSizeCalculatorStringUnitLength = 10
)

// SizeCalculatorOption configures a SizeCalculator instance.
type SizeCalculatorOption func(*SizeCalculator)

// SizeCalculatorMaxDepth sets the maximum object depth limit before saturating to math.MaxUint32.
func SizeCalculatorMaxDepth(depth int) SizeCalculatorOption {
	return func(s *SizeCalculator) {
		s.maxDepth = depth
	}
}

// SizeCalculatorMaxTraversal sets the maximum object traversal limit before saturating to math.MaxUint32.
func SizeCalculatorMaxTraversal(traversal int) SizeCalculatorOption {
	return func(s *SizeCalculator) {
		s.maxTraversal = traversal
	}
}

// SizeCalculatorStringUnitLength sets the number of string or bytes value bytes which count as
// a single element (default 10). Values less than 1 are treated as 1, meaning each byte counts
// as a whole element.
//
// String sizes are measured in bytes rather than characters so that sizing large values is
// O(1) rather than a full UTF-8 scan per observation; byte length is never smaller than the
// character count, so byte-based sizing is conservative for limit enforcement.
func SizeCalculatorStringUnitLength(length int) SizeCalculatorOption {
	return func(s *SizeCalculator) {
		if length < 1 {
			length = 1
		}
		s.stringUnitLength = length
	}
}

// SizeCalculator calculates the recursive element size of values.
//
// Aggregate values may memoize their computed size on first calculation as an optimization
// for repeated sizing of shared structures. The memoized size reflects the configuration of
// the calculator which first sized the value; hosts requiring differently configured
// calculators, e.g. distinct depth or traversal limits, should not share value instances
// across them. Memoized totals are also a snapshot of the value's contents at first sizing:
// hosts which mutate data underlying a sized aggregate, e.g. a proto message held as a list
// element, will observe the total computed before the mutation. Sizes computed from
// calculations aborted at the depth or traversal limits are never memoized.
type SizeCalculator struct {
	version          int
	maxDepth         int
	maxTraversal     int
	stringUnitLength int
}

// NewSizeCalculator returns a new SizeCalculator configured with optional SizeCalculatorOption settings.
func NewSizeCalculator(opts ...SizeCalculatorOption) *SizeCalculator {
	s := &SizeCalculator{
		version:          0,
		maxDepth:         defaultSizeCalculatorMaxDepth,
		maxTraversal:     defaultSizeCalculatorMaxTraversal,
		stringUnitLength: defaultSizeCalculatorStringUnitLength,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Version returns the calculation version.
func (s *SizeCalculator) Version() int {
	return s.version
}

var sizeContextPool = sync.Pool{
	New: func() any {
		return &sizeContext{}
	},
}

type sizeContext struct {
	calc           *SizeCalculator
	depth          int
	traversalCount int
	limitExceeded  bool
}

func (c *sizeContext) visitNode() bool {
	c.traversalCount++
	if c.traversalCount > c.calc.maxTraversal || c.depth > c.calc.maxDepth {
		c.limitExceeded = true
		return false
	}
	return true
}

// aggregateSizeStatus exposes whether the in-flight size computation has exceeded the
// calculator's depth or traversal limits.
type aggregateSizeStatus interface {
	aggregateSizeLimitExceeded() bool
}

// aggregateSizeLimitExceeded implements the aggregateSizeStatus interface method.
func (c *sizeContext) aggregateSizeLimitExceeded() bool {
	return c.limitExceeded
}

// cacheableAggregateSize reports whether a size computed with the given sizer is safe to
// memoize on the value. Only totals from computations which verifiably stayed within the
// calculator's depth and traversal limits are stable properties of the value; totals from
// aborted computations depend on where in the traversal the value was encountered and would
// poison the memoized size.
func cacheableAggregateSize(sizer AggregateSizer) bool {
	status, ok := sizer.(aggregateSizeStatus)
	return ok && !status.aggregateSizeLimitExceeded()
}

// ApproximateAggregateSize captures the outcome of an aggregate size computation.
//
// The Size saturates at math.MaxUint32 when the accumulated element count overflows uint32.
// LimitExceeded reports the computation was aborted because the value was too expensive to
// traverse (too deep, or too many nodes visited); in that case Size is also math.MaxUint32,
// but the value's true size may be smaller — the two conditions are distinguishable by the flag.
type ApproximateAggregateSize struct {
	Size          uint32
	LimitExceeded bool
}

// ApproximateAggregateSize returns the aggregate size of the input value along with an indication
// of whether the computation was aborted due to the calculator's depth or traversal limits.
func (s *SizeCalculator) ApproximateAggregateSize(val any) ApproximateAggregateSize {
	ctx := sizeContextPool.Get().(*sizeContext)
	ctx.calc = s
	est := ctx.estimateAggregateSize(val)
	ctx.calc = nil

	sizeContextPool.Put(ctx)
	return est
}

// AggregateSize implements the AggregateSizer interface by returning the computed size.
func (s *SizeCalculator) AggregateSize(val any) uint32 {
	return s.ApproximateAggregateSize(val).Size
}

// stringSize converts a character or byte length to an element count where stringUnitLength
// characters count as a single element, rounding up with a minimum size of 1.
func (s *SizeCalculator) stringSize(length int) uint32 {
	if length <= 0 {
		return 1
	}
	return safeUint32FromInt((length + s.stringUnitLength - 1) / s.stringUnitLength)
}

// AggregateSize implements the ref.Val interface and allows for the generation of nested
// child context values which are necessary for correct traversal count tracking.
func (c *sizeContext) AggregateSize(val any) uint32 {
	c.depth++
	res := c.aggregateSize(val)
	c.depth--
	return res
}

func (c *sizeContext) aggregateSize(val any) uint32 {
	if !c.visitNode() {
		return math.MaxUint32
	}
	switch v := val.(type) {
	case String:
		return c.calc.stringSize(len(v))
	case Bytes:
		return c.calc.stringSize(len(v))
	case AggregateSizeVisitor:
		return v.AggregateSize(c)
	case traits.Foldable:
		f := foldableAggregateSizer{sizer: c, total: 1}
		v.Fold(&f)
		return f.total
	case traits.Mapper:
		total := uint32(1)
		it := v.Iterator()
		for it.HasNext() == True {
			key := it.Next()
			val, _ := v.Find(key)
			total = safeAddUint32(total, c.AggregateSize(key))
			total = safeAddUint32(total, c.AggregateSize(val))
		}
		return total
	case traits.Lister:
		total := uint32(1)
		it := v.Iterator()
		for it.HasNext() == True {
			total = safeAddUint32(total, c.AggregateSize(it.Next()))
		}
		return total
	case traits.Sizer:
		return max(1, safeUint32FromBoxedInt(v.Size().(Int)))
	case Bool, Int, Uint, Double, Duration, Timestamp, Null, *Type, *Err, *Unknown:
		return 1
	case ref.Val:
		return c.aggregateSize(v.Value())
	case protoreflect.Value:
		return getProtoValueAggregateSize(c, v)
	case protoreflect.MapKey:
		return getProtoValueAggregateSize(c, v.Value())
	case protoreflect.Message:
		return getProtoMessageAggregateSize(c, v)
	case protoreflect.List:
		return getProtoListAggregateSize(c, v)
	case protoreflect.Map:
		return getProtoMapAggregateSize(c, v)
	case proto.Message:
		if v == nil {
			return 1
		}
		return getProtoMessageAggregateSize(c, v.ProtoReflect())
	case string:
		return c.calc.stringSize(len(v))
	case []byte:
		return c.calc.stringSize(len(v))
	case int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64, bool, time.Time, time.Duration, nil:
		return 1
	case reflect.Value:
		return getReflectValueAggregateSize(c, v)
	default:
		return getReflectValueAggregateSize(c, reflect.ValueOf(val))
	}
}

// estimateAggregateSize evaluates the aggregate size using this context and resets state.
func (c *sizeContext) estimateAggregateSize(val any) ApproximateAggregateSize {
	c.depth = 0
	c.traversalCount = 0
	c.limitExceeded = false

	size := c.AggregateSize(val)
	return ApproximateAggregateSize{Size: size, LimitExceeded: c.limitExceeded}
}

func getProtoValueAggregateSize(c *sizeContext, v protoreflect.Value) uint32 {
	if !v.IsValid() {
		return 1
	}
	switch val := v.Interface().(type) {
	case protoreflect.Message:
		return getProtoMessageAggregateSize(c, val)
	case protoreflect.List:
		return getProtoListAggregateSize(c, val)
	case protoreflect.Map:
		return getProtoMapAggregateSize(c, val)
	case string:
		return c.calc.stringSize(len(val))
	case []byte:
		return c.calc.stringSize(len(val))
	default:
		return 1
	}
}

func getProtoFieldAggregateSize(c *sizeContext, fd protoreflect.FieldDescriptor, v protoreflect.Value) uint32 {
	if !c.visitNode() {
		return math.MaxUint32
	}
	if fd.IsMap() {
		c.depth++
		sz := getProtoMapAggregateSize(c, v.Map())
		c.depth--
		return sz
	}
	if fd.IsList() {
		c.depth++
		sz := getProtoListAggregateSize(c, v.List())
		c.depth--
		return sz
	}
	if fd.Message() != nil {
		return c.AggregateSize(v.Message())
	}
	switch fd.Kind() {
	case protoreflect.StringKind:
		return c.calc.stringSize(len(v.String()))
	case protoreflect.BytesKind:
		return c.calc.stringSize(len(v.Bytes()))
	default:
		return 1
	}
}

func getProtoMessageAggregateSize(c *sizeContext, m protoreflect.Message) uint32 {
	if !m.IsValid() {
		return 1
	}
	if !c.visitNode() {
		return math.MaxUint32
	}
	total := uint32(1)
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		total = safeAddUint32(total, getProtoFieldAggregateSize(c, fd, v))
		return true
	})
	return total
}

func getProtoListAggregateSize(c *sizeContext, l protoreflect.List) uint32 {
	if !l.IsValid() {
		return 1
	}
	if !c.visitNode() {
		return math.MaxUint32
	}
	total := uint32(1)
	for i := range l.Len() {
		total = safeAddUint32(total, c.AggregateSize(l.Get(i)))
	}
	return total
}

func getProtoMapAggregateSize(c *sizeContext, m protoreflect.Map) uint32 {
	if !m.IsValid() {
		return 1
	}
	if !c.visitNode() {
		return math.MaxUint32
	}
	total := uint32(1)
	m.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
		total = safeAddUint32(total, c.AggregateSize(k))
		total = safeAddUint32(total, c.AggregateSize(v))
		return true
	})
	return total
}

// getSliceElementsAggregateSize computes the aggregate size of known slice payloads without reflection.
func getSliceElementsAggregateSize(sizer AggregateSizer, val any) (uint32, bool) {
	switch v := val.(type) {
	case []byte:
		if sc, ok := sizer.(*sizeContext); ok {
			return sc.calc.stringSize(len(v)), true
		}
		return 1, true
	case []int:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []int8:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []int16:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []int32:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []int64:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []uint:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []uint16:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []uint32:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []uint64:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []float32:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []float64:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []bool:
		return safeAddUint32(1, safeUint32FromInt(len(v))), true
	case []string:
		total := uint32(1)
		for _, s := range v {
			total = safeAddUint32(total, sizer.AggregateSize(s))
		}
		return total, true
	case []ref.Val:
		total := uint32(1)
		for _, elem := range v {
			total = safeAddUint32(total, sizer.AggregateSize(elem))
		}
		return total, true
	case protoreflect.List:
		total := uint32(1)
		for i := range v.Len() {
			total = safeAddUint32(total, sizer.AggregateSize(v.Get(i)))
		}
		return total, true
	case *structpb.ListValue:
		if v == nil {
			return 1, true
		}
		total := uint32(1)
		for _, elem := range v.GetValues() {
			total = safeAddUint32(total, sizer.AggregateSize(elem))
		}
		return total, true
	case []any:
		total := uint32(1)
		for _, elem := range v {
			total = safeAddUint32(total, sizer.AggregateSize(elem))
		}
		return total, true
	default:
		return 0, false
	}
}

// getMapElementsAggregateSize computes the aggregate size of a map payload.
func getMapElementsAggregateSize(sizer AggregateSizer, val any) (uint32, bool) {
	switch v := val.(type) {
	case map[string]string:
		total := uint32(1)
		for k, val := range v {
			total = safeAddUint32(total, sizer.AggregateSize(k))
			total = safeAddUint32(total, sizer.AggregateSize(val))
		}
		return total, true
	case map[string]any:
		total := uint32(1)
		for k, val := range v {
			total = safeAddUint32(total, sizer.AggregateSize(k))
			total = safeAddUint32(total, sizer.AggregateSize(val))
		}
		return total, true
	case map[ref.Val]ref.Val:
		total := uint32(1)
		for k, val := range v {
			total = safeAddUint32(total, sizer.AggregateSize(k))
			total = safeAddUint32(total, sizer.AggregateSize(val))
		}
		return total, true
	case *structpb.Struct:
		if v == nil {
			return 1, true
		}
		total := uint32(1)
		for k, val := range v.GetFields() {
			total = safeAddUint32(total, sizer.AggregateSize(k))
			total = safeAddUint32(total, sizer.AggregateSize(val))
		}
		return total, true
	default:
		return 0, false
	}
}

func getReflectValueAggregateSize(c *sizeContext, fieldVal reflect.Value) uint32 {
	if !fieldVal.IsValid() {
		return 1
	}
	if !c.visitNode() {
		return math.MaxUint32
	}
	switch fieldVal.Kind() {
	case reflect.String:
		return c.calc.stringSize(fieldVal.Len())
	case reflect.Slice:
		elemType := fieldVal.Type().Elem()
		if elemType.Kind() == reflect.Uint8 {
			return c.calc.stringSize(fieldVal.Len())
		}
		if fieldVal.CanInterface() {
			if total, ok := getSliceElementsAggregateSize(c, fieldVal.Interface()); ok {
				return total
			}
		}
		total := safeAddUint32(1, safeUint32FromInt(fieldVal.Len()))
		switch elemType.Kind() {
		case reflect.String, reflect.Struct, reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map, reflect.Interface:
			total = 1
			for i := 0; i < fieldVal.Len(); i++ {
				total = safeAddUint32(total, c.AggregateSize(fieldVal.Index(i)))
			}
		}
		return total
	case reflect.Array:
		elemType := fieldVal.Type().Elem()
		if elemType.Kind() == reflect.Uint8 {
			return c.calc.stringSize(fieldVal.Len())
		}
		total := safeAddUint32(1, safeUint32FromInt(fieldVal.Len()))
		switch elemType.Kind() {
		case reflect.String, reflect.Struct, reflect.Pointer, reflect.Slice, reflect.Array, reflect.Map, reflect.Interface:
			total = 1
			for i := 0; i < fieldVal.Len(); i++ {
				total = safeAddUint32(total, c.AggregateSize(fieldVal.Index(i)))
			}
		}
		return total
	case reflect.Map:
		if fieldVal.CanInterface() {
			if total, ok := getMapElementsAggregateSize(c, fieldVal.Interface()); ok {
				return total
			}
		}
		total := uint32(1)
		iter := fieldVal.MapRange()
		for iter.Next() {
			total = safeAddUint32(total, c.AggregateSize(iter.Key()))
			total = safeAddUint32(total, c.AggregateSize(iter.Value()))
		}
		return total
	case reflect.Pointer, reflect.Interface:
		if fieldVal.IsNil() {
			return 1
		}
		if sz, ok := checkCustomSizer(c, fieldVal); ok {
			return sz
		}
		return getReflectValueAggregateSize(c, fieldVal.Elem())
	case reflect.Struct:
		if fieldVal.Type() == timestampType || fieldVal.Type() == durationType {
			return 1
		}
		if sz, ok := checkCustomSizer(c, fieldVal); ok {
			return sz
		}
		total := uint32(1)
		t := fieldVal.Type()
		numFields := fieldVal.NumField()
		for i := range numFields {
			if !t.Field(i).IsExported() {
				continue
			}
			fVal := fieldVal.Field(i)
			if !fVal.IsValid() || fVal.IsZero() {
				continue
			}
			total = safeAddUint32(total, c.AggregateSize(fVal))
		}
		return total
	default:
		return 1
	}
}

var (
	aggregateSizeVisitorType = reflect.TypeFor[AggregateSizeVisitor]()
	sizerType                = reflect.TypeFor[traits.Sizer]()
	protoMessageType         = reflect.TypeFor[proto.Message]()
)

func checkCustomSizer(c *sizeContext, fieldVal reflect.Value) (uint32, bool) {
	if !fieldVal.IsValid() || !fieldVal.CanInterface() {
		return 0, false
	}
	t := fieldVal.Type()
	if t.Implements(aggregateSizeVisitorType) {
		if sizer, ok := fieldVal.Interface().(AggregateSizeVisitor); ok {
			return sizer.AggregateSize(c), true
		}
	}
	if t.Implements(sizerType) {
		if sizer, ok := fieldVal.Interface().(traits.Sizer); ok {
			return max(1, safeUint32FromBoxedInt(sizer.Size().(Int))), true
		}
	}
	if t.Implements(protoMessageType) {
		if fieldVal.Kind() == reflect.Pointer && fieldVal.IsNil() {
			return 1, true
		}
		if sizer, ok := fieldVal.Interface().(proto.Message); ok {
			if sizer == nil {
				return 1, true
			}
			return getProtoMessageAggregateSize(c, sizer.ProtoReflect()), true
		}
	}
	return 0, false
}
