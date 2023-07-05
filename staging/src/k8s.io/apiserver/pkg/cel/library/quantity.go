package library

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"k8s.io/apimachinery/pkg/api/resource"
	apiservercel "k8s.io/apiserver/pkg/cel"
)

// Quantity provides a CEL function library extension of Kubernetes
// resource.Quantity parsing functions. See `resource.Quantity`
// documentation for more detailed information about the format itself.
//
// quantity
//
// Converts a string to a URL or results in an error if the string is not a valid URL. The URL must be an absolute URI
// or an absolute path.
//
//	quantity(<string>) <Quantity>
//
// Examples:
//
//	quantity('1.5G') // returns a Quantity
//	quantity('200k') // returns a Quantity
//	quantity('200K') // error
//	quantity('Three') // error
//	quantity('Mi') // error
//
// isQuantity
//
// Returns true if a string is a valid Quantity. isQuantity returns true if and
// only if quantity does not result in error.
//
//	isQuantity( <string>) <bool>
//
// Examples:
//
//	isQuantity('1.3G') // returns true
//	isQuantity('1.3Gi') // returns true
//	isQuantity('1,3G') // returns false
//	isQuantity('10000k') // returns true
//	isQuantity('200K') // returns false
//	isQuantity('Three') // returns false
//	isQuantity('Mi') // returns false
//
// asApproximateFloat / asInteger
//
// Return the parsed components of a URL.
//
//   - asInteger: returns a representation of the current value as an int64 if a fast conversion
//     is possible. If false is returned, callers must use the inf.Dec form of this quantity.
//
//   - asApproximateFloat: returns a float64 representation of the quantity which may
//     lose precision. If the value of the quantity is outside the range of a float64
//     +Inf/-Inf will be returned.
//
//   - sign: Returns `1` if the quantity is positive, `-1` if it is negative. `0` if it is zero
//
//   - add: Returns sum of two quantities or a quantity and an integer
//  -  sub: Returns difference between two quantities or a quantity and an integer
//
//     <Quantity>.asInteger() <int>
//     <Quantity>.asApproximateFloat() <float>
//     <Quantity>.sign() <int>
//     <Quantity>.add(<quantity>) <quantity>
//     <Quantity>.add(<integer>) <quantity>
//     <Quantity>.sub(<quantity>) <quantity>
//     <Quantity>.sub(<integer>) <quantity>
//
// Examples:
// quantity("50k").add(20).sub(quantity("100k")).sub(-50000) == quantity("20") // returns true
// quantity("50k").asInteger() == 50000 // returns true
// quantity("50k").sub(20000).asApproximateFloat() == 30000 // returns true

func Quantity() cel.EnvOption {
	return cel.Lib(quantityLib)
}

var quantityLib = &quantity{}

type quantity struct{}

var quantityLibraryDecls = map[string][]cel.FunctionOpt{
	"quantity": {
		cel.Overload("string_to_quantity", []*cel.Type{cel.StringType}, apiservercel.QuantityType, cel.UnaryBinding((stringToQuantity))),
	},
	"isQuantity": {
		cel.Overload("is_quantity_string", []*cel.Type{cel.StringType}, cel.BoolType, cel.UnaryBinding(isQuantity)),
	},
	"sign": {
		cel.Overload("quantity_sign", []*cel.Type{apiservercel.QuantityType}, cel.IntType, cel.UnaryBinding(quantityGetSign)),
	},
	"isGreaterThan": {
		cel.MemberOverload("quantity_is_greater_than", []*cel.Type{apiservercel.QuantityType, apiservercel.QuantityType}, cel.BoolType, cel.BinaryBinding(quantityIsGreaterThan)),
	},
	"isLessThan": {
		cel.MemberOverload("quantity_is_less_than", []*cel.Type{apiservercel.QuantityType, apiservercel.QuantityType}, cel.BoolType, cel.BinaryBinding(quantityIsLessThan)),
	},
	"compareTo": {
		cel.MemberOverload("quantity_compare_to", []*cel.Type{apiservercel.QuantityType, apiservercel.QuantityType}, cel.IntType, cel.BinaryBinding(quantityCompareTo)),
	},
	"asApproximateFloat": {
		cel.MemberOverload("quantity_get_float", []*cel.Type{apiservercel.QuantityType}, cel.DoubleType, cel.UnaryBinding(quantityGetApproximateFloat)),
	},
	"asInteger": {
		cel.MemberOverload("quantity_get_int", []*cel.Type{apiservercel.QuantityType}, cel.IntType, cel.UnaryBinding(quantityGetValue)),
	},
	"add": {
		cel.MemberOverload("quantity_add", []*cel.Type{apiservercel.QuantityType, apiservercel.QuantityType}, apiservercel.QuantityType, cel.BinaryBinding(quantityAdd)),
		cel.MemberOverload("quantity_add_int", []*cel.Type{apiservercel.QuantityType, cel.IntType}, apiservercel.QuantityType, cel.BinaryBinding(quantityAddInt)),
		// cel.MemberOverload("quantity_add_float", []*cel.Type{apiservercel.QuantityType, cel.DoubleType}, apiservercel.QuantityType, cel.UnaryBinding(quantityAddFloat)),
	},
	"sub": {
		cel.MemberOverload("quantity_sub", []*cel.Type{apiservercel.QuantityType, apiservercel.QuantityType}, apiservercel.QuantityType, cel.BinaryBinding(quantitySub)),
		cel.MemberOverload("quantity_sub_int", []*cel.Type{apiservercel.QuantityType, cel.IntType}, apiservercel.QuantityType, cel.BinaryBinding(quantitySubInt)),
		// cel.MemberOverload("quantity_sub_float", []*cel.Type{apiservercel.QuantityType, cel.DoubleType}, apiservercel.QuantityType, cel.UnaryBinding(quantitySubFloat)),
	},
}

func (*quantity) CompileOptions() []cel.EnvOption {
	options := make([]cel.EnvOption, 0, len(quantityLibraryDecls))
	for name, overloads := range quantityLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*quantity) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func isQuantity(arg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	_, err := resource.ParseQuantity(str)
	if err != nil {
		return types.Bool(false)
	}

	return types.Bool(true)
}

func stringToQuantity(arg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q, err := resource.ParseQuantity(str)
	if err != nil {
		return types.WrapErr(err)
	}

	return apiservercel.Quantity{Quantity: &q}
}

func quantityGetApproximateFloat(arg ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Double(q.AsApproximateFloat64())
}

func quantityGetValue(arg ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Int(q.Value())
}

func quantityGetSign(arg ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Int(q.Sign())
}

func quantityIsGreaterThan(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Bool(q.Cmp(*q2) == -1)
}

func quantityIsLessThan(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Bool(q.Cmp(*q2) == 1)
}

func quantityCompareTo(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Int(q.Cmp(*q2))
}

func quantityAdd(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	copy := *q
	copy.Add(*q2)
	return &apiservercel.Quantity{
		Quantity: &copy,
	}
}

func quantityAddInt(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(types.Int)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2Converted := *resource.NewQuantity(int64(q2), resource.DecimalExponent)

	copy := *q
	copy.Add(q2Converted)
	return &apiservercel.Quantity{
		Quantity: &copy,
	}
}

func quantitySub(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	copy := *q
	copy.Sub(*q2)
	return &apiservercel.Quantity{
		Quantity: &copy,
	}
}

func quantitySubInt(arg ref.Val, other ref.Val) ref.Val {
	q, ok := arg.Value().(*resource.Quantity)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2, ok := other.Value().(types.Int)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	q2Converted := *resource.NewQuantity(int64(q2), resource.DecimalExponent)

	copy := *q
	copy.Sub(q2Converted)
	return &apiservercel.Quantity{
		Quantity: &copy,
	}
}
