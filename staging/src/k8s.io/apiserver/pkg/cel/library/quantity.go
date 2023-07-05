package library

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"k8s.io/apimachinery/pkg/api/resource"
	apiservercel "k8s.io/apiserver/pkg/cel"
)

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
