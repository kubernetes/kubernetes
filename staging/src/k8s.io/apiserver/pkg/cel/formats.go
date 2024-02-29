package cel

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

var (
	FormatObject = decls.NewObjectType("kubernetes.Format")
	FormatType   = cel.ObjectType("kubernetes.Format")
)

// Format provdes a CEL representation of kubernetes format
type Format struct {
	Name         string
	ValidateFunc func(string) []string
}

func (d *Format) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	return nil, fmt.Errorf("type conversion error from 'Format' to '%v'", typeDesc)
}

func (d *Format) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case FormatType:
		return d
	case types.TypeType:
		return FormatType
	default:
		return types.NewErr("type conversion error from '%s' to '%s'", FormatType, typeVal)
	}
}

func (d *Format) Equal(other ref.Val) ref.Val {
	otherDur, ok := other.(*Format)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	return types.Bool(d.Name == otherDur.Name)
}

func (d *Format) Type() ref.Type {
	return FormatType
}

func (d *Format) Value() interface{} {
	return d
}
