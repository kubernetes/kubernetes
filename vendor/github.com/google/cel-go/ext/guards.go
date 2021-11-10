// Copyright 2020 Google LLC
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

package ext

import (
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter/functions"
)

// function invocation guards for common call signatures within extension functions.

func callInBytesOutString(fn func([]byte) (string, error)) functions.UnaryOp {
	return func(val ref.Val) ref.Val {
		vVal, ok := val.(types.Bytes)
		if !ok {
			return types.MaybeNoSuchOverloadErr(val)
		}
		str, err := fn([]byte(vVal))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.String(str)
	}
}

func callInStrOutBytes(fn func(string) ([]byte, error)) functions.UnaryOp {
	return func(val ref.Val) ref.Val {
		vVal, ok := val.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(val)
		}
		byt, err := fn(string(vVal))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.Bytes(byt)
	}
}

func callInStrOutStr(fn func(string) (string, error)) functions.UnaryOp {
	return func(val ref.Val) ref.Val {
		vVal, ok := val.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(val)
		}
		str, err := fn(string(vVal))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.String(str)
	}
}

func callInStrIntOutStr(fn func(string, int64) (string, error)) functions.BinaryOp {
	return func(val, arg ref.Val) ref.Val {
		vVal, ok := val.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(val)
		}
		argVal, ok := arg.(types.Int)
		if !ok {
			return types.MaybeNoSuchOverloadErr(arg)
		}
		out, err := fn(string(vVal), int64(argVal))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.String(out)
	}
}

func callInStrStrOutInt(fn func(string, string) (int64, error)) functions.BinaryOp {
	return func(val, arg ref.Val) ref.Val {
		vVal, ok := val.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(val)
		}
		argVal, ok := arg.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(arg)
		}
		out, err := fn(string(vVal), string(argVal))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.Int(out)
	}
}

func callInStrStrOutListStr(fn func(string, string) ([]string, error)) functions.BinaryOp {
	return func(val, arg ref.Val) ref.Val {
		vVal, ok := val.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(val)
		}
		argVal, ok := arg.(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(arg)
		}
		out, err := fn(string(vVal), string(argVal))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.DefaultTypeAdapter.NativeToValue(out)
	}
}

func callInStrIntIntOutStr(fn func(string, int64, int64) (string, error)) functions.FunctionOp {
	return func(args ...ref.Val) ref.Val {
		if len(args) != 3 {
			return types.NoSuchOverloadErr()
		}
		vVal, ok := args[0].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[0])
		}
		arg1Val, ok := args[1].(types.Int)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[1])
		}
		arg2Val, ok := args[2].(types.Int)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[2])
		}
		out, err := fn(string(vVal), int64(arg1Val), int64(arg2Val))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.String(out)
	}
}

func callInStrStrStrOutStr(fn func(string, string, string) (string, error)) functions.FunctionOp {
	return func(args ...ref.Val) ref.Val {
		if len(args) != 3 {
			return types.NoSuchOverloadErr()
		}
		vVal, ok := args[0].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[0])
		}
		arg1Val, ok := args[1].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[1])
		}
		arg2Val, ok := args[2].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[2])
		}
		out, err := fn(string(vVal), string(arg1Val), string(arg2Val))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.String(out)
	}
}

func callInStrStrIntOutInt(fn func(string, string, int64) (int64, error)) functions.FunctionOp {
	return func(args ...ref.Val) ref.Val {
		if len(args) != 3 {
			return types.NoSuchOverloadErr()
		}
		vVal, ok := args[0].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[0])
		}
		arg1Val, ok := args[1].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[1])
		}
		arg2Val, ok := args[2].(types.Int)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[2])
		}
		out, err := fn(string(vVal), string(arg1Val), int64(arg2Val))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.Int(out)
	}
}

func callInStrStrIntOutListStr(fn func(string, string, int64) ([]string, error)) functions.FunctionOp {
	return func(args ...ref.Val) ref.Val {
		if len(args) != 3 {
			return types.NoSuchOverloadErr()
		}
		vVal, ok := args[0].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[0])
		}
		arg1Val, ok := args[1].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[1])
		}
		arg2Val, ok := args[2].(types.Int)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[2])
		}
		out, err := fn(string(vVal), string(arg1Val), int64(arg2Val))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.DefaultTypeAdapter.NativeToValue(out)
	}
}

func callInStrStrStrIntOutStr(fn func(string, string, string, int64) (string, error)) functions.FunctionOp {
	return func(args ...ref.Val) ref.Val {
		if len(args) != 4 {
			return types.NoSuchOverloadErr()
		}
		vVal, ok := args[0].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[0])
		}
		arg1Val, ok := args[1].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[1])
		}
		arg2Val, ok := args[2].(types.String)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[2])
		}
		arg3Val, ok := args[3].(types.Int)
		if !ok {
			return types.MaybeNoSuchOverloadErr(args[3])
		}
		out, err := fn(string(vVal), string(arg1Val), string(arg2Val), int64(arg3Val))
		if err != nil {
			return types.NewErr(err.Error())
		}
		return types.String(out)
	}
}
