/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cmd

import (
	"fmt"
	"io"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	protofile "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd"
)

//TODO: those whole file could be generated since it's templated

// TestOptions
type TestOptions struct {
	//Flags must be the first element of Options struct, and must be exposed
	Flags *protofile.TestCmd
}

func (o *TestOptions) Complete(f cmdutil.Factory, in io.Reader, out, err io.Writer) error {
	fmt.Println("This is Complete function")
	fmt.Println("int32 is", o.Flags.GetInt32Flag())
	fmt.Println("bool is", o.Flags.GetBoolFlag())
	fmt.Println("array is", o.Flags.GetArrayFlag())
	fmt.Println("time is", o.Flags.GetTimeFlag())
	fmt.Println("string is", o.Flags.GetStringFlag())
	fmt.Println("int64 is", o.Flags.GetInt64Flag())
	fmt.Println("===============================")
	fmt.Println()
	return nil
}

func (o *TestOptions) Validate(f cmdutil.Factory, in io.Reader, out, err io.Writer) error {
	fmt.Println("This is Validate function")
	fmt.Println("int32 is", o.Flags.GetInt32Flag())
	fmt.Println("bool is", o.Flags.GetBoolFlag())
	fmt.Println("array is", o.Flags.GetArrayFlag())
	fmt.Println("time is", o.Flags.GetTimeFlag())
	fmt.Println("string is", o.Flags.GetStringFlag())
	fmt.Println("int64 is", o.Flags.GetInt64Flag())
	fmt.Println("===============================")
	fmt.Println()
	return nil
}

func (o *TestOptions) Run(f cmdutil.Factory, in io.Reader, out, err io.Writer) error {
	fmt.Println("This is Run function")
	fmt.Println("int32 is", o.Flags.GetInt32Flag())
	fmt.Println("bool is", o.Flags.GetBoolFlag())
	fmt.Println("array is", o.Flags.GetArrayFlag())
	fmt.Println("time is", o.Flags.GetTimeFlag())
	fmt.Println("string is", o.Flags.GetStringFlag())
	fmt.Println("int64 is", o.Flags.GetInt64Flag())
	fmt.Println("===============================")
	fmt.Println()
	return nil
}
