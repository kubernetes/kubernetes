/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package vm

import (
	"context"
	"errors"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type question struct {
	*flags.VirtualMachineFlag

	answer string
}

func init() {
	cli.Register("vm.question", &question{})
}

func (cmd *question) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.StringVar(&cmd.answer, "answer", "", "Answer to question")
}

func (cmd *question) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *question) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return errors.New("No VM specified")
	}

	var mvm mo.VirtualMachine

	pc := property.DefaultCollector(c)
	err = pc.RetrieveOne(ctx, vm.Reference(), []string{"runtime.question"}, &mvm)
	if err != nil {
		return err
	}

	q := mvm.Runtime.Question
	if q == nil {
		fmt.Printf("No pending question\n")
		return nil
	}

	// Print question if no answer is specified
	if cmd.answer == "" {
		fmt.Printf("Question:\n%s\n\n", q.Text)
		fmt.Printf("Possible answers:\n")
		for _, e := range q.Choice.ChoiceInfo {
			ed := e.(*types.ElementDescription)
			fmt.Printf("%s) %s\n", ed.Key, ed.Description.Label)
		}
		return nil
	}

	// Answer question
	return vm.Answer(ctx, q.Id, cmd.answer)
}
