/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"text/tabwriter"
)

type HasFlags interface {
	// Register may be called more than once and should be idempotent.
	Register(ctx context.Context, f *flag.FlagSet)

	// Process may be called more than once and should be idempotent.
	Process(ctx context.Context) error
}

type Command interface {
	HasFlags

	Run(ctx context.Context, f *flag.FlagSet) error
}

func generalHelp(w io.Writer) {
	fmt.Fprintf(w, "Usage of %s:\n", os.Args[0])

	cmds := []string{}
	for name := range commands {
		cmds = append(cmds, name)
	}

	sort.Strings(cmds)

	for _, name := range cmds {
		fmt.Fprintf(w, "  %s\n", name)
	}
}

func commandHelp(w io.Writer, name string, cmd Command, f *flag.FlagSet) {
	type HasUsage interface {
		Usage() string
	}

	fmt.Fprintf(w, "Usage: %s %s [OPTIONS]", os.Args[0], name)
	if u, ok := cmd.(HasUsage); ok {
		fmt.Fprintf(w, " %s", u.Usage())
	}
	fmt.Fprintf(w, "\n")

	type HasDescription interface {
		Description() string
	}

	if u, ok := cmd.(HasDescription); ok {
		fmt.Fprintf(w, "\n%s\n", u.Description())
	}

	n := 0
	f.VisitAll(func(_ *flag.Flag) {
		n += 1
	})

	if n > 0 {
		fmt.Fprintf(w, "\nOptions:\n")
		tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)
		f.VisitAll(func(f *flag.Flag) {
			fmt.Fprintf(tw, "\t-%s=%s\t%s\n", f.Name, f.DefValue, f.Usage)
		})
		tw.Flush()
	}
}

func clientLogout(ctx context.Context, cmd Command) error {
	type logout interface {
		Logout(context.Context) error
	}

	if l, ok := cmd.(logout); ok {
		return l.Logout(ctx)
	}

	return nil
}

func Run(args []string) int {
	hw := os.Stderr
	rc := 1
	hwrc := func(arg string) {
		if arg == "-h" {
			hw = os.Stdout
			rc = 0
		}
	}

	var err error

	if len(args) == 0 {
		generalHelp(hw)
		return rc
	}

	// Look up real command name in aliases table.
	name, ok := aliases[args[0]]
	if !ok {
		name = args[0]
	}

	cmd, ok := commands[name]
	if !ok {
		hwrc(name)
		generalHelp(hw)
		return rc
	}

	fs := flag.NewFlagSet("", flag.ContinueOnError)
	fs.SetOutput(ioutil.Discard)

	ctx := context.Background()
	cmd.Register(ctx, fs)

	if err = fs.Parse(args[1:]); err != nil {
		goto error
	}

	if err = cmd.Process(ctx); err != nil {
		goto error
	}

	if err = cmd.Run(ctx, fs); err != nil {
		goto error
	}

	if err = clientLogout(ctx, cmd); err != nil {
		goto error
	}

	return 0

error:
	if err == flag.ErrHelp {
		if len(args) == 2 {
			hwrc(args[1])
		}
		commandHelp(hw, args[0], cmd, fs)
	} else {
		fmt.Fprintf(os.Stderr, "%s: %s\n", os.Args[0], err)
	}

	_ = clientLogout(ctx, cmd)

	return rc
}
