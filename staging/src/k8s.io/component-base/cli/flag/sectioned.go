/*
Copyright 2018 The Kubernetes Authors.

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

package flag

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

const (
	usageFmt = "Usage:\n  %s\n"
)

// NamedFlagSets stores named flag sets in the order of calling FlagSet.
type NamedFlagSets struct {
	// Order is an ordered list of flag set names.
	Order []string
	// FlagSets stores the flag sets by name.
	FlagSets map[string]*pflag.FlagSet
	// NormalizeNameFunc is the normalize function which used to initialize FlagSets created by NamedFlagSets.
	NormalizeNameFunc func(f *pflag.FlagSet, name string) pflag.NormalizedName
}

// FlagSet returns the flag set with the given name and adds it to the
// ordered name list if it is not in there yet.
func (nfs *NamedFlagSets) FlagSet(name string) *pflag.FlagSet {
	if nfs.FlagSets == nil {
		nfs.FlagSets = map[string]*pflag.FlagSet{}
	}
	if _, ok := nfs.FlagSets[name]; !ok {
		flagSet := pflag.NewFlagSet(name, pflag.ExitOnError)
		flagSet.SetNormalizeFunc(pflag.CommandLine.GetNormalizeFunc())
		if nfs.NormalizeNameFunc != nil {
			flagSet.SetNormalizeFunc(nfs.NormalizeNameFunc)
		}
		nfs.FlagSets[name] = flagSet
		nfs.Order = append(nfs.Order, name)
	}
	return nfs.FlagSets[name]
}

// PrintSections prints the given names flag sets in sections, with the maximal given column number.
// If cols is zero, lines are not wrapped.
func PrintSections(w io.Writer, fss NamedFlagSets, cols int) {
	for _, name := range fss.Order {
		fs := fss.FlagSets[name]
		if !fs.HasFlags() {
			continue
		}

		wideFS := pflag.NewFlagSet("", pflag.ExitOnError)
		wideFS.AddFlagSet(fs)

		var zzz string
		if cols > 24 {
			zzz = strings.Repeat("z", cols-24)
			wideFS.Int(zzz, 0, strings.Repeat("z", cols-24))
		}

		var buf bytes.Buffer
		fmt.Fprintf(&buf, "\n%s flags:\n\n%s", strings.ToUpper(name[:1])+name[1:], wideFS.FlagUsagesWrapped(cols))

		if cols > 24 {
			i := strings.Index(buf.String(), zzz)
			lines := strings.Split(buf.String()[:i], "\n")
			fmt.Fprint(w, strings.Join(lines[:len(lines)-1], "\n"))
			fmt.Fprintln(w)
		} else {
			fmt.Fprint(w, buf.String())
		}
	}
}

// SetUsageAndHelpFunc set both usage and help function.
// Print the flag sets we need instead of all of them.
func SetUsageAndHelpFunc(cmd *cobra.Command, fss NamedFlagSets, cols int) {
	cmd.SetUsageFunc(func(cmd *cobra.Command) error {
		fmt.Fprintf(cmd.OutOrStderr(), usageFmt, cmd.UseLine())
		PrintSections(cmd.OutOrStderr(), fss, cols)
		return nil
	})
	cmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n\n"+usageFmt, cmd.Long, cmd.UseLine())
		PrintSections(cmd.OutOrStdout(), fss, cols)
	})
}
