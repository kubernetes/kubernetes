// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"flag"
	"fmt"
	"testing"
)

// TestFlag verifies Properties.MustFlag without flag.FlagSet.Parse
func TestFlag(t *testing.T) {
	f := flag.NewFlagSet("src", flag.PanicOnError)
	gotS := f.String("s", "?", "string flag")
	gotI := f.Int("i", -1, "int flag")

	p := NewProperties()
	p.Set("s", "t")
	p.Set("i", "9")
	p.MustFlag(f)

	if want := "t"; *gotS != want {
		t.Errorf("Got string s=%q, want %q", *gotS, want)
	}
	if want := 9; *gotI != want {
		t.Errorf("Got int i=%d, want %d", *gotI, want)
	}
}

// TestFlagOverride verifies Properties.MustFlag with flag.FlagSet.Parse.
func TestFlagOverride(t *testing.T) {
	f := flag.NewFlagSet("src", flag.PanicOnError)
	gotA := f.Int("a", 1, "remain default")
	gotB := f.Int("b", 2, "customized")
	gotC := f.Int("c", 3, "overridden")

	f.Parse([]string{"-c", "4"})

	p := NewProperties()
	p.Set("b", "5")
	p.Set("c", "6")
	p.MustFlag(f)

	if want := 1; *gotA != want {
		t.Errorf("Got remain default a=%d, want %d", *gotA, want)
	}
	if want := 5; *gotB != want {
		t.Errorf("Got customized b=%d, want %d", *gotB, want)
	}
	if want := 4; *gotC != want {
		t.Errorf("Got overriden c=%d, want %d", *gotC, want)
	}
}

func ExampleProperties_MustFlag() {
	x := flag.Int("x", 0, "demo customize")
	y := flag.Int("y", 0, "demo override")

	// Demo alternative for flag.Parse():
	flag.CommandLine.Parse([]string{"-y", "10"})
	fmt.Printf("flagged as x=%d, y=%d\n", *x, *y)

	p := NewProperties()
	p.Set("x", "7")
	p.Set("y", "42") // note discard
	p.MustFlag(flag.CommandLine)
	fmt.Printf("configured to x=%d, y=%d\n", *x, *y)

	// Output:
	// flagged as x=0, y=10
	// configured to x=7, y=10
}
