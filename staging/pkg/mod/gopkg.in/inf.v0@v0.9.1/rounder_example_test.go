package inf_test

import (
	"fmt"
	"os"
	"text/tabwriter"

	"gopkg.in/inf.v0"
)

// This example displays the results of Dec.Round with each of the Rounders.
//
func ExampleRounder() {
	var vals = []struct {
		x string
		s inf.Scale
	}{
		{"-0.18", 1}, {"-0.15", 1}, {"-0.12", 1}, {"-0.10", 1},
		{"-0.08", 1}, {"-0.05", 1}, {"-0.02", 1}, {"0.00", 1},
		{"0.02", 1}, {"0.05", 1}, {"0.08", 1}, {"0.10", 1},
		{"0.12", 1}, {"0.15", 1}, {"0.18", 1},
	}

	var rounders = []struct {
		name    string
		rounder inf.Rounder
	}{
		{"RoundDown", inf.RoundDown}, {"RoundUp", inf.RoundUp},
		{"RoundCeil", inf.RoundCeil}, {"RoundFloor", inf.RoundFloor},
		{"RoundHalfDown", inf.RoundHalfDown}, {"RoundHalfUp", inf.RoundHalfUp},
		{"RoundHalfEven", inf.RoundHalfEven}, {"RoundExact", inf.RoundExact},
	}

	fmt.Println("The results of new(inf.Dec).Round(x, s, inf.RoundXXX):")
	fmt.Println()
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', tabwriter.AlignRight)
	fmt.Fprint(w, "x\ts\t|\t")
	for _, r := range rounders {
		fmt.Fprintf(w, "%s\t", r.name[5:])
	}
	fmt.Fprintln(w)
	for _, v := range vals {
		fmt.Fprintf(w, "%s\t%d\t|\t", v.x, v.s)
		for _, r := range rounders {
			x, _ := new(inf.Dec).SetString(v.x)
			z := new(inf.Dec).Round(x, v.s, r.rounder)
			fmt.Fprintf(w, "%d\t", z)
		}
		fmt.Fprintln(w)
	}
	w.Flush()

	// Output:
	// The results of new(inf.Dec).Round(x, s, inf.RoundXXX):
	//
	//      x s | Down   Up Ceil Floor HalfDown HalfUp HalfEven Exact
	//  -0.18 1 | -0.1 -0.2 -0.1  -0.2     -0.2   -0.2     -0.2 <nil>
	//  -0.15 1 | -0.1 -0.2 -0.1  -0.2     -0.1   -0.2     -0.2 <nil>
	//  -0.12 1 | -0.1 -0.2 -0.1  -0.2     -0.1   -0.1     -0.1 <nil>
	//  -0.10 1 | -0.1 -0.1 -0.1  -0.1     -0.1   -0.1     -0.1  -0.1
	//  -0.08 1 |  0.0 -0.1  0.0  -0.1     -0.1   -0.1     -0.1 <nil>
	//  -0.05 1 |  0.0 -0.1  0.0  -0.1      0.0   -0.1      0.0 <nil>
	//  -0.02 1 |  0.0 -0.1  0.0  -0.1      0.0    0.0      0.0 <nil>
	//   0.00 1 |  0.0  0.0  0.0   0.0      0.0    0.0      0.0   0.0
	//   0.02 1 |  0.0  0.1  0.1   0.0      0.0    0.0      0.0 <nil>
	//   0.05 1 |  0.0  0.1  0.1   0.0      0.0    0.1      0.0 <nil>
	//   0.08 1 |  0.0  0.1  0.1   0.0      0.1    0.1      0.1 <nil>
	//   0.10 1 |  0.1  0.1  0.1   0.1      0.1    0.1      0.1   0.1
	//   0.12 1 |  0.1  0.2  0.2   0.1      0.1    0.1      0.1 <nil>
	//   0.15 1 |  0.1  0.2  0.2   0.1      0.1    0.2      0.2 <nil>
	//   0.18 1 |  0.1  0.2  0.2   0.1      0.2    0.2      0.2 <nil>

}
