/*
Copyright The Kubernetes Authors.

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

package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/util/managedfields/formal/bridge"
)

func run() error {
	model := flag.String("model", "", "path to the built ssa-model executable (required)")
	corpusPath := flag.String("corpus", "testdata/bridge.yaml", "independent YAML corpus")
	reproducer := flag.String("reproducer", "", "replay a failing request prefix without adding generated sequences")
	seed := flag.Int64("seed", 336, "deterministic generated-sequence seed")
	count := flag.Int("generated", 64, "sequences per scalar, structural, nullable-container and bootstrap family")
	steps := flag.Int("steps", 32, "requests per generated sequence")
	flag.Parse()
	if *model == "" || *count < 0 || *steps < 1 || flag.NArg() != 0 {
		return fmt.Errorf("supply -model, nonnegative -generated, positive -steps, and no positional arguments")
	}
	inputPath := *corpusPath
	if *reproducer != "" {
		inputPath = *reproducer
	}
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return err
	}
	read := bridge.ReadCorpus
	if *reproducer != "" {
		read = bridge.ReadReproducer
	}
	corpus, err := read(data)
	if err != nil {
		return err
	}
	cases := corpus.Cases
	if *reproducer == "" {
		cases = append(cases, bridge.GeneratedScalarCases(*seed, *count, *steps)...)
		cases = append(cases, bridge.GeneratedStructuralCases(*seed, *count, *steps)...)
		cases = append(cases, bridge.GeneratedNullableContainerCases(*seed, *count, *steps)...)
		cases = append(cases, bridge.GeneratedBootstrapCases(*seed, *count, *steps)...)
	}
	summary, err := bridge.Compare(context.Background(), *model, cases)
	if err != nil {
		return err
	}
	fmt.Printf("Compared %d cases / %d transitions: %d accepted, %d conflicts, %d proposal differences (seed %d).\n",
		summary.Cases, summary.Steps, summary.Accepted, summary.Conflicts,
		summary.Differences, *seed)
	if *reproducer == "" {
		pairs, err := bridge.CompareElementAlgebra(context.Background(), *model)
		if err != nil {
			return err
		}
		fmt.Printf("Imported element-set algebra: %d exact Go/interpreter/oracle subset pairs.\n", pairs)
		if err := bridge.CompareNumericCounterexamples(context.Background(), *model); err != nil {
			return err
		}
		fmt.Println("Numeric counterexamples: source model agrees with native equality, element sets and managedfields.")
		if err := bridge.CompareNumericCodecCounterexamples(context.Background(), *model); err != nil {
			return err
		}
		fmt.Println("Numeric FieldsV1: round-trip, false-conflict, forced claim-loss and exclusive-omission witnesses agree with Go.")
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
