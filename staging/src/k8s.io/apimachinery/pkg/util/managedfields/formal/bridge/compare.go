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

package bridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os/exec"
	"time"

	"github.com/google/go-cmp/cmp"
)

type Summary struct {
	Cases       int
	Steps       int
	Differences int
	ModelGaps   int
	Accepted    int
	Conflicts   int
}

func equalSnapshot(want, got Snapshot) error {
	if diff := cmp.Diff(want, got); diff != "" {
		return fmt.Errorf("observation mismatch (-expected +actual):\n%s", diff)
	}
	return nil
}

// Compare checks both engines against independent expectations and against each
// other. A documented proposal difference must occur and match both exact oracles.
func Compare(ctx context.Context, model string, cases []Case) (Summary, error) {
	summary := Summary{Cases: len(cases)}
	fixtures := make([]fixture, 0, len(cases))
	for _, c := range cases {
		f, err := normalize(c)
		if err != nil {
			return summary, fmt.Errorf("%s: %w", c.Name, err)
		}
		fixtures = append(fixtures, f)
	}
	input, err := json.Marshal(fixtures)
	if err != nil {
		return summary, err
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, model, "--replay", "-")
	command.Stdin = bytes.NewReader(input)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		return summary, fmt.Errorf("Lean replay failed: %w\n%s", err, stderr.String())
	}
	var reports []struct {
		Name         string `json:"name"`
		Observations []struct {
			Error *string `json:"error"`
			State state   `json:"state"`
		} `json:"observations"`
	}
	if err := json.Unmarshal(output, &reports); err != nil {
		return summary, fmt.Errorf("invalid Lean replay output: %w", err)
	}
	if len(reports) != len(cases) || len(cases) == 0 {
		return summary, fmt.Errorf("Lean returned %d cases for %d inputs", len(reports), len(cases))
	}
	for index, c := range cases {
		report := reports[index]
		if report.Name != c.Name || len(report.Observations) != len(c.Steps) {
			return summary, fmt.Errorf("Lean replay omitted or reordered observations for %s", c.Name)
		}
		engine, err := newGoEngine()
		if err != nil {
			return summary, err
		}
		for i, step := range c.Steps {
			observation := report.Observations[i]
			errorName := ""
			if observation.Error != nil {
				errorName = *observation.Error
			}
			lean, err := modelSnapshot(observation.State, errorName)
			if err != nil {
				return summary, err
			}
			actual, err := engine.step(step)
			if err != nil {
				return summary, err
			}
			if err := compareStep(step, lean, actual); err != nil {
				// The first failing prefix is enough to replay the discrepancy.
				prefix, _ := json.Marshal(Corpus{Version: 1, Cases: []Case{{
					Name: c.Name, Required: c.Required, Steps: c.Steps[:i+1],
				}}})
				return summary, fmt.Errorf("%s step %d: %w\nreproducer: %s", c.Name, i+1, err, prefix)
			}
			summary.Steps++
			if step.Difference != "" {
				summary.Differences++
				if step.DifferenceKind == "model-gap" {
					summary.ModelGaps++
				}
			}
			if actual.Error == "" {
				summary.Accepted++
			} else if actual.Error == "conflict" {
				summary.Conflicts++
			}
		}
	}
	return summary, nil
}

func compareStep(step Step, lean, goResult Snapshot) error {
	if step.Difference != "" && (step.Expect == nil || step.GoExpect == nil) {
		return fmt.Errorf("a proposal difference needs both independent expected results")
	}
	if step.Expect != nil {
		want, err := expectedSnapshot(*step.Expect)
		if err != nil {
			return err
		}
		if err := equalSnapshot(want, lean); err != nil {
			return fmt.Errorf("Lean oracle: %w", err)
		}
		goExpected := step.Expect
		if step.GoExpect != nil {
			goExpected = step.GoExpect
		}
		wantGo, err := expectedSnapshot(*goExpected)
		if err != nil {
			return err
		}
		if err := equalSnapshot(wantGo, goResult); err != nil {
			return fmt.Errorf("Go oracle: %w", err)
		}
	}
	err := equalSnapshot(lean, goResult)
	if step.Difference == "" {
		return err
	}
	if err == nil {
		return fmt.Errorf("documented difference no longer occurs: %s; update the corpus", step.Difference)
	}
	return nil
}

// GeneratedScalarCases explores the common scalar domain independently of
// either transition. The seed and first failing prefix make failures reproducible.
func GeneratedScalarCases(seed int64, count, steps int) []Case {
	random := rand.New(rand.NewSource(seed))
	cases := make([]Case, 0, count)
	for i := 0; i < count; i++ {
		c := Case{Name: fmt.Sprintf("generated-seed-%d-case-%d", seed, i)}
		for j := 0; j < steps; j++ {
			object := map[string]any{}
			for _, field := range []string{"x", "y", "z"} {
				if random.Intn(2) == 0 {
					object[field] = fmt.Sprintf("value-%d", random.Intn(3))
				}
			}
			c.Steps = append(c.Steps, Step{
				Manager: []string{"A", "B", "C"}[random.Intn(3)],
				Force:   random.Intn(4) == 0, Apply: object,
			})
		}
		cases = append(cases, c)
	}
	return cases
}
