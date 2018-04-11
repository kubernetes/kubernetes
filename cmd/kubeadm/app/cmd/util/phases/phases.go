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

package phases

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

// PhaseWorkflow represents an order sequence of phases.
type PhaseWorkflow []*Phase

// HasByArgOrAlias returns true if the given phase (identified by phase.Arg)
// is included in the workflow
func (w PhaseWorkflow) HasByArgOrAlias(arg string) bool {
	for _, p := range w {
		found := false
		p.visitAll(func(p *Phase) {
			if p.HasArgOrAlias(arg) {
				found = true
			}
		})
		if found {
			return true
		}
	}

	return false
}

// Phase defines an atomic part of a Phased command logic.
// It is required you to define the usage and short description to ensure usability.
type Phase struct {
	// Use is the one-line usage message.
	// The first word in the use line will be used as Phase.Name, and the main phase
	// identifier - that is Phase.Arg - will be derived from name.
	Use string

	// Aliases is an array of term that will be to compute Phase.ArgAliases, that is
	// a list of alternative phase identifiers.
	Aliases []string

	// Short is the short description shown in the 'help' output for this phase.
	Short string

	// Hidden define if a phase should be hidden from the final user; hidden phases
	// are not shown in the command help and can not be invoked directly by the user.
	// In turn hidden phases are executed when their parent phase (or the command) is invoked.
	Hidden bool

	// Run holds the function that implements the phase logic.
	// In case a nested workflow is defined using the Phase.Phases attribute, definition
	// of the Run function is optional.
	//
	// If both Run function and nested workflow are defined, Run is executed before the
	// nested workflows
	Run func(cmd *cobra.Command, args []string) error

	// WorkflowIf hold the function that determine if the phase logic should be executed
	// inside a workflow. If this function is not defined, the phase logic is always executed.
	//
	// Please note that when a phase is invoked directly, it is always executed.
	WorkflowIf func(cmd *cobra.Command, args []string) (bool, error)

	// Phases allows to define the nested workflows.
	Phases PhaseWorkflow

	// Internal fields.
	// Following fields are computed and used during the PhasedCommand build process.

	// readyToRun is a status flag that return true if the phase has been built
	// please note that some phase methods will panic if invoked when the phase is
	// yet not ready.
	readyToRun bool

	// level of nesting of this Phase.
	level int

	// name is the first word in the use line
	name string

	// arg is the  command line argument that should be used for invoking this phase.
	arg string

	// argAliases are alternative command line argument that could be used for
	// invoking the phase.
	argAliases []string

	// useLine is the representation if the phase identifiers to be printed in the command help.
	useLine string
}

// Build method prepare and validates the internal state of the phase for execution.
func (p *Phase) Build() error {
	return p.build(nil)
}

// build the phase (and nested workflow recursively)
func (p *Phase) build(parent *Phase) error {
	// sets internal attributes (based on hierarchy)
	if err := p.setAttributes(parent); err != nil {
		return err
	}

	// validate the phaseTree for inconsistencies e.g. there are phases that cannot be executed
	if err := p.validate(parent); err != nil {
		return err
	}

	// computes arg and argAlias for child workflow
	for _, c := range p.Phases {
		if err := c.build(p); err != nil {
			return err
		}
	}

	p.readyToRun = true

	return nil
}

// panicIfNotReady avoid that public methods are called before the phase is build
func (p *Phase) panicIfNotReady() {
	if !p.readyToRun {
		panic("Invalid method call. Execute phase.Build before invoking this method.")
	}
}

// setAttributes sets derived phase attributes; most of them are influenced by the hierarchy of phases
func (p *Phase) setAttributes(parent *Phase) error {
	// TODO: fails if name is empty string or name is not valid RFC string
	// TODO: fails if alias is empty string, is more than one word, is not a valid RFC string
	// TODO: fails if short is empty string
	p.name = name(p.Use)

	// if this phase is root
	if parent == nil {
		// initial values are assigned
		p.level = 0
		p.arg = arg("", p.name)
		p.argAliases = append(p.argAliases, p.Aliases...)
	} else {
		// computes level starting from parent level
		p.level = parent.level + 1

		// computes the main arg, that is derived from root arg and phase.Name
		p.arg = arg(parent.arg, p.name)

		// computes aliases to the main arg
		for _, alias := range p.Aliases {
			p.argAliases = append(p.argAliases, arg(parent.arg, alias))
		}

		// computes additional arg and aliases derived from root aliases
		for _, parentAlias := range parent.Aliases {
			p.argAliases = append(p.argAliases, arg(parentAlias, p.name))

			for _, alias := range p.Aliases {
				p.argAliases = append(p.argAliases, arg(parentAlias, alias))
			}
		}
	}

	p.useLine = useLine(p.name, p.Aliases, p.level)

	return nil
}

// name returns the first word in the phase one-line usage message.
func name(use string) string {
	name := strings.ToLower(use)
	i := strings.Index(name, " ")
	if i >= 0 {
		name = name[:i]
	}
	return name
}

var nestedArgSeparator = "/"

// arg returns a the main phase identifier (that can be used as a command argument
// for invoking this phase)
func arg(parentArgOrAlias, nameOrAlias string) string {
	arg := nameOrAlias
	if parentArgOrAlias != "" {
		arg = fmt.Sprintf("%s%s%s", parentArgOrAlias, nestedArgSeparator, arg)
	}
	return strings.ToLower(arg)
}

// UseLine return the phase UseLine
func useLine(name string, aliases []string, level int) string {
	prefix := ""
	if level > 1 {
		prefix = nestedArgSeparator
	}
	nameList := append([]string{name}, aliases...)

	return fmt.Sprintf("%s%s", prefix, strings.Join(nameList, "|"))
}

// validate if the phase has a consistent configuration (prevents unexpected
// behaviors during phase.Execute)
func (p *Phase) validate(parent *Phase) error {
	// Leaf phases (phases without nested workflows) must have a Phase.Run function defined
	if len(p.Phases) == 0 && p.Run == nil {
		return fmt.Errorf("Invalid phase %s definition; it is required to assign a Phase.Run function to leaf phases", p.arg)
	}

	return nil
}

// visitAll provide a convenience method for executing a function on this phase and
// all the nested workflow recursively
func (p *Phase) visitAll(fn func(*Phase)) {
	fn(p)
	for _, c := range p.Phases {
		c.visitAll(fn)
	}
}

// Name is the first word in the phase one-line usage message.
func (p *Phase) Name() string {
	p.panicIfNotReady()
	return p.name
}

// Arg is the command line argument that should be used for invoking this phase.
// It is computed starting from Phase.Name concatenated to the parent phase.Arg.
func (p *Phase) Arg() string {
	p.panicIfNotReady()
	return p.arg
}

// ArgAliases are alternative command line argument that could be used for invoking the phase.
// ArgAliases are computed starting from Phase.Aliases concatenated to the parent phase.Arg
// (and eventually to the list of parent phase.ArgAliases if defined).
func (p *Phase) ArgAliases() []string {
	p.panicIfNotReady()
	return p.argAliases
}

// HasArgOrAlias returns true if the given arg is one of the Phase.Arg and Phase.ArgAliases
func (p *Phase) HasArgOrAlias(arg string) bool {
	p.panicIfNotReady()

	for _, a := range append(p.argAliases, p.arg) {
		if a == arg {
			return true
		}
	}

	return false
}

// Level retruns the level of nesting of this Phase.
func (p *Phase) Level() int {
	p.panicIfNotReady()
	return p.level
}

// UseLine return the phase UseLine, that is a string composed by the Phase.Name and Phase.Aliases
// that can be used for invoking this phase
func (p *Phase) UseLine() string {
	p.panicIfNotReady()
	return p.useLine
}

// Execute the function that implement the phase logic and the nested workflow.
func (p *Phase) Execute(cmd *cobra.Command, args []string) (err error) {
	p.panicIfNotReady()

	// defers a function for recovering from panic during phase execution
	defer func() {
		if r := recover(); r != nil {
			switch x := r.(type) {
			case string:
				err = fmt.Errorf("Error executing phase %s: %s", p.arg, x)
			case error:
				err = fmt.Errorf("Error executing phase %s: %v", p.arg, x)
			default:
				err = fmt.Errorf("Error executing phase %s: unknown panic", p.arg)
			}
		}
	}()

	// if the phase has a run function, executes it
	if p.Run != nil {
		if err = p.Run(cmd, args); err != nil {
			// in case of errors, the workflow stops
			return err
		}
	}

	// if the phase has nested workflow, executes it
	if len(p.Phases) > 0 {
		for _, c := range p.Phases {

			// if the WorkflowIf function is defined, check if this phase should be executed
			if c.WorkflowIf != nil {
				exec, err := c.WorkflowIf(cmd, args)
				if err != nil {
					// in case of errors, the workflow stops
					return fmt.Errorf("Error executing phase %s - WorkflowIf function: %v", c.arg, err)
				}
				// if the phase should not be executed, moves to next phase
				if !exec {
					continue
				}
			}

			// executes the phase
			if err := c.Execute(cmd, args); err != nil {
				// in case of errors, the workflow stops
				return err
			}
		}
	}

	return nil
}
