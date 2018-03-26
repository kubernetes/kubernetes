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

// PhasedCommandBuilder allows to build customized cobra.Command configured
// for supporting execution of atomic parts of the command logic.
// The PhasedCommandBuilder provides a semantic similar to simple cobra.Command, but
// with the addition of the attributes for defining Phases and their behavior.
type PhasedCommandBuilder struct {
	// Use is the one-line usage message.
	// The usage message will be automatically extended with the addition of the
	// [phases] argument (If not already present).
	Use string

	// Aliases is an array of aliases that can be used instead of the first word
	// in Use.
	Aliases []string

	// Short is the short description shown in the 'help' output.
	Short string

	// Long is the long message shown in the 'help <this-command>' output.
	// This message will be automatically modified with the addition of the list of
	// defined phases (with the exception of Hidden phases).
	Long string

	// Example is examples of how to use the command.
	Example string

	// ValidArgs is list of non-flag arguments that are accepted in bash completions
	// Please note that this list must not includes arguments used for invoking
	// phases, which are managed separately.
	ValidArgs []string

	// ArgsValidator, if provided, defines the function used to validate positional
	// and custom arguments.
	// Please note that this function does not validates arguments used for
	// invoking phases, which are validated separately.
	ArgsValidator cobra.PositionalArgs

	// PhaseArgsValidator, if provided, defines the function used to validate arguments
	// used for invoking phases.
	// Please note that function does not validates other custom/positional arguments.
	PhaseArgsValidator cobra.PositionalArgs

	// Phases defines the command logic as an ordered sequence of atomic phases.
	// In case of complex workflows, each phase can be split into nested workflows
	// recursively (see Phases for more details).
	//
	// When the phased command will be executed, the default behavior ensures that
	// the all the phases are executed in an ordered workflow. Otherwise phases
	// can be invoked directly/atomically by passing the [phases] argument to the command
	// line.
	Phases PhaseWorkflow

	// RunPhases, allows to override the default logic that executes the ordered
	// workflow of phases.
	RunPhases func(cmd *cobra.Command, phases PhaseWorkflow, args []string) error

	// Internal fields.
	// Following fields are computed and used during the PhasedCommand build process.

	// phasesTree defines the full tree of phases, that includes a "technical"
	// root phase representing the entire command (a workflow with all the phases)
	phasesTree *Phase

	// phasesMap provides a data structure for quickly access phases given their id
	// (or one of the alternatives ids/aliases).
	phasesMap map[string]*Phase
}

// MustBuild is an helper function that wraps a call to Build method and panics
// if it returns nil.
// MustBuild is intended for use when you want to ensure that the PhasedCommand
// configuration doesn't contains any formal error.
func (pc *PhasedCommandBuilder) MustBuild() *cobra.Command {
	cmd, err := pc.Build()
	if err != nil {
		panic(fmt.Sprintf("Invalid phases definition: %v", err))
	}

	return cmd
}

// Build method create the customized cobra.Command configured
// for supporting execution of atomic parts of the command logic.
//
// See the package documentation for additional information.
//
// Important: in order to not compromise the PhasedCommand behavior, following
// attributes of the generated command should not be changed after build:
// Use, Aliases, Short, Long, Example, Args, ValidArgs, RunE.
func (pc *PhasedCommandBuilder) Build() (*cobra.Command, error) {

	// sets the phasesTree the phasesMap internal attributes
	if err := pc.setPhaseTreeAndMap(); err != nil {
		return nil, err
	}

	// If a custom RunPhases function is not provided by the user, use the DefaultPhaseExecutor
	if pc.RunPhases == nil {
		pc.RunPhases = DefaultPhaseExecutor
	}

	cmd := &cobra.Command{
		Use:       pc.useLine(), // adds [phases] to usage
		Aliases:   pc.Aliases,
		Short:     pc.Short,
		Long:      pc.descriptionWithPhases(), // adds phases list to the long description
		Example:   pc.Example,
		Args:      pc.validateArgs(pc.PhaseArgsValidator, pc.ArgsValidator), // triggers separated validation for for phaseArgs and other custom/positional args.
		ValidArgs: pc.ValidArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			// filter phaseArgs (arguments specifying which phases to invoke) from the command args
			phaseArgs, otherArgs := pc.splitArgs(args)

			// if user is requesting to run only some phases
			if len(phaseArgs) > 0 {
				// retrives selected phases from the phase map
				// TODO: retrive phases according to workflow sequence (visitAll)
				var phasesToRun []*Phase
				for _, p := range phaseArgs {
					phasesToRun = append(phasesToRun, pc.phasesMap[p])
				}
				// executes only selected phases
				return pc.RunPhases(cmd, phasesToRun, otherArgs)
			}

			// otherwise executes the entire command workflow
			return pc.RunPhases(cmd, PhaseWorkflow{pc.phasesTree}, otherArgs)
		},
	}

	return cmd, nil
}

// setPhaseTreeAndMap sets the phasesTree the phasesMap.
// If duplicate ars and/or argAliases are detected, the function returns an error.
// If less than two phases are defined, the functions returns an error
func (pc *PhasedCommandBuilder) setPhaseTreeAndMap() error {

	// creates a "technical" root phase representing the entire command (a workflow with all the phases)
	phasesTree := &Phase{
		Phases: pc.Phases, // Sets the command workflow, that is composed by all the phases provided by the user
		Hidden: true,      // Avoids that the root phase is invoked directly
	}

	// Builds the phase tree > prepare and validates the internal state of the phase tree for execution.
	if err := phasesTree.Build(); err != nil {
		return err
	}

	// Creates the phase map (an alternative way for accessing phases)
	// and validates for duplicated phase Arg or ArgAliases
	phasesMap := make(map[string]*Phase)

	duplicatedArgs := []string{}

	phasesTree.visitAll(func(p *Phase) {
		if !p.Hidden {
			args := append(p.argAliases, p.arg)

			for _, a := range args {
				if _, ok := phasesMap[a]; ok == true {
					duplicatedArgs = append(duplicatedArgs, a)
				}
				phasesMap[a] = p
			}
		}
	})

	if len(duplicatedArgs) > 0 {
		return fmt.Errorf("Invalid phases definition. Following phase Arg or ArgAliases are used by more than one phase [%s]", strings.Join(duplicatedArgs, ", "))
	}

	pc.phasesTree = phasesTree
	pc.phasesMap = phasesMap

	return nil
}

// useLine returns the use line for this command adding [phases] if necessary
func (pc *PhasedCommandBuilder) useLine() string {
	if !strings.Contains(pc.Use, "[phases]") {
		return fmt.Sprintf("%s [phases]", pc.Use)
	}
	return pc.Use
}

// descriptionWithPhases returns the PhasedCommand long description with the addition of the
// message for using phases
func (pc *PhasedCommandBuilder) descriptionWithPhases() string {
	// computes the max length of the use line for each phase (arg + argAliases)
	maxLength := 0
	pc.phasesTree.visitAll(func(p *Phase) {
		if !p.Hidden {
			length := len(p.UseLine())
			if maxLength < length {
				maxLength = length
			}
		}
	})

	// prints the command description using the Long/Short description
	// followed by the list of phases indented by level and formatted using the maxlength
	line := pc.Long
	if line == "" {
		line = pc.Short
	}

	line += fmt.Sprintf("\n\nThe %q command executes the following internal workflow:\n", pc.Use)

	// prints phase list enclosed in a mardown code block for ensuring better readability in the help online
	line += "```\n"
	offset := 2
	pc.phasesTree.visitAll(func(p *Phase) {
		if !p.Hidden {
			useLine := p.UseLine()

			line += strings.Repeat(" ", offset*p.level)                // indentation
			line += useLine                                            // phase useLine
			line += strings.Repeat(" ", maxLength-len(useLine)+offset) // padding right up to max length (+ offset for spacing)
			line += p.Short                                            // phase short description
			line += "\n"
		}
	})
	line += "```\n\n"

	line += fmt.Sprintf("You can execute the entire workflow of the %q or execute only \n", pc.Use)
	line += "single steps of the command logic by specifying the [phases] argument.\n"

	return line
}

// splitArgs divides command line args into phaseArgs (arguments specifying which phases to invoke)
// and other custom/positional args.
func (pc *PhasedCommandBuilder) splitArgs(args []string) (phaseArgs []string, otherArgs []string) {
	phaseArgs = []string{}
	otherArgs = []string{}

	for _, a := range args {
		if _, ok := pc.phasesMap[a]; ok == true {
			phaseArgs = append(phaseArgs, a)
			continue
		}

		otherArgs = append(otherArgs, a)
	}

	return
}

// validateArgs implements a wrapper validation function that split command line args and
// triggers dedicated validation function for phaseArgs (arguments specifying which phases to invoke)
// and other custom/positional args.
func (pc *PhasedCommandBuilder) validateArgs(phaseArgValidator cobra.PositionalArgs, customArgValidator cobra.PositionalArgs) cobra.PositionalArgs {
	return func(cmd *cobra.Command, args []string) error {
		phaseArgs, customArgs := pc.splitArgs(args)

		if customArgValidator != nil {
			if err := customArgValidator(cmd, customArgs); err != nil {
				return fmt.Errorf("invalid args: %v", err)
			}
		}

		if phaseArgValidator != nil {
			if err := phaseArgValidator(cmd, phaseArgs); err != nil {
				return fmt.Errorf("invalid phases args: %v", err)
			}
		}

		return nil
	}
}

// DefaultPhaseExecutor is the default implementation of a function that executes
// phases are executed in an ordered workflow.
//
// In case any phase includes a nested workflows, this is executed as well (and so on recursively).
func DefaultPhaseExecutor(cmd *cobra.Command, phasesToRun PhaseWorkflow, args []string) error {
	for _, p := range phasesToRun {
		if err := p.Execute(cmd, args); err != nil {
			return err
		}
	}
	return nil
}
