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

package workflow

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// phaseSeparator defines the separator to be used when concatenating nested
// phase names
const phaseSeparator = "/"

// RunnerOptions defines the options supported during the execution of a
// kubeadm composable workflows
type RunnerOptions struct {
	// FilterPhases defines the list of phases to be executed (if empty, all).
	FilterPhases []string

	// SkipPhases defines the list of phases to be excluded by execution (if empty, none).
	SkipPhases []string
}

// RunData defines the data shared among all the phases included in the workflow, that is any type.
type RunData = interface{}

// Runner implements management of composable kubeadm workflows.
type Runner struct {
	// Options that regulate the runner behavior.
	Options RunnerOptions

	// Phases composing the workflow to be managed by the runner.
	Phases []Phase

	// runDataInitializer defines a function that creates the runtime data shared
	// among all the phases included in the workflow
	runDataInitializer func(*cobra.Command, []string) (RunData, error)

	// runData is part of the internal state of the runner and it is used for implementing
	// a singleton in the InitData methods (thus avoiding to initialize data
	// more than one time)
	runData RunData

	// runCmd is part of the internal state of the runner and it is used to track the
	// command that will trigger the runner (only if the runner is BindToCommand).
	runCmd *cobra.Command

	// cmdAdditionalFlags holds additional, shared flags that could be added to the subcommands generated
	// for phases. Flags could be inherited from the parent command too or added directly to each phase
	cmdAdditionalFlags *pflag.FlagSet

	// phaseRunners is part of the internal state of the runner and provides
	// a list of wrappers to phases composing the workflow with contextual
	// information supporting phase execution.
	phaseRunners []*phaseRunner
}

// phaseRunner provides a wrapper to a Phase with the addition of a set
// of contextual information derived by the workflow managed by the Runner.
// TODO: If we ever decide to get more sophisticated we can swap this type with a well defined dag or tree library.
type phaseRunner struct {
	// Phase provide access to the phase implementation
	Phase

	// provide access to the parent phase in the workflow managed by the Runner.
	parent *phaseRunner

	// level define the level of nesting of this phase into the workflow managed by
	// the Runner.
	level int

	// selfPath contains all the elements of the path that identify the phase into
	// the workflow managed by the Runner.
	selfPath []string

	// generatedName is the full name of the phase, that corresponds to the absolute
	// path of the phase in the workflow managed by the Runner.
	generatedName string

	// use is the phase usage string that will be printed in the workflow help.
	// It corresponds to the relative path of the phase in the workflow managed by the Runner.
	use string
}

// NewRunner return a new runner for composable kubeadm workflows.
func NewRunner() *Runner {
	return &Runner{
		Phases: []Phase{},
	}
}

// AppendPhase adds the given phase to the ordered sequence of phases managed by the runner.
func (e *Runner) AppendPhase(t Phase) {
	e.Phases = append(e.Phases, t)
}

// computePhaseRunFlags return a map defining which phase should be run and which not.
// PhaseRunFlags are computed according to RunnerOptions.
func (e *Runner) computePhaseRunFlags() (map[string]bool, error) {
	// Initialize support data structure
	phaseRunFlags := map[string]bool{}
	phaseHierarchy := map[string][]string{}
	e.visitAll(func(p *phaseRunner) error {
		// Initialize phaseRunFlags assuming that all the phases should be run.
		phaseRunFlags[p.generatedName] = true

		// Initialize phaseHierarchy for the current phase (the list of phases
		// depending on the current phase
		phaseHierarchy[p.generatedName] = []string{}

		// Register current phase as part of its own parent hierarchy
		parent := p.parent
		for parent != nil {
			phaseHierarchy[parent.generatedName] = append(phaseHierarchy[parent.generatedName], p.generatedName)
			parent = parent.parent
		}
		return nil
	})

	// If a filter option is specified, set all phaseRunFlags to false except for
	// the phases included in the filter and their hierarchy of nested phases.
	if len(e.Options.FilterPhases) > 0 {
		for i := range phaseRunFlags {
			phaseRunFlags[i] = false
		}
		for _, f := range e.Options.FilterPhases {
			if _, ok := phaseRunFlags[f]; !ok {
				return phaseRunFlags, errors.Errorf("invalid phase name: %s", f)
			}
			phaseRunFlags[f] = true
			for _, c := range phaseHierarchy[f] {
				phaseRunFlags[c] = true
			}
		}
	}

	// If a phase skip option is specified, set the corresponding phaseRunFlags
	// to false and apply the same change to the underlying hierarchy
	for _, f := range e.Options.SkipPhases {
		if _, ok := phaseRunFlags[f]; !ok {
			return phaseRunFlags, errors.Errorf("invalid phase name: %s", f)
		}
		phaseRunFlags[f] = false
		for _, c := range phaseHierarchy[f] {
			phaseRunFlags[c] = false
		}
	}

	return phaseRunFlags, nil
}

// SetDataInitializer allows to setup a function that initialize the runtime data shared
// among all the phases included in the workflow.
// The method will receive in input the cmd that triggers the Runner (only if the runner is BindToCommand)
func (e *Runner) SetDataInitializer(builder func(cmd *cobra.Command, args []string) (RunData, error)) {
	e.runDataInitializer = builder
}

// InitData triggers the creation of runtime data shared among all the phases included in the workflow.
// This action can be executed explicitly out, when it is necessary to get the RunData
// before actually executing Run, or implicitly when invoking Run.
func (e *Runner) InitData(args []string) (RunData, error) {
	if e.runData == nil && e.runDataInitializer != nil {
		var err error
		if e.runData, err = e.runDataInitializer(e.runCmd, args); err != nil {
			return nil, err
		}
	}

	return e.runData, nil
}

// Run the kubeadm composable kubeadm workflows.
func (e *Runner) Run(args []string) error {
	e.prepareForExecution()

	// determine which phase should be run according to RunnerOptions
	phaseRunFlags, err := e.computePhaseRunFlags()
	if err != nil {
		return err
	}

	// precheck phase dependencies before actual execution
	missedDeps := make(map[string][]string)
	visited := make(map[string]struct{})
	for _, p := range e.phaseRunners {
		if run, ok := phaseRunFlags[p.generatedName]; !run || !ok {
			continue
		}
		for _, dep := range p.Phase.Dependencies {
			if _, ok := visited[dep]; !ok {
				missedDeps[p.Phase.Name] = append(missedDeps[p.Phase.Name], dep)
			}
		}
		visited[p.Phase.Name] = struct{}{}
	}
	if len(missedDeps) > 0 {
		var msg strings.Builder
		msg.WriteString("unresolved dependencies:")
		for phase, missedPhases := range missedDeps {
			msg.WriteString(fmt.Sprintf("\n\tmissing %v phase(s) needed by %q phase", missedPhases, phase))
		}
		return errors.New(msg.String())
	}

	// builds the runner data
	var data RunData
	if data, err = e.InitData(args); err != nil {
		return err
	}

	err = e.visitAll(func(p *phaseRunner) error {
		// if the phase should not be run, skip the phase.
		if run, ok := phaseRunFlags[p.generatedName]; !run || !ok {
			return nil
		}

		// Errors if phases that are meant to create special subcommands only
		// are wrongly assigned Run Methods
		if p.RunAllSiblings && (p.RunIf != nil || p.Run != nil) {
			return errors.Errorf("phase marked as RunAllSiblings can not have Run functions %s", p.generatedName)
		}

		// If the phase defines a condition to be checked before executing the phase action.
		if p.RunIf != nil {
			// Check the condition and returns if the condition isn't satisfied (or fails)
			ok, err := p.RunIf(data)
			if err != nil {
				return errors.Wrapf(err, "error execution run condition for phase %s", p.generatedName)
			}

			if !ok {
				return nil
			}
		}

		// Runs the phase action (if defined)
		if p.Run != nil {
			if err := p.Run(data); err != nil {
				return errors.Wrapf(err, "error execution phase %s", p.generatedName)
			}
		}

		return nil
	})

	return err
}

// Help returns text with the list of phases included in the workflow.
func (e *Runner) Help(cmdUse string) string {
	e.prepareForExecution()

	// computes the max length of for each phase use line
	maxLength := 0
	e.visitAll(func(p *phaseRunner) error {
		if !p.Hidden && !p.RunAllSiblings {
			length := len(p.use)
			if maxLength < length {
				maxLength = length
			}
		}
		return nil
	})

	// prints the list of phases indented by level and formatted using the maxlength
	// the list is enclosed in a mardown code block for ensuring better readability in the public web site
	line := fmt.Sprintf("The %q command executes the following phases:\n", cmdUse)
	line += "```\n"
	offset := 2
	e.visitAll(func(p *phaseRunner) error {
		if !p.Hidden && !p.RunAllSiblings {
			padding := maxLength - len(p.use) + offset
			line += strings.Repeat(" ", offset*p.level) // indentation
			line += p.use                               // name + aliases
			line += strings.Repeat(" ", padding)        // padding right up to max length (+ offset for spacing)
			line += p.Short                             // phase short description
			line += "\n"
		}

		return nil
	})
	line += "```"
	return line
}

// SetAdditionalFlags allows to define flags to be added
// to the subcommands generated for each phase (but not existing in the parent command).
// Please note that this command needs to be done before BindToCommand.
// Nb. if a flag is used only by one phase, please consider using phase LocalFlags.
func (e *Runner) SetAdditionalFlags(fn func(*pflag.FlagSet)) {
	// creates a new NewFlagSet
	e.cmdAdditionalFlags = pflag.NewFlagSet("phaseAdditionalFlags", pflag.ContinueOnError)
	// invokes the function that sets additional flags
	fn(e.cmdAdditionalFlags)
}

// BindToCommand bind the Runner to a cobra command by altering
// command help, adding phase related flags and by adding phases subcommands
// Please note that this command needs to be done once all the phases are added to the Runner.
func (e *Runner) BindToCommand(cmd *cobra.Command) {
	// keep track of the command triggering the runner
	e.runCmd = cmd

	// return early if no phases were added
	if len(e.Phases) == 0 {
		return
	}

	e.prepareForExecution()

	// adds the phases subcommand
	phaseCommand := &cobra.Command{
		Use:   "phase",
		Short: fmt.Sprintf("Use this command to invoke single phase of the %s workflow", cmd.Name()),
	}

	cmd.AddCommand(phaseCommand)

	// generate all the nested subcommands for invoking single phases
	subcommands := map[string]*cobra.Command{}
	e.visitAll(func(p *phaseRunner) error {
		// skip hidden phases
		if p.Hidden {
			return nil
		}

		// initialize phase selector
		phaseSelector := p.generatedName

		// if requested, set the phase to run all the sibling phases
		if p.RunAllSiblings {
			phaseSelector = p.parent.generatedName
		}

		// creates phase subcommand
		phaseCmd := &cobra.Command{
			Use:     strings.ToLower(p.Name),
			Short:   p.Short,
			Long:    p.Long,
			Example: p.Example,
			Aliases: p.Aliases,
			RunE: func(cmd *cobra.Command, args []string) error {
				// if the phase has subphases, print the help and exits
				if len(p.Phases) > 0 {
					return cmd.Help()
				}

				// overrides the command triggering the Runner using the phaseCmd
				e.runCmd = cmd
				e.Options.FilterPhases = []string{phaseSelector}
				return e.Run(args)
			},
		}

		// makes the new command inherits local flags from the parent command
		// Nb. global flags will be inherited automatically
		inheritsFlags(cmd.Flags(), phaseCmd.Flags(), p.InheritFlags)

		// makes the new command inherits additional flags for phases
		if e.cmdAdditionalFlags != nil {
			inheritsFlags(e.cmdAdditionalFlags, phaseCmd.Flags(), p.InheritFlags)
		}

		// If defined, added phase local flags
		if p.LocalFlags != nil {
			p.LocalFlags.VisitAll(func(f *pflag.Flag) {
				phaseCmd.Flags().AddFlag(f)
			})
		}

		// if this phase has children (not a leaf) it doesn't accept any args
		if len(p.Phases) > 0 {
			phaseCmd.Args = cobra.NoArgs
		} else {
			if p.ArgsValidator == nil {
				phaseCmd.Args = cmd.Args
			} else {
				phaseCmd.Args = p.ArgsValidator
			}
		}

		// adds the command to parent
		if p.level == 0 {
			phaseCommand.AddCommand(phaseCmd)
		} else {
			subcommands[p.parent.generatedName].AddCommand(phaseCmd)
		}

		subcommands[p.generatedName] = phaseCmd
		return nil
	})

	// alters the command description to show available phases
	if cmd.Long != "" {
		cmd.Long = fmt.Sprintf("%s\n\n%s\n", cmd.Long, e.Help(cmd.Use))
	} else {
		cmd.Long = fmt.Sprintf("%s\n\n%s\n", cmd.Short, e.Help(cmd.Use))
	}

	// adds phase related flags to the main command
	cmd.Flags().StringSliceVar(&e.Options.SkipPhases, "skip-phases", nil, "List of phases to be skipped")
}

func inheritsFlags(sourceFlags, targetFlags *pflag.FlagSet, cmdFlags []string) {
	// If the list of flag to be inherited from the parent command is not defined, no flag is added
	if cmdFlags == nil {
		return
	}

	// add all the flags to be inherited to the target flagSet
	sourceFlags.VisitAll(func(f *pflag.Flag) {
		for _, c := range cmdFlags {
			if f.Name == c {
				targetFlags.AddFlag(f)
			}
		}
	})
}

// visitAll provides a utility method for visiting all the phases in the workflow
// in the execution order and executing a func on each phase.
// Nested phase are visited immediately after their parent phase.
func (e *Runner) visitAll(fn func(*phaseRunner) error) error {
	for _, currentRunner := range e.phaseRunners {
		if err := fn(currentRunner); err != nil {
			return err
		}
	}
	return nil
}

// prepareForExecution initialize the internal state of the Runner (the list of phaseRunner).
func (e *Runner) prepareForExecution() {
	e.phaseRunners = []*phaseRunner{}
	var parentRunner *phaseRunner
	for _, phase := range e.Phases {
		// skips phases that are meant to create special subcommands only
		if phase.RunAllSiblings {
			continue
		}

		// add phases to the execution list
		addPhaseRunner(e, parentRunner, phase)
	}
}

// addPhaseRunner adds the phaseRunner for a given phase to the phaseRunners list
func addPhaseRunner(e *Runner, parentRunner *phaseRunner, phase Phase) {
	// computes contextual information derived by the workflow managed by the Runner.
	use := cleanName(phase.Name)
	generatedName := use
	selfPath := []string{generatedName}

	if parentRunner != nil {
		generatedName = strings.Join([]string{parentRunner.generatedName, generatedName}, phaseSeparator)
		use = fmt.Sprintf("%s%s", phaseSeparator, use)
		selfPath = append(parentRunner.selfPath, selfPath...)
	}

	// creates the phaseRunner
	currentRunner := &phaseRunner{
		Phase:         phase,
		parent:        parentRunner,
		level:         len(selfPath) - 1,
		selfPath:      selfPath,
		generatedName: generatedName,
		use:           use,
	}

	// adds to the phaseRunners list
	e.phaseRunners = append(e.phaseRunners, currentRunner)

	// iterate for the nested, ordered list of phases, thus storing
	// phases in the expected executing order (child phase are stored immediately after their parent phase).
	for _, childPhase := range phase.Phases {
		addPhaseRunner(e, currentRunner, childPhase)
	}
}

// cleanName makes phase name suitable for the runner help, by lowercasing the name
// and removing args descriptors, if any
func cleanName(name string) string {
	ret := strings.ToLower(name)
	if pos := strings.Index(ret, " "); pos != -1 {
		ret = ret[:pos]
	}
	return ret
}
