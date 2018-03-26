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

/*
Package phases implements a support for PhasedCommand, that are customized
cobra.Command configured for supporting execution of atomic parts of the command logic.

Prerequisites

In order to implement PhasedCommand, the command logic must be divided into a set of
functions; each function should correspond to subset of the command logic, a phase,
that could be invoked atomically or as a part of the command workflow.

A recommended practice is to bind all phase functions to a single type that
will act as a execution context for the command by sharing its own attributes across
all the phases function.

	type myCommandContext struct {}

	func (o *myCommandContext) RunPhase1(cmd *cobra.Command, args []string) error {
	  ...
	}

	func (o *myCommandContext) RunPhase2(cmd *cobra.Command, args []string) error {
	  ...
	}

	...



Basics

If prerequisites are satisfied, then you can create a PhasedCommand using the
PhasedCommandBuilder.

A PhasedCommandBuilder gives you a semantic similar to the one used for creating
cobra.Command, with the addition of the attribute Phase.

The attribute Phase let you define the command logic as an ordered sequence of atomic phases.

    var context := myCommandContext{}

    var cmdBuilder = &cmdutil.PhasedCommandBuilder{
        Use:    "myCommand",
        Short:  "my awesome Phased command.",
        Long:   "...",
        Phases: []*.Phase{
            {
                Use:     "phase1",
                Short:   "my awesome phase 1",
                Run:     context.RunPhase1
            },
            {
                Use:     "phase2",
                Short:   "my awesome phase 2",
                Run:     context.RunPhase2
            },
            ...
        },
    }

Once the PhasedCommandBuilder is initialized, the cobra.Command can be generated invoking
the Build method.

    cmd, err := cmdBuilder.Build(myReceiver)

The generated cobra.Command can be completed as usual e.g. adding flags; At run time,
the PhasedCommand will show the following usage.

    $ myCommand -h

    my awesome Phased command.

    This command support execution of selected steps of the overall command logic by specifying [phases]:
      * phase1   my awesome phase 1
      * phase2   my awesome phase 2
      * ...

    Usage:
      myCommand [phases] [flags]

And it supports following types of invocation:

	# exectue all phases
	myCommand

	# exectue only on phase
	myCommand phase1

	# exectue more than one phase
	myCommand phase1 phase2

Phases definition

The Phase attribute of a PhasedCommand allows to define the command logic as an ordered
sequence of atomic phases.
In case of complex workflows, each phase can be split into nested workflows recursively.

Similarly to cobra.Command each phase has a one-line usage message, a short description,
and optionally one or more Phase.Aliases.

A phase can be assigned a Run function that implements the phase logic.

Eventually, it is possible to define a function that evaluates if the phase logic should
be executed inside a workflow or not (see Phased command execution below).

Finally, phases can be marked as Hidden; hidden phases are not shown in the command help
and cannot be invoked directly (see next paragraph for more detail).

Phase identifiers

Each not Hidden phase can be invoked atomically by indicating the phase identifier on the command line
when executing the command; valid identifiers are the Phase.name, that is the first word in
the use line, and also all Phase.Aliases.

	myCommand myPhase
	myCommand myPhaseAlias1

To invoke nested phases directly, you should consider that identifiers for nested phases
are computed by concatenating parent identifiers and nested phase identifiers.

	myCommand myPhase/myNestedPhase1

Args validation

A PhasedCommand when invoked automatically splits phaseArgs (args used for invoking phases) and other
custom/positional args.

The user can configure validation for custom/positional args in the same way that cobra.Command.

Additionally it is possible to set also additional phaseArgValidation e.g. enforcing
that only one phase can be invoked for each command execution.

Phased command execution

When the PhasedCommand is executed without specifying a [phases] argument, the command workflow
defined by the complete ordered sequence of Phases is executed.

When the PhasedCommand is executed specifying one [phases] argument, only the selected phases
are executed.

For each phase invoked directly:

- if defined, the Run function is executed.

- if defined, the nested workflow is executed.

When a Phase workflow is executed (either command workflow or a nested workflow), the entire
ordered sequence of Phases is executed, Hidden phases included.

For each phase in the workflow:

- if defined, the WorkflowIf function is executed; in case it returns false, the next two
actions are skipped.

- if defined; the Run function is executed.

- if defined, the nested workflow is executed (see Phase workflow execution).

NB. Hidden phases cannot be invoked directly.

NB. When a phase is invoked directly, the Run function and it's nested workflow are always
executed.

NB. When a phase is invoked in the context of a workflow (either command workflow or a nested workflow),
the Run function and it's nested workflow are executed only when the WorkflowIf
function is not defined or if is WorkflowIf defined and returns true.

*/
package phases
