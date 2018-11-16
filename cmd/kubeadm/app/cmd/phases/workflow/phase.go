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

// Phase provides an implementation of a workflow phase that allows
// creation of new phases by simply instantiating a variable of this type.
type Phase struct {
	// name of the phase.
	// Phase name should be unique among peer phases (phases belonging to
	// the same workflow or phases belonging to the same parent phase).
	Name string

	// Aliases returns the aliases for the phase.
	Aliases []string

	// Short description of the phase.
	Short string

	// Long returns the long description of the phase.
	Long string

	// Example returns the example for the phase.
	Example string

	// Hidden define if the phase should be hidden in the workflow help.
	// e.g. PrintFilesIfDryRunning phase in the kubeadm init workflow is candidate for being hidden to the users
	Hidden bool

	// Phases defines a nested, ordered sequence of phases.
	Phases []Phase

	// Run defines a function implementing the phase action.
	// It is recommended to implent type assertion, e.g. using golang type switch,
	// for validating the RunData type.
	Run func(data RunData) error

	// RunIf define a function that implements a condition that should be checked
	// before executing the phase action.
	// If this function return nil, the phase action is always executed.
	RunIf func(data RunData) (bool, error)

	// CmdFlags defines the list of flags that should be assigned to the cobra command generated
	// for this phase; flags are inherited from the parent command or defined as additional flags
	// in the phase runner. If the values is not set or empty, no flags will be assigned to the command
	// Nb. global flags are automatically inherited by nested cobra command
	CmdFlags []string
}

// AppendPhase adds the given phase to the nested, ordered sequence of phases.
func (t *Phase) AppendPhase(phase Phase) {
	t.Phases = append(t.Phases, phase)
}
