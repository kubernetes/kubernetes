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

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
)

var myWorkflowRunner = NewRunner()

type myWorkflowData struct {
	data string
}

func (c *myWorkflowData) Data() string {
	return c.data
}

type myPhaseData interface {
	Data() string
}

func ExamplePhase() {
	// Create a phase
	var myPhase1 = Phase{
		Name:  "myPhase1",
		Short: "A phase of a kubeadm composable workflow...",
		Run: func(data RunData) error {
			// transform data into a typed data struct
			d, ok := data.(myPhaseData)
			if !ok {
				return errors.New("invalid RunData type")
			}

			// implement your phase logic...
			fmt.Printf("%v\n", d.Data())
			return nil
		},
	}

	// Create another phase
	var myPhase2 = Phase{
		Name:  "myPhase2",
		Short: "Another phase of a kubeadm composable workflow...",
		Run: func(data RunData) error {
			// transform data into a typed data struct
			d, ok := data.(myPhaseData)
			if !ok {
				return errors.New("invalid RunData type")
			}

			// implement your phase logic...
			fmt.Printf("%v\n", d.Data())
			return nil
		},
	}

	// Adds the new phases to the workflow
	// Phases will be executed the same order they are added to the workflow
	myWorkflowRunner.AppendPhase(myPhase1)
	myWorkflowRunner.AppendPhase(myPhase2)
}

func ExampleRunner_Run() {
	// Create a phase
	var myPhase = Phase{
		Name:  "myPhase",
		Short: "A phase of a kubeadm composable workflow...",
		Run: func(data RunData) error {
			// transform data into a typed data struct
			d, ok := data.(myPhaseData)
			if !ok {
				return errors.New("invalid RunData type")
			}

			// implement your phase logic...
			fmt.Printf("%v\n", d.Data())
			return nil
		},
	}

	// Adds the new phase to the workflow
	var myWorkflowRunner = NewRunner()
	myWorkflowRunner.AppendPhase(myPhase)

	// Defines the method that creates the runtime data shared
	// among all the phases included in the workflow
	myWorkflowRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (RunData, error) {
		return myWorkflowData{data: "some data"}, nil
	})

	// Runs the workflow by passing a list of arguments
	myWorkflowRunner.Run([]string{})
}
