/*

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

package install

import (
	"fmt"
	"io"
	"os"
	"os/exec"

	"sigs.k8s.io/addon-operators/installer/pkg/apis/config"
)

type Runtime struct {
	Config *config.AddonInstallerConfiguration
	Stdout io.Writer
	Stderr io.Writer
	cmdSet map[*exec.Cmd]struct{}
}

func (r *Runtime) CheckDeps() error {
	if _, err := exec.LookPath("kubectl"); err != nil {
		return err
	}

	return nil
}

func (r *Runtime) CheckConfig() error {
	// check for duplicates
	var duplicates []string
	refCounts := map[string]int{}
	for _, ref := range r.Config.Addons {
		refCounts[ref]++
	}
	for ref, count := range refCounts {
		if count >= 2 {
			duplicates = append(duplicates, ref)
		}
	}
	if len(duplicates) > 0 {
		return fmt.Errorf("AddonInstallerConfiguration lists duplicate addons: %v", duplicates)
	}

	return nil
}

func (r *Runtime) InstallAddons() error {
	for _, ref := range r.Config.Addons {
		err := r.InstallSingleAddon(ref)
		if err != nil {
			return err
		}
		// Add some visual space since the caller delegated the list of addons to us
		fmt.Fprintln(r.Stdout)
	}
	return nil
}

func (r *Runtime) InstallSingleAddon(ref string) error {
	msg := "...installing " + ref
	args := []string{"apply", "-k", ref}
	if r.Config.DryRun {
		msg += " (dry run)"
		args = append(args, "--dry-run")
	}
	fmt.Fprintln(r.Stdout, msg)
	err := r.runCommand("kubectl", args...)
	if err != nil {
		return err
	}
	return nil
}

func (r *Runtime) runCommand(command string, args ...string) error {
	cmd := exec.Command(command, args...)
	cmd.Stdout = r.Stdout
	cmd.Stderr = r.Stderr

	if r.cmdSet == nil {
		r.cmdSet = make(map[*exec.Cmd]struct{})
	}
	// Add this cmd to the set for the duration of its runtime
	r.cmdSet[cmd] = struct{}{}
	defer delete(r.cmdSet, cmd)

	return cmd.Run()
}

func (r *Runtime) HandleSignal(signal os.Signal) (errs []error) {
	fmt.Fprintf(r.Stdout, "\nHandling Signal (%s)\n", signal)
	for cmd := range r.cmdSet {
		err := cmd.Process.Signal(signal)
		if err != nil {
			errs = append(errs, fmt.Errorf("Sending %v to %v returned: %v", signal, cmd, err))
		}
	}
	return
}
