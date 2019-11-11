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

	// KubeConfigPath is optional and will set the KUBECONFIG for communication to the APIServer
	KubeConfigPath string
	// ServerDryRun is optional and gates whether to fetch dryRun diffs from an APIServer
	ServerDryRun bool
}

func (r *Runtime) CheckDeps() error {
	if _, err := exec.LookPath("kubectl"); err != nil {
		return err
	}

	if r.KubeConfigPath != "" {
		if _, err := os.Stat(r.KubeConfigPath); err != nil {
			return err
		}
	}

	return nil
}

func (r *Runtime) CheckConfig() error {
	// check for no refs passed
	// check for multiple refs passed
	var noRefs []string
	var multiRefs []string
	for _, addon := range r.Config.Addons {
		if addon.ManifestRef == "" && addon.KustomizeRef == "" {
			noRefs = append(noRefs, addon.Name)
		}
		if addon.ManifestRef != "" && addon.KustomizeRef != "" {
			multiRefs = append(multiRefs, addon.Name)
		}
	}
	if len(noRefs) > 0 {
		return fmt.Errorf("AddonInstallerConfiguration contains addons that have no ref: %v", noRefs)
	}
	if len(multiRefs) > 0 {
		return fmt.Errorf("AddonInstallerConfiguration contains addons defining multiple refs: %v", multiRefs)
	}

	// check for duplicates
	var duplicateNames []string
	nameCounts := map[string]int{}
	var duplicateRefs []string
	refCounts := map[string]int{}
	for _, addon := range r.Config.Addons {
		nameCounts[addon.Name]++

		ref := addon.ManifestRef
		if addon.KustomizeRef != "" {
			ref = addon.KustomizeRef
		}
		refCounts[ref]++
	}
	for name, count := range nameCounts {
		if count >= 2 {
			duplicateNames = append(duplicateNames, name)
		}
	}
	for ref, count := range refCounts {
		if count >= 2 {
			duplicateRefs = append(duplicateRefs, ref)
		}
	}
	if len(duplicateNames) > 0 {
		return fmt.Errorf("AddonInstallerConfiguration lists addons with duplicate names: %v", duplicateNames)
	}
	if len(duplicateRefs) > 0 {
		return fmt.Errorf("AddonInstallerConfiguration lists addons with duplicate refs: %v", duplicateRefs)
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

func (r *Runtime) InstallSingleAddon(addon config.Addon) error {
	ref := addon.ManifestRef
	args := []string{"apply", "-R", "-f", ref}
	msg := "...installing '" + addon.Name + "' using manifest: " + ref

	if addon.KustomizeRef != "" {
		ref = addon.KustomizeRef
		args = []string{"apply", "-k", ref}
		msg = "...installing '" + addon.Name + "' using kustomize: " + ref
	}

	if r.Config.DryRun {
		msg += " (dry run)"
		args = append(args, "--server-dry-run")
	}
	fmt.Fprintln(r.Stdout, msg)

	if r.Config.DryRun && !r.ServerDryRun {
		return nil
	}

	err := r.runCommand("kubectl", args...)
	if err != nil {
		return err
	}
	return nil
}

func (r *Runtime) DeleteSingleAddon(addon config.Addon) error {
	ref := addon.ManifestRef
	args := []string{"delete", "-R", "-f", ref, "--ignore-not-found=true"}
	msg := "...deleting '" + addon.Name + "' using manifest: " + ref

	if addon.KustomizeRef != "" {
		ref = addon.KustomizeRef
		args = []string{"delete", "-k", ref, "--ignore-not-found=true"}
		msg = "...deleting '" + addon.Name + "' using kustomize: " + ref
	}

	if r.Config.DryRun {
		msg += " (dry run)"
	}
	fmt.Fprintln(r.Stdout, msg)

	if r.Config.DryRun {
		// kubectl delete does not support ServerDryRun -- do not run command
		return nil
	}

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
	if r.KubeConfigPath != "" {
		cmd.Env = os.Environ()
		cmd.Env = append(cmd.Env, "KUBECONFIG="+r.KubeConfigPath)
	}

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
