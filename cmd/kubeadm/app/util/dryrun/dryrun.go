/*
Copyright 2017 The Kubernetes Authors.

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

package dryrun

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// FileToPrint represents a temporary file on disk that might want to be aliased when printing
// Useful for things like loading a file from /tmp/ but saying to the user "Would write file foo to /etc/kubernetes/..."
type FileToPrint struct {
	RealPath  string
	PrintPath string
}

// NewFileToPrint makes a new instance of FileToPrint with the specified arguments
func NewFileToPrint(realPath, printPath string) FileToPrint {
	return FileToPrint{
		RealPath:  realPath,
		PrintPath: printPath,
	}
}

// PrintDryRunFile is a helper method around PrintDryRunFiles
func PrintDryRunFile(fileName, realDir, printDir string, w io.Writer) error {
	return PrintDryRunFiles([]FileToPrint{
		NewFileToPrint(filepath.Join(realDir, fileName), filepath.Join(printDir, fileName)),
	}, w)
}

// PrintDryRunFiles prints the contents of the FileToPrints given to it to the writer w
func PrintDryRunFiles(files []FileToPrint, w io.Writer) error {
	errs := []error{}
	for _, file := range files {
		if len(file.RealPath) == 0 {
			continue
		}

		fileBytes, err := os.ReadFile(file.RealPath)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		// Make it possible to fake the path of the file; i.e. you may want to tell the user
		// "Here is what would be written to /etc/kubernetes/admin.conf", although you wrote it to /tmp/kubeadm-dryrun/admin.conf and are loading it from there
		// Fall back to the "real" path if PrintPath is not set
		outputFilePath := file.PrintPath
		if len(outputFilePath) == 0 {
			outputFilePath = file.RealPath
		}
		outputFilePath = filepath.ToSlash(outputFilePath)

		fmt.Fprintf(w, "[dryrun] Would write file %q with content:\n", outputFilePath)
		fmt.Fprintf(w, "%s", fileBytes)
	}
	return errorsutil.NewAggregate(errs)
}

// Waiter is an implementation of apiclient.Waiter that should be used for dry-running
type Waiter struct{}

// NewWaiter returns a new Waiter object that talks to the given Kubernetes cluster
func NewWaiter() apiclient.Waiter {
	return &Waiter{}
}

// WaitForControlPlaneComponents just returns a dummy nil, to indicate that the program should just proceed
func (w *Waiter) WaitForControlPlaneComponents(podsMap map[string]*v1.Pod, apiServerAddress string) error {
	return nil
}

// WaitForAPI just returns a dummy nil, to indicate that the program should just proceed
func (w *Waiter) WaitForAPI() error {
	fmt.Println("[dryrun] Would wait for the API Server's /healthz endpoint to return 'ok'")
	return nil
}

// WaitForPodsWithLabel just returns a dummy nil, to indicate that the program should just proceed
func (w *Waiter) WaitForPodsWithLabel(kvLabel string) error {
	fmt.Printf("[dryrun] Would wait for the Pods with the label %q in the %s namespace to become Running\n", kvLabel, metav1.NamespaceSystem)
	return nil
}

// WaitForKubelet blocks until the kubelet /healthz endpoint returns 'ok'
func (w *Waiter) WaitForKubelet(healthzAddress string, healthzPort int32) error {
	fmt.Printf("[dryrun] Would make sure the kubelet returns 'ok' at http://%s:%d/healthz\n", healthzAddress, healthzPort)
	return nil
}

// SetTimeout is a no-op; we don't wait in this implementation
func (w *Waiter) SetTimeout(_ time.Duration) {}

// WaitForStaticPodControlPlaneHashes returns an empty hash for all control plane images;
func (w *Waiter) WaitForStaticPodControlPlaneHashes(_ string) (map[string]string, error) {
	return map[string]string{
		kubeadmconstants.KubeAPIServer:         "",
		kubeadmconstants.KubeControllerManager: "",
		kubeadmconstants.KubeScheduler:         "",
	}, nil
}

// WaitForStaticPodSingleHash returns an empty hash
// but the empty strings there are needed
func (w *Waiter) WaitForStaticPodSingleHash(_ string, _ string) (string, error) {
	return "", nil
}

// WaitForStaticPodHashChange returns a dummy nil error in order for the flow to just continue as we're dryrunning
func (w *Waiter) WaitForStaticPodHashChange(_, _, _ string) error {
	return nil
}

// PrintFilesIfDryRunning prints the static pod manifests to stdout and informs about the temporary directory to go and lookup when dry running
func PrintFilesIfDryRunning(needPrintManifest bool, manifestDir string, outputWriter io.Writer) error {
	var files []FileToPrint
	// Print static pod manifests if it is a control plane
	if needPrintManifest {
		fmt.Printf("[dryrun] Wrote certificates, kubeconfig files and control plane manifests to the %q directory\n", manifestDir)
		for _, component := range kubeadmconstants.ControlPlaneComponents {
			realPath := kubeadmconstants.GetStaticPodFilepath(component, manifestDir)
			outputPath := kubeadmconstants.GetStaticPodFilepath(component, kubeadmconstants.GetStaticPodDirectory())
			files = append(files, NewFileToPrint(realPath, outputPath))
		}
	} else {
		fmt.Printf("[dryrun] Wrote certificates and kubeconfig files to the %q directory\n", manifestDir)
	}

	fmt.Println("[dryrun] The certificates or kubeconfig files would not be printed due to their sensitive nature")
	fmt.Printf("[dryrun] Please examine the %q directory for details about what would be written\n", manifestDir)

	// Print kubelet config manifests
	kubeletConfigFiles := []string{kubeadmconstants.KubeletConfigurationFileName, kubeadmconstants.KubeletEnvFileName}
	for _, filename := range kubeletConfigFiles {
		realPath := filepath.Join(manifestDir, filename)
		outputPath := filepath.Join(kubeadmconstants.KubeletRunDirectory, filename)
		files = append(files, NewFileToPrint(realPath, outputPath))
	}

	return PrintDryRunFiles(files, outputWriter)
}
