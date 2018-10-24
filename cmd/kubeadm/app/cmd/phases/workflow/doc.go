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
Package workflow implements a workflow manager to be used for
implementing composable kubeadm workflows.

Composable kubeadm workflows are built by an ordered sequence of phases;
each phase can have it's own, nested, ordered sequence of sub phases.
For instance

	preflight     	Run master pre-flight checks
	certs         	Generates all PKI assets necessary to establish the control plane
		/ca             Generates a self-signed kubernetes CA to provision identities for Kubernetes components
		/apiserver      Generates an API server serving certificate and key
		...
	kubeconfig		Generates all kubeconfig files necessary to establish the control plane
		/admin          Generates a kubeconfig file for the admin to use and for kubeadm itself
		/kubelet        Generates a kubeconfig file for the kubelet to use.
		...
	...

Phases are designed to be reusable across different kubeadm workflows thus allowing
e.g. reuse of phase certs in both kubeadm init and kubeadm join --control-plane workflows.

Each workflow can be defined and managed using a Runner, that will run all
the phases according to the given order; nested phases will be executed immediately
after their parent phase.

The Runner behavior can be changed by setting the RunnerOptions, typically
exposed as kubeadm command line flags, thus allowing to filter the list of phases
to be executed.
*/
package workflow
