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

package apiclient

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// DryRunGetter is an interface that must be supplied to the NewDryRunClient function in order to contstruct a fully functional fake dryrun clientset
type DryRunGetter interface {
	HandleGetAction(core.GetAction) (bool, runtime.Object, error)
	HandleListAction(core.ListAction) (bool, runtime.Object, error)
}

// MarshalFunc takes care of converting any object to a byte array for displaying the object to the user
type MarshalFunc func(runtime.Object, schema.GroupVersion) ([]byte, error)

// DefaultMarshalFunc is the default MarshalFunc used; uses YAML to print objects to the user
func DefaultMarshalFunc(obj runtime.Object, gv schema.GroupVersion) ([]byte, error) {
	return kubeadmutil.MarshalToYaml(obj, gv)
}

// DryRunClientOptions specifies options to pass to NewDryRunClientWithOpts in order to get a dryrun clientset
type DryRunClientOptions struct {
	Writer          io.Writer
	Getter          DryRunGetter
	PrependReactors []core.Reactor
	AppendReactors  []core.Reactor
	MarshalFunc     MarshalFunc
	PrintGETAndLIST bool
}

// GetDefaultDryRunClientOptions returns the default DryRunClientOptions values
func GetDefaultDryRunClientOptions(drg DryRunGetter, w io.Writer) DryRunClientOptions {
	return DryRunClientOptions{
		Writer:          w,
		Getter:          drg,
		PrependReactors: []core.Reactor{},
		AppendReactors:  []core.Reactor{},
		MarshalFunc:     DefaultMarshalFunc,
		PrintGETAndLIST: false,
	}
}

// actionWithName is the generic interface for an action that has a name associated with it
// This just makes it easier to catch all actions that has a name; instead of hard-coding all request that has it associated
type actionWithName interface {
	core.Action
	GetName() string
}

// actionWithObject is the generic interface for an action that has an object associated with it
// This just makes it easier to catch all actions that has an object; instead of hard-coding all request that has it associated
type actionWithObject interface {
	core.Action
	GetObject() runtime.Object
}

// NewDryRunClient is a wrapper for NewDryRunClientWithOpts using some default values
func NewDryRunClient(drg DryRunGetter, w io.Writer) clientset.Interface {
	return NewDryRunClientWithOpts(GetDefaultDryRunClientOptions(drg, w))
}

// NewDryRunClientWithOpts returns a clientset.Interface that can be used normally for talking to the Kubernetes API.
// This client doesn't apply changes to the backend. The client gets GET/LIST values from the DryRunGetter implementation.
// This client logs all I/O to the writer w in YAML format
func NewDryRunClientWithOpts(opts DryRunClientOptions) clientset.Interface {
	// Build a chain of reactors to act like a normal clientset; but log everything that is happening and don't change any state
	client := fakeclientset.NewSimpleClientset()

	// Build the chain of reactors. Order matters; first item here will be invoked first on match, then the second one will be evaluated, etc.
	defaultReactorChain := []core.Reactor{
		// Log everything that happens. Default the object if it's about to be created/updated so that the logged object is representative.
		&core.SimpleReactor{
			Verb:     "*",
			Resource: "*",
			Reaction: func(action core.Action) (bool, runtime.Object, error) {
				logDryRunAction(action, opts.Writer, opts.MarshalFunc)

				return false, nil, nil
			},
		},
		// Let the DryRunGetter implementation take care of all GET requests.
		// The DryRunGetter implementation may call a real API Server behind the scenes or just fake everything
		&core.SimpleReactor{
			Verb:     "get",
			Resource: "*",
			Reaction: func(action core.Action) (bool, runtime.Object, error) {
				getAction, ok := action.(core.GetAction)
				if !ok {
					// something's wrong, we can't handle this event
					return true, nil, fmt.Errorf("can't cast get reactor event action object to GetAction interface")
				}
				handled, obj, err := opts.Getter.HandleGetAction(getAction)

				if opts.PrintGETAndLIST {
					// Print the marshalled object format with one tab indentation
					objBytes, err := opts.MarshalFunc(obj, action.GetResource().GroupVersion())
					if err == nil {
						fmt.Println("[dryrun] Returning faked GET response:")
						PrintBytesWithLinePrefix(opts.Writer, objBytes, "\t")
					}
				}

				return handled, obj, err
			},
		},
		// Let the DryRunGetter implementation take care of all GET requests.
		// The DryRunGetter implementation may call a real API Server behind the scenes or just fake everything
		&core.SimpleReactor{
			Verb:     "list",
			Resource: "*",
			Reaction: func(action core.Action) (bool, runtime.Object, error) {
				listAction, ok := action.(core.ListAction)
				if !ok {
					// something's wrong, we can't handle this event
					return true, nil, fmt.Errorf("can't cast list reactor event action object to ListAction interface")
				}
				handled, objs, err := opts.Getter.HandleListAction(listAction)

				if opts.PrintGETAndLIST {
					// Print the marshalled object format with one tab indentation
					objBytes, err := opts.MarshalFunc(objs, action.GetResource().GroupVersion())
					if err == nil {
						fmt.Println("[dryrun] Returning faked LIST response:")
						PrintBytesWithLinePrefix(opts.Writer, objBytes, "\t")
					}
				}

				return handled, objs, err
			},
		},
		// For the verbs that modify anything on the server; just return the object if present and exit successfully
		&core.SimpleReactor{
			Verb:     "create",
			Resource: "*",
			Reaction: successfulModificationReactorFunc,
		},
		&core.SimpleReactor{
			Verb:     "update",
			Resource: "*",
			Reaction: successfulModificationReactorFunc,
		},
		&core.SimpleReactor{
			Verb:     "delete",
			Resource: "*",
			Reaction: successfulModificationReactorFunc,
		},
		&core.SimpleReactor{
			Verb:     "delete-collection",
			Resource: "*",
			Reaction: successfulModificationReactorFunc,
		},
		&core.SimpleReactor{
			Verb:     "patch",
			Resource: "*",
			Reaction: successfulModificationReactorFunc,
		},
	}

	// The chain of reactors will look like this:
	// opts.PrependReactors | defaultReactorChain | opts.AppendReactors | client.Fake.ReactionChain (default reactors for the fake clientset)
	fullReactorChain := append(opts.PrependReactors, defaultReactorChain...)
	fullReactorChain = append(fullReactorChain, opts.AppendReactors...)

	// Prepend the reaction chain with our reactors. Important, these MUST be prepended; not appended due to how the fake clientset works by default
	client.Fake.ReactionChain = append(fullReactorChain, client.Fake.ReactionChain...)
	return client
}

// successfulModificationReactorFunc is a no-op that just returns the POSTed/PUTed value if present; but does nothing to edit any backing data store.
func successfulModificationReactorFunc(action core.Action) (bool, runtime.Object, error) {
	objAction, ok := action.(actionWithObject)
	if ok {
		return true, objAction.GetObject(), nil
	}
	return true, nil, nil
}

// logDryRunAction logs the action that was recorded by the "catch-all" (*,*) reactor and tells the user what would have happened in an user-friendly way
func logDryRunAction(action core.Action, w io.Writer, marshalFunc MarshalFunc) {

	group := action.GetResource().Group
	if len(group) == 0 {
		group = "core"
	}
	fmt.Fprintf(w, "[dryrun] Would perform action %s on resource %q in API group \"%s/%s\"\n", strings.ToUpper(action.GetVerb()), action.GetResource().Resource, group, action.GetResource().Version)

	namedAction, ok := action.(actionWithName)
	if ok {
		fmt.Fprintf(w, "[dryrun] Resource name: %q\n", namedAction.GetName())
	}

	objAction, ok := action.(actionWithObject)
	if ok && objAction.GetObject() != nil {
		// Print the marshalled object with a tab indentation
		objBytes, err := marshalFunc(objAction.GetObject(), action.GetResource().GroupVersion())
		if err == nil {
			fmt.Println("[dryrun] Attached object:")
			PrintBytesWithLinePrefix(w, objBytes, "\t")
		}
	}

	patchAction, ok := action.(core.PatchAction)
	if ok {
		// Replace all occurrences of \" with a simple " when printing
		fmt.Fprintf(w, "[dryrun] Attached patch:\n\t%s\n", strings.Replace(string(patchAction.GetPatch()), `\"`, `"`, -1))
	}
}

// PrintBytesWithLinePrefix prints objBytes to writer w with linePrefix in the beginning of every line
func PrintBytesWithLinePrefix(w io.Writer, objBytes []byte, linePrefix string) {
	scanner := bufio.NewScanner(bytes.NewReader(objBytes))
	for scanner.Scan() {
		fmt.Fprintf(w, "%s%s\n", linePrefix, scanner.Text())
	}
}
