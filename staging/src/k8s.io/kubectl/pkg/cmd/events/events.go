/*
Copyright 2021 The Kubernetes Authors.

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

package events

import (
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

const (
	eventsUsageStr = "events TODO"
)

var (
	logsLong = templates.LongDesc(i18n.T(`
		Get events TODO.`))

	logsExample = templates.Examples(i18n.T(`
		# Return recent events
		kubectl events

		TODO`))

	selectorTail       int64 = 10
	eventssUsageErrStr       = fmt.Sprintf("expected '%s'.\nPOD or TYPE/NAME is a required argument for the logs command", eventsUsageStr)
)

type EventsOptions struct {
	AllNamespaces bool
	Namespace     string
	Watch         bool

	builder *resource.Builder

	genericclioptions.IOStreams
}

func NewEventsOptions(streams genericclioptions.IOStreams, watch bool) *EventsOptions {
	return &EventsOptions{
		IOStreams: streams,
		Watch:     watch,
	}
}

// NewCmdEvents creates a new pod logs command
func NewCmdEvents(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewEventsOptions(streams, false)

	cmd := &cobra.Command{
		Use:                   eventsUsageStr,
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Print the logs for a container in a pod"),
		Long:                  logsLong,
		Example:               logsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	o.AddFlags(cmd)
	return cmd
}

func (o *EventsOptions) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "After listing the requested events, watch for more events.")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
}

func (o *EventsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.builder = f.NewBuilder()

	return nil
}

func (o EventsOptions) Validate() error {
	// Nothing to do here.
	return nil
}

// Run retrieves events
func (o EventsOptions) Run() error {
	r := o.builder.
		Unstructured().
		NamespaceParam(o.Namespace).DefaultNamespace().AllNamespaces(o.AllNamespaces).
		ResourceTypes("events").
		SelectAllParam(true).
		Flatten().
		Do()

	if err := r.Err(); err != nil {
		return err
	}

	r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			fmt.Printf("visit err: %#v\n", err)
		}
		fmt.Printf("visit: %#v\n", info.Object)
		return nil
	})
	return nil
}
