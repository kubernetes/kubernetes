/*
Copyright 2019 The Kubernetes Authors.

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

package genericclioptions

import (
	"time"

	"github.com/spf13/cobra"
)

type StreamFlags struct {
	PingInterval time.Duration
}

func (o *StreamFlags) AddFlags(c *cobra.Command) {
	// TODO: on `get`, only allow with `-w`
	// TODO: on `logs`, only allow with `-f`
	c.Flags().DurationVar(&o.PingInterval, "ping-interval", 1*time.Minute, "Perform ping on this interval to keep the connection alive")
}
