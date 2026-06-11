/*
Copyright 2026 The Kubernetes Authors.

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

package resourceconsumer

import (
	"fmt"
	"log"
	"net/http"

	"github.com/spf13/cobra"
)

// CmdResourceConsumer is used by agnhost Cobra.
var CmdResourceConsumer = &cobra.Command{
	Use:   "resource-consumer",
	Short: "Starts a HTTP server for consuming CPU, memory, and custom metrics",
	Args:  cobra.MaximumNArgs(0),
	Run:   runResourceConsumer,
}

var port int

func init() {
	CmdResourceConsumer.Flags().IntVar(&port, "port", 8080, "Port number.")
}

func runResourceConsumer(cmd *cobra.Command, args []string) {
	handler := NewResourceConsumerHandler()
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), handler))
}
