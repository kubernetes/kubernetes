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

package main

import (
	goflag "flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"os"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	utilflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/apiserver/pkg/util/logs"
	"k8s.io/kubernetes/pkg/version/verflag"
)

// This Pod's /checknosnat takes `target` and `ips` arguments, and queries {target}/checknosnat?ips={ips}

type MasqTestProxy struct {
	Port string
}

func NewMasqTestProxy() *MasqTestProxy {
	return &MasqTestProxy{
		Port: "31235",
	}
}

// NewTestProxyCommand creates a *cobra.Command object with default parameters
func NewTestProxyCommand() *cobra.Command {
	m := NewMasqTestProxy()
	cmd := &cobra.Command{
		Use:  "MasqTestProxy",
		Long: `MasqTestProxy`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			m.Run()
		},
	}
	m.AddFlags(pflag.CommandLine)

	return cmd
}
func (m *MasqTestProxy) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&m.Port, "port", m.Port, "The port to serve /checknosnat endpoint on.")
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	command := NewTestProxyCommand()

	// TODO: once we switch everything over to Cobra commands, we can go back to calling
	// utilflag.InitFlags() (by removing its pflag.Parse() call). For now, we have to set the
	// normalize func and add the go flag set by hand.
	pflag.CommandLine.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	// utilflag.InitFlags()
	logs.InitLogs()
	defer logs.FlushLogs()

	if err := command.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func (m *MasqTestProxy) Run() error {
	// register handler
	http.HandleFunc("/checknosnat", checknosnat)

	// spin up the server
	return http.ListenAndServe(":"+m.Port, nil)
}

func checknosnatURL(pip, ips string) string {
	return fmt.Sprintf("http://%s/checknosnat?ips=%s", pip, ips)
}

func checknosnat(w http.ResponseWriter, req *http.Request) {
	url := checknosnatURL(req.URL.Query().Get("target"), req.URL.Query().Get("ips"))
	resp, err := http.Get(url)
	if err != nil {
		w.WriteHeader(500)
		fmt.Fprintf(w, "error querying %q, err: %v", url, err)
	} else {
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			w.WriteHeader(500)
			fmt.Fprintf(w, "error reading body of response from %q, err: %v", url, err)
		} else {
			// Respond the same status code and body as /checknosnat on the internal Pod
			w.WriteHeader(resp.StatusCode)
			w.Write(body)
		}
	}
	resp.Body.Close()
}
