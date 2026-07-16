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

package nosnatproxy

import (
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/spf13/cobra"
	"k8s.io/component-base/logs"
)

// CmdNoSnatTestProxy is used by agnhost Cobra.
var CmdNoSnatTestProxy = &cobra.Command{
	Use:   "no-snat-test-proxy",
	Short: "Creates a proxy for the /checknosnat endpoint",
	Long:  `Creates the /checknosnat endpoint which proxies the request to the given target (/checknosnat?target=target_ip&ips=ip1,ip2) and returns its response, or a 500 response on error.`,
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var port string

func init() {
	CmdNoSnatTestProxy.Flags().StringVar(&port, "port", "31235", "The port to serve /checknosnat endpoint on.")
}

// This Pod's /checknosnat takes `target` and `ips` arguments, and queries {target}/checknosnat?ips={ips}

type masqTestProxy struct {
	Port string
}

func main(cmd *cobra.Command, args []string) {
	m := &masqTestProxy{
		Port: port,
	}

	logs.InitLogs()
	defer logs.FlushLogs()

	if err := m.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func (m *masqTestProxy) Run() error {
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
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		w.WriteHeader(500)
		fmt.Fprintf(w, "error reading body of response from %q, err: %v", url, err)
		return
	}

	// Respond the same status code and body as /checknosnat on the internal Pod
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}
