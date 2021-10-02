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

package inclusterclient

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
)

var pollInterval int

// CmdInClusterClient is used by agnhost Cobra.
var CmdInClusterClient = &cobra.Command{
	Use:   "inclusterclient",
	Short: "Periodically poll the Kubernetes \"/healthz\" endpoint",
	Long: `Periodically polls the Kubernetes "/healthz" endpoint using the in-cluster config. Because of this, this subcommand is meant to be run inside of a Kubernetes pod.

This subcommand can also be used to validate token rotation.`,
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

func init() {
	CmdInClusterClient.Flags().IntVar(&pollInterval, "poll-interval", 30,
		"poll interval of call to /healthz in seconds")
}

func main(cmd *cobra.Command, args []string) {
	logs.InitLogs()
	defer logs.FlushLogs()

	flag.Set("logtostderr", "true")

	klog.Infof("started")

	cfg, err := rest.InClusterConfig()
	if err != nil {
		log.Fatalf("err: %v", err)
	}

	cfg.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return &debugRt{
			rt: rt,
		}
	})

	c := kubernetes.NewForConfigOrDie(cfg).RESTClient()

	t := time.Tick(time.Duration(pollInterval) * time.Second)
	for {
		<-t
		klog.Infof("calling /healthz")
		b, err := c.Get().AbsPath("/healthz").Do(context.TODO()).Raw()
		if err != nil {
			klog.Errorf("status=failed")
			klog.Errorf("error checking /healthz: %v\n%s\n", err, string(b))
		}
	}
}

type debugRt struct {
	rt http.RoundTripper
}

func (rt *debugRt) RoundTrip(req *http.Request) (*http.Response, error) {
	authHeader := req.Header.Get("Authorization")
	if len(authHeader) != 0 {
		authHash := sha256.Sum256([]byte(fmt.Sprintf("%s|%s", "salt", authHeader)))
		klog.Infof("authz_header=%s", base64.RawURLEncoding.EncodeToString(authHash[:]))
	} else {
		klog.Errorf("authz_header=<empty>")
	}
	return rt.rt.RoundTrip(req)
}

func (rt *debugRt) WrappedRoundTripper() http.RoundTripper { return rt.rt }
