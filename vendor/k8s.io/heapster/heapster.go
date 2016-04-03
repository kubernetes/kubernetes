// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:generate ./hooks/run_extpoints.sh

package main

import (
	"crypto/tls"
	"flag"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/manager"
	"k8s.io/heapster/sinks"
	"k8s.io/heapster/sinks/cache"
	source_api "k8s.io/heapster/sources/api"
	"k8s.io/heapster/version"
)

var (
	argStatsResolution = flag.Duration("stats_resolution", 5*time.Second, "The resolution at which heapster will retain stats.")
	argSinkFrequency   = flag.Duration("sink_frequency", 10*time.Second, "Frequency at which data will be pushed to sinks")
	argCacheDuration   = flag.Duration("cache_duration", 4*time.Minute, "The total duration of the historical data that will be cached by heapster.")
	argUseModel        = flag.Bool("use_model", true, "When true, the internal model representation will be used")
	argModelResolution = flag.Duration("model_resolution", 1*time.Minute, "The resolution of the timeseries stored in the model. Applies only if use_model is true")
	argModelFrequency  = flag.Duration("model_frequency", 145*time.Second, "Frequency at which model will be updated. Applies only if use_model is true")
	argPort            = flag.Int("port", 8082, "port to listen to")
	argIp              = flag.String("listen_ip", "", "IP to listen on, defaults to all IPs")
	argMaxProcs        = flag.Int("max_procs", 0, "max number of CPUs that can be used simultaneously. Less than 1 for default (number of cores)")
	argTLSCertFile     = flag.String("tls_cert", "", "file containing TLS certificate")
	argTLSKeyFile      = flag.String("tls_key", "", "file containing TLS key")
	argTLSClientCAFile = flag.String("tls_client_ca", "", "file containing TLS client CA for client cert validation")
	argAllowedUsers    = flag.String("allowed_users", "", "comma-separated list of allowed users")
	argSources         manager.Uris
	argSinks           manager.Uris
)

func main() {
	defer glog.Flush()
	flag.Var(&argSources, "source", "source(s) to watch")
	flag.Var(&argSinks, "sink", "external sink(s) that receive data")
	flag.Parse()
	setMaxProcs()
	glog.Infof(strings.Join(os.Args, " "))
	glog.Infof("Heapster version %v", version.HeapsterVersion)
	if err := validateFlags(); err != nil {
		glog.Fatal(err)
	}
	sources, sink, manager, err := doWork()
	if err != nil {
		glog.Fatal(err)
	}
	handler := setupHandlers(sources, sink, manager)
	addr := fmt.Sprintf("%s:%d", *argIp, *argPort)
	glog.Infof("Starting heapster on port %d", *argPort)

	server := &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	if len(*argTLSCertFile) > 0 && len(*argTLSKeyFile) > 0 {
		if len(*argTLSClientCAFile) > 0 {
			authHandler, err := newAuthHandler(handler)
			if err != nil {
				glog.Fatal(err)
			}
			server.Handler = authHandler
			server.TLSConfig = &tls.Config{ClientAuth: tls.RequestClientCert}
		}

		glog.Fatal(server.ListenAndServeTLS(*argTLSCertFile, *argTLSKeyFile))
	} else {
		glog.Fatal(server.ListenAndServe())
	}
}

func validateFlags() error {
	if *argStatsResolution < time.Second {
		return fmt.Errorf("stats resolution needs to be greater than a second - %d", *argStatsResolution)
	}
	if *argUseModel && (*argStatsResolution >= *argModelResolution) {
		return fmt.Errorf("stats resolution '%d' is not less than model resolution '%d'", *argStatsResolution, *argModelResolution)
	}
	if *argSinkFrequency >= *argCacheDuration {
		return fmt.Errorf("sink frequency '%d' is expected to be lesser than cache duration '%d'", *argSinkFrequency, *argCacheDuration)
	}
	if *argModelFrequency <= *argModelResolution {
		return fmt.Errorf("model frequency '%d' is expected to be greater than model resolution '%d'", *argModelFrequency, *argModelResolution)
	}
	if (len(*argTLSCertFile) > 0 && len(*argTLSKeyFile) == 0) || (len(*argTLSCertFile) == 0 && len(*argTLSKeyFile) > 0) {
		return fmt.Errorf("both TLS certificate & key are required to enable TLS serving")
	}
	if len(*argTLSClientCAFile) > 0 && len(*argTLSCertFile) == 0 {
		return fmt.Errorf("client cert authentication requires TLS certificate & key")
	}
	return nil
}

func doWork() ([]source_api.Source, sinks.ExternalSinkManager, manager.Manager, error) {
	c := cache.NewCache(*argCacheDuration, time.Minute)
	sources, err := newSources(c)
	if err != nil {
		// Do not fail heapster is source setup fails for any reason.
		glog.Errorf("failed to setup sinks - %v", err)
	}
	sinkManager, err := sinks.NewExternalSinkManager(nil, c, *argSinkFrequency)
	if err != nil {
		return nil, nil, nil, err
	}

	// Spawn the Model Housekeeping goroutine even if the model is not enabled.
	// This will allow the model to be activated/deactivated in runtime.
	modelDuration := *argModelFrequency
	if (*argCacheDuration).Nanoseconds() < modelDuration.Nanoseconds() {
		modelDuration = *argCacheDuration
	}

	manager, err := manager.NewManager(sources, sinkManager, *argStatsResolution, *argCacheDuration, c, *argUseModel, *argModelResolution,
		modelDuration)
	if err != nil {
		return nil, nil, nil, err
	}
	if err := manager.SetSinkUris(argSinks); err != nil {
		// Do not fail heapster is sink setup fails for any reason.
		glog.Errorf("failed to setup sinks - %v", err)
	}

	manager.Start()
	return sources, sinkManager, manager, nil
}

func setMaxProcs() {
	// Allow as many threads as we have cores unless the user specified a value.
	var numProcs int
	if *argMaxProcs < 1 {
		numProcs = runtime.NumCPU()
	} else {
		numProcs = *argMaxProcs
	}
	runtime.GOMAXPROCS(numProcs)

	// Check if the setting was successful.
	actualNumProcs := runtime.GOMAXPROCS(0)
	if actualNumProcs != numProcs {
		glog.Warningf("Specified max procs of %d but using %d", numProcs, actualNumProcs)
	}
}
