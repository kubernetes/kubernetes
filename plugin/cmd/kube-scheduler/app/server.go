/*
Copyright 2014 The Kubernetes Authors.

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

// Package app implements a Server object for running the scheduler.
package app

import (
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	goruntime "runtime"
	"strconv"

	"k8s.io/apiserver/pkg/server/healthz"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/plugin/cmd/kube-scheduler/app/options"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// NewSchedulerCommand creates a *cobra.Command object with default parameters
func NewSchedulerCommand() *cobra.Command {
	s := options.NewSchedulerServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "kube-scheduler",
		Long: `The Kubernetes scheduler is a policy-rich, topology-aware,
workload-specific function that significantly impacts availability, performance,
and capacity. The scheduler needs to take into account individual and collective
resource requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality, inter-workload
interference, deadlines, and so on. Workload-specific requirements will be exposed
through the API as necessary.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// Run runs the specified SchedulerServer.  This should never exit.
func Run(s *options.SchedulerServer) error {
	kubecli, err := createClient(s)
	if err != nil {
		return fmt.Errorf("unable to create kube client: %v", err)
	}

	recorder := createRecorder(kubecli, s)

	informerFactory := informers.NewSharedInformerFactory(kubecli, 0)
	// cache only non-terminal pods
	podInformer := factory.NewPodInformer(kubecli, 0)

	sched, err := CreateScheduler(
		s,
		kubecli,
		informerFactory.Core().V1().Nodes(),
		podInformer,
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		recorder,
	)
	if err != nil {
		return fmt.Errorf("error creating scheduler: %v", err)
	}

	if s.Port != -1 {
		go startHTTP(s)
	}

	stop := make(chan struct{})
	defer close(stop)
	go podInformer.Informer().Run(stop)
	informerFactory.Start(stop)
	// Waiting for all cache to sync before scheduling.
	informerFactory.WaitForCacheSync(stop)
	controller.WaitForCacheSync("scheduler", stop, podInformer.Informer().HasSynced)

	run := func(stopCh <-chan struct{}) {
		sched.Run()
		<-stopCh
	}

	if !s.LeaderElection.LeaderElect {
		run(stop)
		return fmt.Errorf("finished without leader elect")
	}

	id, err := os.Hostname()
	if err != nil {
		return fmt.Errorf("unable to get hostname: %v", err)
	}

	rl, err := resourcelock.New(s.LeaderElection.ResourceLock,
		s.LockObjectNamespace,
		s.LockObjectName,
		kubecli.CoreV1(),
		resourcelock.ResourceLockConfig{
			Identity:      id,
			EventRecorder: recorder,
		})
	if err != nil {
		return fmt.Errorf("error creating lock: %v", err)
	}

	leaderElector, err := leaderelection.NewLeaderElector(
		leaderelection.LeaderElectionConfig{
			Lock:          rl,
			LeaseDuration: s.LeaderElection.LeaseDuration.Duration,
			RenewDeadline: s.LeaderElection.RenewDeadline.Duration,
			RetryPeriod:   s.LeaderElection.RetryPeriod.Duration,
			Callbacks: leaderelection.LeaderCallbacks{
				OnStartedLeading: run,
				OnStoppedLeading: func() {
					utilruntime.HandleError(fmt.Errorf("lost master"))
				},
			},
		})
	if err != nil {
		return err
	}

	leaderElector.Run()

	return fmt.Errorf("lost lease")
}

func startHTTP(s *options.SchedulerServer) {
	mux := http.NewServeMux()
	healthz.InstallHandler(mux)
	if s.EnableProfiling {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		if s.EnableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
	}
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.KubeSchedulerConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}
	configz.InstallHandler(mux)
	mux.Handle("/metrics", prometheus.Handler())

	server := &http.Server{
		Addr:    net.JoinHostPort(s.Address, strconv.Itoa(int(s.Port))),
		Handler: mux,
	}
	glog.Fatal(server.ListenAndServe())
}
