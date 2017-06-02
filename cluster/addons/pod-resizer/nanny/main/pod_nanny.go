// Copyright 2016 The Kubernetes Authors. All rights reserved.
package main

import (
	"os"
	"time"

	log "github.com/golang/glog"
	flag "github.com/spf13/pflag"

	"k8s.io/kubernetes/cluster/addons/pod-resizer/nanny"
	resource "k8s.io/kubernetes/pkg/api/resource"

	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_2"
	"k8s.io/kubernetes/pkg/client/restclient"
)

const NO_VALUE = "MISSING"

var (
	// Flags to define the resource requirements.
	baseCpu        = flag.String("cpu", NO_VALUE, "The base CPU resource requirement.")
	cpuPerNode     = flag.String("extra_cpu", "0", "The amount of CPU to add per node.")
	baseMemory     = flag.String("memory", NO_VALUE, "The base memory resource requirement.")
	memoryPerNode  = flag.String("extra_memory", "0Mi", "The amount of memory to add per node.")
	baseStorage    = flag.String("storage", NO_VALUE, "The base storage resource requirement.")
	storagePerNode = flag.String("extra_storage", "0Gi", "The amount of storage to add per node.")
	threshold      = flag.Int("threshold", 0, "A number between 0-100. The dependent's resources are rewritten when they deviate from expected by more than threshold.")
	// Flags to identify the container to nanny.
	podNamespace  = flag.String("namespace", os.Getenv("MY_POD_NAMESPACE"), "The namespace of the ward. This defaults to the nanny's own pod.")
	deployment    = flag.String("deployment", "", "The name of the deployment being monitored. This is required.")
	podName       = flag.String("pod", os.Getenv("MY_POD_NAME"), "The name of the pod to watch. This defaults to the nanny's own pod.")
	containerName = flag.String("container", "pod-nanny", "The name of the container to watch. This defaults to the nanny itself.")
	// The poll period, in ms, to check the dependent container.
	pollPeriod = time.Millisecond * time.Duration(*flag.Int("poll_period", 10000, "The time, in milliseconds, to poll the dependent container."))
)

func main() {
	// First log our starting config, and then set up.
	log.Infof("Invoked by %v", os.Args)
	flag.Parse()

	// Perform further validation of flags.
	if *deployment == "" {
		log.Fatal("Must specify a deployment.")
	}

	if *threshold < 0 || *threshold > 100 {
		log.Fatal("Threshold must be between 0 and 100 inclusively, was %d.", threshold)
	}

	log.Infof("Watching namespace: %s, pod: %s, container: %s.", *podNamespace, *podName, *containerName)
	log.Infof("cpu: %s, extra_cpu: %s, memory: %s, extra_memory: %s, storage: %s, extra_storage: %s", *baseCpu, *cpuPerNode, *baseMemory, *memoryPerNode, *baseStorage, *storagePerNode)

	// Set up work objects.
	config, err := restclient.InClusterConfig()
	if err != nil {
		log.Fatal(err)
	}

	clientset, err := release_1_2.NewForConfig(config)
	if err != nil {
		log.Fatal(err)
	}
	k8s := nanny.NewKubernetesClient(*podNamespace, *deployment, *podName, *containerName, clientset)

	var resources []nanny.Resource

	// Monitor only the resources specified.
	if *baseCpu != NO_VALUE {
		resources = append(resources, nanny.Resource{
			Base:         resource.MustParse(*baseCpu),
			ExtraPerNode: resource.MustParse(*cpuPerNode),
			Name:         "cpu",
		})
	}

	if *baseMemory != NO_VALUE {
		resources = append(resources, nanny.Resource{
			Base:         resource.MustParse(*baseMemory),
			ExtraPerNode: resource.MustParse(*memoryPerNode),
			Name:         "memory",
		})
	}

	if *baseStorage != NO_VALUE {
		resources = append(resources, nanny.Resource{
			Base:         resource.MustParse(*baseStorage),
			ExtraPerNode: resource.MustParse(*memoryPerNode),
			Name:         "storage",
		})
	}

	log.Infof("Resources: %v", resources)
	est := nanny.LinearEstimator{
		Resources: resources,
	}

	// Begin nannying.
	nanny.PollApiServer(k8s, est, *containerName, pollPeriod, uint64(*threshold))
}
