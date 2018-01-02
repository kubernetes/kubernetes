//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package kubeexec

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	kubeletcmd "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"

	"github.com/heketi/heketi/executors/sshexec"
	"github.com/heketi/heketi/pkg/kubernetes"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/lpabon/godbc"
)

const (
	KubeGlusterFSPodLabelKey = "glusterfs-node"
)

type KubeExecutor struct {
	// Embed all sshexecutor functions
	sshexec.SshExecutor

	// save kube configuration
	config     *KubeConfig
	namespace  string
	kube       *client.Clientset
	rest       restclient.Interface
	kubeConfig *restclient.Config
}

var (
	logger          = utils.NewLogger("[kubeexec]", utils.LEVEL_DEBUG)
	inClusterConfig = func() (*restclient.Config, error) {
		return restclient.InClusterConfig()
	}
)

func setWithEnvVariables(config *KubeConfig) {
	var env string

	// Namespace / Project
	env = os.Getenv("HEKETI_KUBE_NAMESPACE")
	if "" != env {
		config.Namespace = env
	}

	// FSTAB
	env = os.Getenv("HEKETI_FSTAB")
	if "" != env {
		config.Fstab = env
	}

	// Snapshot Limit
	env = os.Getenv("HEKETI_SNAPSHOT_LIMIT")
	if "" != env {
		i, err := strconv.Atoi(env)
		if err == nil {
			config.SnapShotLimit = i
		}
	}

	// Determine if Heketi should communicate with Gluster
	// pods deployed by a DaemonSet
	env = os.Getenv("HEKETI_KUBE_GLUSTER_DAEMONSET")
	if "" != env {
		env = strings.ToLower(env)
		if env[0] == 'y' || env[0] == '1' {
			config.GlusterDaemonSet = true
		} else if env[0] == 'n' || env[0] == '0' {
			config.GlusterDaemonSet = false
		}
	}

	// Use POD names
	env = os.Getenv("HEKETI_KUBE_USE_POD_NAMES")
	if "" != env {
		env = strings.ToLower(env)
		if env[0] == 'y' || env[0] == '1' {
			config.UsePodNames = true
		} else if env[0] == 'n' || env[0] == '0' {
			config.UsePodNames = false
		}
	}
}

func NewKubeExecutor(config *KubeConfig) (*KubeExecutor, error) {
	// Override configuration
	setWithEnvVariables(config)

	// Initialize
	k := &KubeExecutor{}
	k.config = config
	k.Throttlemap = make(map[string]chan bool)
	k.RemoteExecutor = k

	if k.config.Fstab == "" {
		k.Fstab = "/etc/fstab"
	} else {
		k.Fstab = config.Fstab
	}

	// Get namespace
	var err error
	if k.config.Namespace == "" {
		k.config.Namespace, err = kubernetes.GetNamespace()
		if err != nil {
			return nil, logger.LogError("Namespace must be provided in configuration: %v", err)
		}
	}
	k.namespace = k.config.Namespace

	// Create a Kube client configuration
	k.kubeConfig, err = inClusterConfig()
	if err != nil {
		return nil, logger.LogError("Unable to create configuration for Kubernetes: %v", err)
	}

	// Get a raw REST client.  This is still needed for kube-exec
	restCore, err := coreclient.NewForConfig(k.kubeConfig)
	if err != nil {
		return nil, logger.LogError("Unable to create a client connection: %v", err)
	}
	k.rest = restCore.RESTClient()

	// Get a Go-client for Kubernetes
	k.kube, err = client.NewForConfig(k.kubeConfig)
	if err != nil {
		logger.Err(err)
		return nil, fmt.Errorf("Unable to create a client set")
	}

	// Show experimental settings
	if k.config.RebalanceOnExpansion {
		logger.Warning("Rebalance on volume expansion has been enabled.  This is an EXPERIMENTAL feature")
	}

	godbc.Ensure(k != nil)
	godbc.Ensure(k.Fstab != "")

	return k, nil
}

func (k *KubeExecutor) RemoteCommandExecute(host string,
	commands []string,
	timeoutMinutes int) ([]string, error) {

	// Throttle
	k.AccessConnection(host)
	defer k.FreeConnection(host)

	// Execute
	return k.ConnectAndExec(host,
		"pods",
		commands,
		timeoutMinutes)
}

func (k *KubeExecutor) ConnectAndExec(host, resource string,
	commands []string,
	timeoutMinutes int) ([]string, error) {

	// Used to return command output
	buffers := make([]string, len(commands))

	// Get pod name
	var (
		podName string
		err     error
	)
	if k.config.UsePodNames {
		podName = host
	} else if k.config.GlusterDaemonSet {
		podName, err = k.getPodNameFromDaemonSet(host)
	} else {
		podName, err = k.getPodNameByLabel(host)
	}
	if err != nil {
		return nil, err
	}

	// Get container name
	podSpec, err := k.kube.Core().Pods(k.namespace).Get(podName, v1.GetOptions{})
	if err != nil {
		return nil, logger.LogError("Unable to get pod spec for %v: %v",
			podName, err)
	}
	containerName := podSpec.Spec.Containers[0].Name

	for index, command := range commands {

		// Remove any whitespace
		command = strings.Trim(command, " ")

		// SUDO is *not* supported

		// Create REST command
		req := k.rest.Post().
			Resource(resource).
			Name(podName).
			Namespace(k.namespace).
			SubResource("exec").
			Param("container", containerName)
		req.VersionedParams(&api.PodExecOptions{
			Container: containerName,
			Command:   []string{"/bin/bash", "-c", command},
			Stdout:    true,
			Stderr:    true,
		}, api.ParameterCodec)

		// Create SPDY connection
		exec, err := remotecommand.NewExecutor(k.kubeConfig, "POST", req.URL())
		if err != nil {
			logger.Err(err)
			return nil, fmt.Errorf("Unable to setup a session with %v", podName)
		}

		// Create a buffer to trap session output
		var b bytes.Buffer
		var berr bytes.Buffer

		// Excute command
		err = exec.Stream(remotecommand.StreamOptions{
			SupportedProtocols: kubeletcmd.SupportedStreamingProtocols,
			Stdout:             &b,
			Stderr:             &berr,
		})
		if err != nil {
			logger.LogError("Failed to run command [%v] on %v: Err[%v]: Stdout [%v]: Stderr [%v]",
				command, podName, err, b.String(), berr.String())
			return nil, fmt.Errorf("Unable to execute command on %v: %v", podName, berr.String())
		}
		logger.Debug("Host: %v Pod: %v Command: %v\nResult: %v", host, podName, command, b.String())
		buffers[index] = b.String()

	}

	return buffers, nil
}

func (k *KubeExecutor) RebalanceOnExpansion() bool {
	return k.config.RebalanceOnExpansion
}

func (k *KubeExecutor) SnapShotLimit() int {
	return k.config.SnapShotLimit
}

func (k *KubeExecutor) getPodNameByLabel(host string) (string, error) {
	// Get a list of pods
	pods, err := k.kube.Core().Pods(k.config.Namespace).List(v1.ListOptions{
		LabelSelector: KubeGlusterFSPodLabelKey + "==" + host,
	})
	if err != nil {
		logger.Err(err)
		return "", fmt.Errorf("Failed to get list of pods")
	}

	numPods := len(pods.Items)
	if numPods == 0 {
		// No pods found with that label
		err := fmt.Errorf("No pods with the label '%v=%v' were found",
			KubeGlusterFSPodLabelKey, host)
		logger.Critical(err.Error())
		return "", err

	} else if numPods > 1 {
		// There are more than one pod with the same label
		err := fmt.Errorf("Found %v pods with the sharing the same label '%v=%v'",
			numPods, KubeGlusterFSPodLabelKey, host)
		logger.Critical(err.Error())
		return "", err
	}

	// Get pod name
	return pods.Items[0].ObjectMeta.Name, nil
}

func (k *KubeExecutor) getPodNameFromDaemonSet(host string) (string, error) {
	// Get a list of pods
	pods, err := k.kube.Core().Pods(k.config.Namespace).List(v1.ListOptions{
		LabelSelector: KubeGlusterFSPodLabelKey,
	})
	if err != nil {
		logger.Err(err)
		return "", logger.LogError("Failed to get list of pods")
	}

	// Go through the pods looking for the node
	var glusterPod string
	for _, pod := range pods.Items {
		if pod.Spec.NodeName == host {
			glusterPod = pod.ObjectMeta.Name
		}
	}
	if glusterPod == "" {
		return "", logger.LogError("Unable to find a GlusterFS pod on host %v "+
			"with a label key %v", host, KubeGlusterFSPodLabelKey)
	}

	// Get pod name
	return glusterPod, nil
}
