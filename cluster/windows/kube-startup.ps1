#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# kube-startup.ps1 is used to run kubelet and kubeproxy as a process. The processes can be viewed using TaskManager(Taskmgr.exe).
# Please note that this startup script does not start the API server. Kubernetes control plane currently runs on Linux
# and only Kubelet and Kube-Proxy can be run on Windows
 
param (
    [Parameter(Mandatory=$true)][string]$ContainerNetwork, 
    [string]$InterfaceForServiceIP = "vEthernet (HNS Internal NIC)",    
    [string]$LogDirectory = "C:\temp",
    [Parameter(Mandatory=$true)][string]$Hostname,
    [Parameter(Mandatory=$true)][string]$APIServer,
    [string]$InfraContainerImage = "apprenda/pause",
    [string]$ClusterDNS = "10.0.0.10",
    [string]$KubeletExePath = ".\kubelet.exe",
    [string]$KubeProxyExePath = ".\kube-proxy.exe"
)

# CONTAINER_NETWORK environment variable is used by kubelet and is used to determine the Docker network to be used by PODs
$env:CONTAINER_NETWORK = $ContainerNetwork

# INTERFACE_TO_ADD_SERVICE_IP environment variable to which Services IPs will be added. By Default "vEthernet (HNS Internal NIC)" which is created by
# when Docker is installed is used as it is private to the Host
$env:INTERFACE_TO_ADD_SERVICE_IP = $InterfaceForServiceIP

# Runs the kubelet with the user provided or default options
Start-Process -FilePath "$KubeletExePath" -ArgumentList --hostname-override=$Hostname, --pod-infra-container-image=$InfraContainerImage, `
--resolv-conf=, --api-servers=$APIServer, --cluster-dns=$ClusterDNS -RedirectStandardError "$LogDirectory\kubelet.log" -NoNewWindow

# Runs the kube-proxy with the user provided or default options
Start-Process -FilePath "$KubeProxyExePath" -ArgumentList --proxy-mode=userspace, --hostname-override=$Hostname, --master=$APIServer, `
--bind-address=$Hostname -RedirectStandardError "$LogDirectory\kube-proxy.log" -NoNewWindow 