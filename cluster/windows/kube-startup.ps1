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

# kube-startup.ps1 is used to run kubelet and kubeproxy as a process. It uses nssm (https://nssm.cc/) process manager to register kubelet and kube-proxy process,
# The processes can be viewed using TaskManager(Taskmgr.exe).
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

$kubeletDirectory = (Get-Item $KubeletExePath).Directory.FullName
$kubeproxyDirectory = (Get-Item $KubeProxyExePath).Directory.FullName

# Assemble the Kubelet executable arguments
$kubeletArgs = @("--hostname-override=$Hostname","--pod-infra-container-image=$InfraContainerImage","--resolv-conf=""""","--api-servers=$APIServer","--cluster-dns=$ClusterDNS")
# Assemble the kube-proxy executable arguments
$kubeproxyArgs = @("--hostname-override=$Hostname","--proxy-mode=userspace","--bind-address=$Hostname","--master=$APIServer")

# Setup kubelet service
nssm install kubelet "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
nssm set kubelet Application "$KubeletExePath"
nssm set kubelet AppDirectory "$kubeletDirectory"
nssm set kubelet AppParameters $kubeletArgs
nssm set kubelet DisplayName kubelet
nssm set kubelet Description kubelet
nssm set kubelet Start SERVICE_AUTO_START
nssm set kubelet ObjectName LocalSystem
nssm set kubelet Type SERVICE_WIN32_OWN_PROCESS
# Delay restart if application runs for less than 1500 ms
nssm set kubelet AppThrottle 1500
nssm set kubelet AppStdout "$LogDirectory\kubelet.log"
nssm set kubelet AppStderr "$LogDirectory\kubelet.err.log"
nssm set kubelet AppStdoutCreationDisposition 4
nssm set kubelet AppStderrCreationDisposition 4
nssm set kubelet AppRotateFiles 1
nssm set kubelet AppRotateOnline 1
# Rotate Logs Every 24 hours or 1 gb 
nssm set kubelet AppRotateSeconds 86400
nssm set kubelet AppRotateBytes 1073741824
nssm set kubelet AppEnvironmentExtra CONTAINER_NETWORK=$ContainerNetwork


# Setup kube-proxy service
nssm install kube-proxy "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
nssm set kube-proxy Application "$KubeProxyExePath"
nssm set kube-proxy AppDirectory "$kubeproxyDirectory"
nssm set kube-proxy AppParameters $kubeproxyArgs
nssm set kube-proxy DisplayName kube-proxy
nssm set kube-proxy Description kube-proxy
nssm set kube-proxy Start SERVICE_AUTO_START
nssm set kube-proxy ObjectName LocalSystem
nssm set kube-proxy Type SERVICE_WIN32_OWN_PROCESS
# Delay restart if application runs for less than 1500 ms
nssm set kube-proxy AppThrottle 1500
nssm set kube-proxy AppStdout "$LogDirectory\kube-proxy.log"
nssm set kube-proxy AppStderr "$LogDirectory\kube-proxy.err.log"
nssm set kube-proxy AppStdoutCreationDisposition 4
nssm set kube-proxy AppStderrCreationDisposition 4
nssm set kube-proxy AppRotateFiles 1
nssm set kube-proxy AppRotateOnline 1
# Rotate Logs Every 24 hours or 1 gb 
nssm set kube-proxy AppRotateSeconds 86400
nssm set kube-proxy AppRotateBytes 1073741824
nssm set kube-proxy AppEnvironmentExtra INTERFACE_TO_ADD_SERVICE_IP=$InterfaceForServiceIP

# Start kubelet and kube-proxy Services
echo "Starting kubelet"
Start-Service kubelet
echo "Starting kube-proxy"
Start-Service kube-proxy
