# Copyright 2019 The Kubernetes Authors.
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

<#
.SYNOPSIS
  Top-level script that runs on Windows nodes to join them to the K8s cluster.
#>

# IMPORTANT PLEASE NOTE:
# Any time the file structure in the `windows` directory changes, `windows/BUILD`
# and `k8s.io/release/lib/releaselib.sh` must be manually updated with the changes.
# We HIGHLY recommend not changing the file structure, because consumers of
# Kubernetes releases depend on the release structure remaining stable.

$ErrorActionPreference = 'Stop'

# Turn on tracing to debug
# Set-PSDebug -Trace 1

# Update TLS setting to enable Github downloads and disable progress bar to
# increase download speed.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'

# Returns the GCE instance metadata value for $Key where key is an "attribute"
# of the instance. If the key is not present in the instance metadata returns
# $Default if set, otherwise returns $null.
function Get-InstanceMetadataAttribute {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$false)] [string]$Default
  )

  $url = ("http://metadata.google.internal/computeMetadata/v1/instance/" +
          "attributes/$Key")
  try {
    $client = New-Object Net.WebClient
    $client.Headers.Add('Metadata-Flavor', 'Google')
    return ($client.DownloadString($url)).Trim()
  }
  catch [System.Net.WebException] {
    if ($Default) {
      return $Default
    }
    else {
      Write-Host "Failed to retrieve value for $Key."
      return $null
    }
  }
}

# Fetches the value of $MetadataKey, saves it to C:\$Filename and imports it as
# a PowerShell module.
#
# Note: this function depends on common.psm1.
function FetchAndImport-ModuleFromMetadata {
  param (
    [parameter(Mandatory=$true)] [string]$MetadataKey,
    [parameter(Mandatory=$true)] [string]$Filename
  )

  $module = Get-InstanceMetadataAttribute $MetadataKey
  if (Test-Path C:\$Filename) {
    if (-not $REDO_STEPS) {
      Log-Output "Skip: C:\$Filename already exists, not overwriting"
      Import-Module -Force C:\$Filename
      return
    }
    Log-Output "Warning: C:\$Filename already exists, will overwrite it."
  }
  New-Item -ItemType file -Force C:\$Filename | Out-Null
  Set-Content C:\$Filename $module
  Import-Module -Force C:\$Filename
}

# Returns true if the ENABLE_STACKDRIVER_WINDOWS or ENABLE_NODE_LOGGING field in kube_env is true.
# $KubeEnv is a hash table containing the kube-env metadata keys+values.
# ENABLE_NODE_LOGGING is used for legacy Stackdriver Logging, and will be deprecated (always set to False)
# soon. ENABLE_STACKDRIVER_WINDOWS is added to indicate whether logging is enabled for windows nodes.
function IsLoggingEnabled {
  param (
    [parameter(Mandatory=$true)] [hashtable]$KubeEnv
  )

  if ($KubeEnv.Contains('ENABLE_STACKDRIVER_WINDOWS') -and `
      ($KubeEnv['ENABLE_STACKDRIVER_WINDOWS'] -eq 'true')) {
    return $true
  } elseif ($KubeEnv.Contains('ENABLE_NODE_LOGGING') -and `
      ($KubeEnv['ENABLE_NODE_LOGGING'] -eq 'true')) {
    return $true
  }
  return $false
}

try {
  # Don't use FetchAndImport-ModuleFromMetadata for common.psm1 - the common
  # module includes variables and functions that any other function may depend
  # on.
  $module = Get-InstanceMetadataAttribute 'common-psm1'
  New-Item -ItemType file -Force C:\common.psm1 | Out-Null
  Set-Content C:\common.psm1 $module
  Import-Module -Force C:\common.psm1

  # TODO(pjh): update the function to set $Filename automatically from the key,
  # then put these calls into a loop over a list of XYZ-psm1 keys.
  FetchAndImport-ModuleFromMetadata 'k8s-node-setup-psm1' 'k8s-node-setup.psm1'

  Dump-DebugInfoToConsole

  $kube_env = Fetch-KubeEnv
  Set-EnvironmentVars

  # Set to true if there's a feature that needs a reboot
  $restart_computer = $false

  $should_enable_hyperv = Test-ShouldEnableHyperVFeature
  $hyperv_feature_enabled = Test-HyperVFeatureEnabled
  if ($should_enable_hyperv -and -not ($hyperv_feature_enabled)) {
    Enable-HyperVFeature
    Log-Output 'Restarting computer after enabling Windows Hyper-V feature'
    $restart_computer = $true
  }

  if (-not (Test-ContainersFeatureInstalled)) {
    Install-ContainersFeature
    Log-Output 'Restarting computer after enabling Windows Containers feature'
    $restart_computer = $true
  }

  if ($restart_computer) {
    Restart-Computer -Force
    # Restart-Computer does not stop the rest of the script from executing.
    exit 0
  }

  # Set the TCP/IP Parameters to keep idle connections alive.
  Set-WindowsTCPParameters

  Set-PrerequisiteOptions

  if (Test-IsTestCluster $kube_env) {
    Log-Output 'Test cluster detected, installing OpenSSH.'
    FetchAndImport-ModuleFromMetadata 'install-ssh-psm1' 'install-ssh.psm1'
    InstallAndStart-OpenSsh
    StartProcess-WriteSshKeys
  }

  Create-Directories
  Download-HelperScripts

  DownloadAndInstall-Crictl
  Configure-Crictl
  Setup-ContainerRuntime
  DownloadAndInstall-KubernetesBinaries
  DownloadAndInstall-NodeProblemDetector
  DownloadAndInstall-CSIProxyBinaries
  DownloadAndInstall-AuthProviderGcpBinary
  Start-CSIProxy
  Create-NodePki
  Create-KubeletKubeconfig
  Create-KubeproxyKubeconfig
  Create-NodeProblemDetectorKubeConfig
  Create-AuthProviderGcpConfig
  Set-PodCidr
  Configure-HostNetworkingService
  Prepare-CniNetworking
  Configure-HostDnsConf
  Configure-GcePdTools
  Configure-Kubelet
  Configure-NodeProblemDetector

  # Even if Logging agent is already installed, the function will still [re]start the service.
  if (IsLoggingEnabled $kube_env) {
    Install-LoggingAgent
    Configure-LoggingAgent
    Restart-LoggingAgent
  }
  # Flush cache to disk before starting kubelet & kube-proxy services
  # to make metadata server route and stackdriver service more persistent.
  Write-Volumecache C -PassThru
  Start-WorkerServices
  Log-Output 'Waiting 15 seconds for node to join cluster.'
  Start-Sleep 15
  Verify-WorkerServices

  $config = New-FileRotationConfig
  # TODO(random-liu): Generate containerd log into the log directory.
  Schedule-LogRotation -Pattern '.*\.log$' -Path ${env:LOGS_DIR} -RepetitionInterval $(New-Timespan -Hour 1) -Config $config

  Pull-InfraContainer
  # Flush cache to disk to persist the setup status
  Write-Volumecache C -PassThru
}
catch {
  Write-Host 'Exception caught in script:'
  Write-Host $_.InvocationInfo.PositionMessage
  Write-Host "Kubernetes Windows node setup failed: $($_.Exception.Message)"
  # Make sure kubelet won't remain running in case any failure happened during the startup.
  Write-Host "Cleaning up, Unregistering WorkerServices..."
  Unregister-WorkerServices
  exit 1
}
