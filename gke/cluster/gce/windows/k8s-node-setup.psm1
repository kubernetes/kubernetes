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
  Library for configuring Windows nodes and joining them to the cluster.

.NOTES
  This module depends on common.psm1.

  Some portions copied / adapted from
  https://github.com/Microsoft/SDN/blob/master/Kubernetes/windows/start-kubelet.ps1.

.EXAMPLE
  Suggested usage for dev/test:
    [Net.ServicePointManager]::SecurityProtocol = `
        [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest `
        https://github.com/kubernetes/kubernetes/raw/master/cluster/gce/windows/k8s-node-setup.psm1 `
        -OutFile C:\k8s-node-setup.psm1
    Invoke-WebRequest `
        https://github.com/kubernetes/kubernetes/raw/master/cluster/gce/windows/configure.ps1 `
        -OutFile C:\configure.ps1
    Import-Module -Force C:\k8s-node-setup.psm1  # -Force to override existing
    # Execute functions manually or run configure.ps1.
#>

# IMPORTANT PLEASE NOTE:
# Any time the file structure in the `windows` directory changes, `windows/BUILD`
# and `k8s.io/release/lib/releaselib.sh` must be manually updated with the changes.
# We HIGHLY recommend not changing the file structure, because consumers of
# Kubernetes releases depend on the release structure remaining stable.

# TODO: update scripts for these style guidelines:
#  - Remove {} around variable references unless actually needed for clarity.
#  - Always use single-quoted strings unless actually interpolating variables
#    or using escape characters.
#  - Use "approved verbs":
#    https://docs.microsoft.com/en-us/powershell/developer/cmdlet/approved-verbs-for-windows-powershell-commands
#  - Document functions using proper syntax:
#    https://technet.microsoft.com/en-us/library/hh847834(v=wps.620).aspx

$GCE_METADATA_SERVER = "169.254.169.254"
# The "management" interface is used by the kubelet and by Windows pods to talk
# to the rest of the Kubernetes cluster *without NAT*. This interface does not
# exist until an initial HNS network has been created on the Windows node - see
# Add_InitialHnsNetwork().
$MGMT_ADAPTER_NAME = "vEthernet (Ethernet*"
$CRICTL_VERSION = 'v1.21.0'
$CRICTL_SHA256 = '437d5301f6f5b9848ef057cee98474ce11a6679c91b4d4e83677a8c1f2415143'

Import-Module -Force C:\common.psm1

# Writes a TODO with $Message to the console.
function Log_Todo {
  param (
    [parameter(Mandatory=$true)] [string]$Message
  )
  Log-Output "TODO: ${Message}"
}

# Writes a not-implemented warning with $Message to the console and exits the
# script.
function Log_NotImplemented {
  param (
    [parameter(Mandatory=$true)] [string]$Message
  )
  Log-Output "Not implemented yet: ${Message}" -Fatal
}

# Fails and exits if the route to the GCE metadata server is not present,
# otherwise does nothing and emits nothing.
function Verify_GceMetadataServerRouteIsPresent {
  Try {
    Get-NetRoute `
        -ErrorAction "Stop" `
        -AddressFamily IPv4 `
        -DestinationPrefix ${GCE_METADATA_SERVER}/32 | Out-Null
  } Catch [Microsoft.PowerShell.Cmdletization.Cim.CimJobException] {
    Log-Output -Fatal `
        ("GCE metadata server route is not present as expected.`n" +
         "$(Get-NetRoute -AddressFamily IPv4 | Out-String)")
  }
}

# Checks if the route to the GCE metadata server is present. Returns when the
# route is NOT present or after a timeout has expired.
function WaitFor_GceMetadataServerRouteToBeRemoved {
  $elapsed = 0
  $timeout = 60
  Log-Output ("Waiting up to ${timeout} seconds for GCE metadata server " +
              "route to be removed")
  while (${elapsed} -lt ${timeout}) {
    Try {
      Get-NetRoute `
          -ErrorAction "Stop" `
          -AddressFamily IPv4 `
          -DestinationPrefix ${GCE_METADATA_SERVER}/32 | Out-Null
    } Catch [Microsoft.PowerShell.Cmdletization.Cim.CimJobException] {
      break
    }
    $sleeptime = 2
    Start-Sleep ${sleeptime}
    ${elapsed} += ${sleeptime}
  }
}

# Adds a route to the GCE metadata server to every network interface.
function Add_GceMetadataServerRoute {
  # Before setting up HNS the Windows VM has a "vEthernet (nat)" interface and
  # a "Ethernet" interface, and the route to the metadata server exists on the
  # Ethernet interface. After adding the HNS network a "vEthernet (Ethernet)"
  # interface is added, and it seems to subsume the routes of the "Ethernet"
  # interface (trying to add routes on the Ethernet interface at this point just
  # results in "New-NetRoute : Element not found" errors). I don't know what's
  # up with that, but since it's hard to know what's the right thing to do here
  # we just try to add the route on all of the network adapters.
  Get-NetAdapter | ForEach-Object {
    $adapter_index = $_.InterfaceIndex
    New-NetRoute `
        -ErrorAction Ignore `
        -DestinationPrefix "${GCE_METADATA_SERVER}/32" `
        -InterfaceIndex ${adapter_index} | Out-Null
  }
}

# Returns a PowerShell object representing the Windows version.
function Get_WindowsVersion {
  # Unlike checking `[System.Environment]::OSVersion.Version`, this long-winded
  # approach gets the OS revision/patch number correctly
  # (https://superuser.com/a/1160428/652018).
  $win_ver = New-Object -TypeName PSObject
  $win_ver | Add-Member -MemberType NoteProperty -Name Major -Value $(Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion' CurrentMajorVersionNumber).CurrentMajorVersionNumber
  $win_ver | Add-Member -MemberType NoteProperty -Name Minor -Value $(Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion' CurrentMinorVersionNumber).CurrentMinorVersionNumber
  $win_ver | Add-Member -MemberType NoteProperty -Name Build -Value $(Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion' CurrentBuild).CurrentBuild
  $win_ver | Add-Member -MemberType NoteProperty -Name Revision -Value $(Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion' UBR).UBR
  return $win_ver
}

# Writes debugging information, such as Windows version and patch info, to the
# console.
function Dump-DebugInfoToConsole {
  Try {
    $version = Get_WindowsVersion | Out-String
    $hotfixes = "$(Get-Hotfix | Out-String)"
    $image = "$(Get-InstanceMetadata 'image' | Out-String)"
    Log-Output "Windows version:`n$version"
    Log-Output "Installed hotfixes:`n$hotfixes"
    Log-Output "GCE Windows image:`n$image"
  } Catch { }
}

# Configures Window Defender preferences
function Configure-WindowsDefender {
  if ((Get-WindowsFeature -Name 'Windows-Defender').Installed) {
    Log-Output "Configuring Windows Defender preferences"
    Set-MpPreference -SubmitSamplesConsent NeverSend
    Log-Output "Disabling Windows Defender sample submission"
    Set-MpPreference -MAPSReporting Disabled
    Log-Output "Disabling Windows Defender Microsoft Active Protection Service Reporting"

    Log-Output "Defender Preferences"
    Get-MpPreference
  }
}

# Converts the kube-env string in Yaml
#
# Returns: a PowerShell Hashtable object containing the key-value pairs from
#   kube-env.
function ConvertFrom_Yaml_KubeEnv {
  param (
    [parameter(Mandatory=$true)] [string]$kube_env_str
  )
  $kube_env_table = @{}
  $currentLine = $null
  switch -regex (${kube_env_str} -split '\r?\n') {
      '^(\S.*)' {
          # record start pattern, line that doesn't start with a whitespace
          if ($null -ne $currentLine) {
              $key, $val = $currentLine -split ":",2
              $kube_env_table[$key] = $val.Trim("'", " ", "`"")
          }
          $currentLine = $matches.1
          continue
      }

      '^(\s+.*)' {
          # line that start with whitespace
          $currentLine += $matches.1
          continue
      }
  }

  # Handle the last line if any
  if ($currentLine) {
      $key, $val = $currentLine -split ":",2
      $kube_env_table[$key] = $val.Trim("'", " ", "`"")
  }

  return ${kube_env_table}
}

# Fetches the kube-env from the instance metadata.
#
# Returns: a PowerShell Hashtable object containing the key-value pairs from
#   kube-env.
function Fetch-KubeEnv {
  # Testing / debugging:
  # First:
  #   ${kube_env} = Get-InstanceMetadataAttribute 'kube-env'
  # or:
  #   ${kube_env} = [IO.File]::ReadAllText(".\kubeEnv.txt")
  # ${kube_env_table} = ConvertFrom_Yaml_KubeEnv ${kube_env}
  # ${kube_env_table}
  # ${kube_env_table}.GetType()

  # The type of kube_env is a powershell String.
  $kube_env = Get-InstanceMetadataAttribute 'kube-env'
  $kube_env_table = ConvertFrom_Yaml_KubeEnv ${kube_env}

  Log-Output "Logging kube-env key-value pairs except CERT and KEY values"
  foreach ($entry in $kube_env_table.GetEnumerator()) {
    if ((-not ($entry.Name.contains("CERT"))) -and (-not ($entry.Name.contains("KEY")))) {
      Log-Output "$($entry.Name): $($entry.Value)"
    }
  }
  return ${kube_env_table}
}

# Sets the environment variable $Key to $Value at the Machine scope (will
# be present in the environment for all new shells after a reboot).
function Set_MachineEnvironmentVar {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$true)] [AllowEmptyString()] [string]$Value
  )
  [Environment]::SetEnvironmentVariable($Key, $Value, "Machine")
}

# Sets the environment variable $Key to $Value in the current shell.
function Set_CurrentShellEnvironmentVar {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$true)] [AllowEmptyString()] [string]$Value
  )
  $expression = '$env:' + $Key + ' = "' + $Value + '"'
  Invoke-Expression ${expression}
}

# Sets environment variables used by Kubernetes binaries and by other functions
# in this module. Depends on numerous ${kube_env} keys.
function Set-EnvironmentVars {
  if ($kube_env.ContainsKey('WINDOWS_CONTAINER_RUNTIME')) {
      $container_runtime = ${kube_env}['WINDOWS_CONTAINER_RUNTIME']
      $container_runtime_endpoint = ${kube_env}['WINDOWS_CONTAINER_RUNTIME_ENDPOINT']
  } else {
      Log-Output "ERROR: WINDOWS_CONTAINER_RUNTIME not set in kube-env, falling back in CONTAINER_RUNTIME"
      $container_runtime = ${kube_env}['CONTAINER_RUNTIME']
      $container_runtime_endpoint = ${kube_env}['CONTAINER_RUNTIME_ENDPOINT']
  }
  # Turning the kube-env values into environment variables is not required but
  # it makes debugging this script easier, and it also makes the syntax a lot
  # easier (${env:K8S_DIR} can be expanded within a string but
  # ${kube_env}['K8S_DIR'] cannot be afaik).
  $env_vars = @{
    "K8S_DIR" = ${kube_env}['K8S_DIR']
    # Typically 'C:\etc\kubernetes\node\bin' (not just 'C:\etc\kubernetes\node')
    "NODE_DIR" = ${kube_env}['NODE_DIR']
    "CNI_DIR" = ${kube_env}['CNI_DIR']
    "CNI_CONFIG_DIR" = ${kube_env}['CNI_CONFIG_DIR']
    "WINDOWS_CNI_STORAGE_PATH" = ${kube_env}['WINDOWS_CNI_STORAGE_PATH']
    "WINDOWS_CNI_VERSION" = ${kube_env}['WINDOWS_CNI_VERSION']
    "CSI_PROXY_STORAGE_PATH" = ${kube_env}['CSI_PROXY_STORAGE_PATH']
    "CSI_PROXY_VERSION" = ${kube_env}['CSI_PROXY_VERSION']
    "ENABLE_CSI_PROXY" = ${kube_env}['ENABLE_CSI_PROXY']
    "PKI_DIR" = ${kube_env}['PKI_DIR']
    "CA_FILE_PATH" = ${kube_env}['CA_FILE_PATH']
    "KUBELET_CONFIG" = ${kube_env}['KUBELET_CONFIG_FILE']
    "BOOTSTRAP_KUBECONFIG" = ${kube_env}['BOOTSTRAP_KUBECONFIG_FILE']
    "KUBECONFIG" = ${kube_env}['KUBECONFIG_FILE']
    "KUBEPROXY_KUBECONFIG" = ${kube_env}['KUBEPROXY_KUBECONFIG_FILE']
    "LOGS_DIR" = ${kube_env}['LOGS_DIR']
    "MANIFESTS_DIR" = ${kube_env}['MANIFESTS_DIR']
    "INFRA_CONTAINER" = ${kube_env}['WINDOWS_INFRA_CONTAINER']
    "WINDOWS_ENABLE_PIGZ" = ${kube_env}['WINDOWS_ENABLE_PIGZ']
    "ENABLE_NODE_PROBLEM_DETECTOR" = ${kube_env}['ENABLE_NODE_PROBLEM_DETECTOR']
    "NODEPROBLEMDETECTOR_KUBECONFIG_FILE" = ${kube_env}['WINDOWS_NODEPROBLEMDETECTOR_KUBECONFIG_FILE']

    "Path" = ${env:Path} + ";" + ${kube_env}['NODE_DIR']
    "KUBE_NETWORK" = "l2bridge".ToLower()
    "KUBELET_CERT_PATH" = ${kube_env}['PKI_DIR'] + '\kubelet.crt'
    "KUBELET_KEY_PATH" = ${kube_env}['PKI_DIR'] + '\kubelet.key'

    "CONTAINER_RUNTIME" = $container_runtime
    "CONTAINER_RUNTIME_ENDPOINT" = $container_runtime_endpoint

    'LICENSE_DIR' = 'C:\Program Files\Google\Compute Engine\THIRD_PARTY_NOTICES'
  }

  # Set the environment variables in two ways: permanently on the machine (only
  # takes effect after a reboot), and in the current shell.
  $env_vars.GetEnumerator() | ForEach-Object{
    $message = "Setting environment variable: " + $_.key + " = " + $_.value
    Log-Output ${message}
    Set_MachineEnvironmentVar $_.key $_.value
    Set_CurrentShellEnvironmentVar $_.key $_.value
  }
}

# Configures various settings and prerequisites needed for the rest of the
# functions in this module and the Kubernetes binaries to operate properly.
function Set-PrerequisiteOptions {
  # Windows updates cause the node to reboot at arbitrary times.
  Log-Output "Disabling Windows Update service"
  & sc.exe config wuauserv start=disabled
  & sc.exe stop wuauserv
  Write-VerboseServiceInfoToConsole -Service 'wuauserv' -Delay 1

  # Use TLS 1.2: needed for Invoke-WebRequest downloads from github.com.
  [Net.ServicePointManager]::SecurityProtocol = `
      [Net.SecurityProtocolType]::Tls12

  Configure-WindowsDefender
}

# Creates directories where other functions in this module will read and write
# data.
# Note: C:\tmp is required for running certain kubernetes tests.
#       C:\var\log is used by kubelet to stored container logs and also
#       hard-coded in the fluentd/stackdriver config for log collection.
function Create-Directories {
  Log-Output "Creating ${env:K8S_DIR} and its subdirectories."
  ForEach ($dir in ("${env:K8S_DIR}", "${env:NODE_DIR}", "${env:LOGS_DIR}",
    "${env:CNI_DIR}", "${env:CNI_CONFIG_DIR}", "${env:MANIFESTS_DIR}",
    "${env:PKI_DIR}", "${env:LICENSE_DIR}"), "C:\tmp", "C:\var\log") {
    mkdir -Force $dir
  }
}

# Downloads some external helper scripts needed by other functions in this
# module.
function Download-HelperScripts {
  if (ShouldWrite-File ${env:K8S_DIR}\hns.psm1) {
    MustDownload-File `
        -OutFile ${env:K8S_DIR}\hns.psm1 `
        -URLs 'https://storage.googleapis.com/gke-release/winnode/config/sdn/master/hns.psm1'
  }
}

# Downloads the gke-exec-auth-plugin for TPM-based authentication to the
# master, if auth plugin support has been requested for this node (see
# Test-NodeUsesAuthPlugin).
# https://github.com/kubernetes/cloud-provider-gcp/tree/master/cmd/gke-exec-auth-plugin
#
# Required ${kube_env} keys:
#   EXEC_AUTH_PLUGIN_LICENSE_URL
#   EXEC_AUTH_PLUGIN_HASH
#   EXEC_AUTH_PLUGIN_URL
function DownloadAndInstall-AuthPlugin {
  if (-not (Test-NodeUsesAuthPlugin ${kube_env})) {
    Log-Output 'Skipping download of auth plugin'
    return
  }
  if (-not (ShouldWrite-File "${env:NODE_DIR}\gke-exec-auth-plugin.exe")) {
    return
  }

  if (-not ($kube_env.ContainsKey('EXEC_AUTH_PLUGIN_LICENSE_URL') -and
            $kube_env.ContainsKey('EXEC_AUTH_PLUGIN_HASH') -and
            $kube_env.ContainsKey('EXEC_AUTH_PLUGIN_URL'))) {
    Log-Output -Fatal ("Missing one or more kube-env keys needed for " +
                       "downloading auth plugin: $(Out-String $kube_env)")
  }
  MustDownload-File `
      -URLs ${kube_env}['EXEC_AUTH_PLUGIN_URL'] `
      -Hash ${kube_env}['EXEC_AUTH_PLUGIN_HASH'] `
      -OutFile "${env:NODE_DIR}\gke-exec-auth-plugin.exe"
  MustDownload-File `
      -URLs ${kube_env}['EXEC_AUTH_PLUGIN_LICENSE_URL'] `
      -OutFile "${env:LICENSE_DIR}\LICENSE_gke-exec-auth-plugin.txt"
}

# Returns true if this node enables workload identity
function Test-EnableWorkloadIdentity {
  param (
    [parameter(Mandatory=$true)] [hashtable]$KubeEnv
  )
  return $KubeEnv.Contains('ENABLE_WINDOWS_WORKLOAD_IDENTITY') -and `
      ($KubeEnv['ENABLE_WINDOWS_WORKLOAD_IDENTITY'] -eq 'true')
}

#  Download and install the gke-metadata-server to support workload identity
#  if ENABLE_WINDOWS_WORKLOAD_IDENTITY is set as true in kube_env
#  Required ${kube_env} keys:
#    GKE_METADATA_SERVER_URL
function DownloadAndInstall-GKEMetadataServer {
  if (-not (Test-EnableWorkloadIdentity ${kube_env})) {
    Log-Output 'Skipping downloading gke-metadata-server as workload identity is not enabled'
    return
  }
  if (-not (ShouldWrite-File "${env:NODE_DIR}\gke-metadata-server.exe")) {
    return
  }

  if (-not ($kube_env.ContainsKey('GKE_METADATA_SERVER_URL'))) {
    Log-Output -Fatal ("Missing GKE_METADATA_SERVER_URL " +
                       "for downloading gke-metadata-server: $(Out-String $kube_env)")
  }

  $tar_url = ${kube_env}['GKE_METADATA_SERVER_URL']
  $sha_url = $tar_url + ".sha512"
  MustDownload-File -URLs $sha_url `
      -OutFile "${env:NODE_DIR}\gke-metadata-server.exe.sha512"
  $sha = $(Get-Content "${env:NODE_DIR}\gke-metadata-server.exe.sha512")

  MustDownload-File `
      -URLs $tar_url `
      -OutFile "${env:NODE_DIR}\gke-metadata-server.exe" `
      -Hash $sha `
      -Algorithm SHA512
}

# Sets up the gke-metadata-server arguments and starts it as native Windows service
# if workload identity is enabled.
#
# Required ${kube_env} keys:
#   GKE_METADATA_SERVER_ARGS
function Start-GKEMetadataServer {
  if (-not (Test-EnableWorkloadIdentity ${kube_env})) {
    Log-Output 'Skipping starting gke-metadata-server as workload identity is not enabled'
    return
  }

  if (Get-Process | Where-Object Name -eq "gke-metadata-server") {
    Log-Output -Fatal `
        "A gke-metadata-server process is already running, don't know what to do"
  }

  $args_str = ${kube_env}['GKE_METADATA_SERVER_ARGS']
  $args = $args_str.Split(" ")

  # Creates log directory if specified in args
  ForEach($arg in $args) {
    $key, $val = $arg -split "=",2
    if ($key -eq "--log_dir") {
      Log-Output "Creating log directory ${val}"
      mkdir -Force $val
      break
    }
  }

  $pod_gateway = Get_Endpoint_Gateway_From_CIDR ${env:POD_CIDR}
  $mgmt_ip = (Get_MgmtNetAdapter | Get-NetIPAddress -AddressFamily IPv4).IPAddress
  $default_args = @(`
      "--addr=${pod_gateway}:988",
      "--metrics_addr=${mgmt_ip}:989"
  )
  $args = $args + $default_args
  Log-Output "gke-metadata-server args: ${args}"


  Log-Output "Creating gke-metadata-server service"
  & sc.exe create gke-metadata-server binPath= "${env:NODE_DIR}\gke-metadata-server.exe ${args}" start= demand
  & sc.exe failure gke-metadata-server reset= 0 actions= restart/10000
  Log-Output "Starting gke-metadata-server service"
  & sc.exe start gke-metadata-server

  # Wait for gke-metadata-server to be ready within 10s
  $waited = 0
  $timeout = 10
  while (((Get-Service gke-metadata-server).Status -ne 'Running') -and $waited -lt $timeout) {
    Start-Sleep 1
    $waited++
  }

  if ($waited -ge $timeout) {
    Log-Output "$(Get-Service gke-metadata-server | Out-String)"
    Throw ("Timeout while waiting ${timeout} seconds for gke-metadata-server to start")
  }
}

# Downloads the Kubernetes binaries from kube-env's NODE_BINARY_TAR_URL and
# puts them in a subdirectory of $env:K8S_DIR.
#
# Required ${kube_env} keys:
#   NODE_BINARY_TAR_URL
function DownloadAndInstall-KubernetesBinaries {
  # Assume that presence of kubelet.exe indicates that the kubernetes binaries
  # were already previously downloaded to this node.
  if (-not (ShouldWrite-File ${env:NODE_DIR}\kubelet.exe)) {
    return
  }

  $tmp_dir = 'C:\k8s_tmp'
  New-Item -Force -ItemType 'directory' $tmp_dir | Out-Null

  $urls = ${kube_env}['NODE_BINARY_TAR_URL'].Split(",")
  $filename = Split-Path -leaf $urls[0]
  $hash = $null
  if ($kube_env.ContainsKey('NODE_BINARY_TAR_HASH')) {
    $hash = ${kube_env}['NODE_BINARY_TAR_HASH']
  }
  MustDownload-File -Hash $hash -OutFile $tmp_dir\$filename -URLs $urls

  tar xzvf $tmp_dir\$filename -C $tmp_dir
  Move-Item -Force $tmp_dir\kubernetes\node\bin\* ${env:NODE_DIR}\
  Move-Item -Force `
      $tmp_dir\kubernetes\LICENSES ${env:LICENSE_DIR}\LICENSES_kubernetes

  # Clean up the temporary directory
  Remove-Item -Force -Recurse $tmp_dir
}

# Downloads the csi-proxy binaries from kube-env's CSI_PROXY_STORAGE_PATH and
# CSI_PROXY_VERSION, and then puts them in a subdirectory of $env:NODE_DIR.
# Note: for now the installation is skipped for non-test clusters. Will be
# installed for all cluster after tests pass.
# Required ${kube_env} keys:
#   CSI_PROXY_STORAGE_PATH and CSI_PROXY_VERSION
function DownloadAndInstall-CSIProxyBinaries {
  if ("${env:ENABLE_CSI_PROXY}" -eq "true") {
    if (ShouldWrite-File ${env:NODE_DIR}\csi-proxy.exe) {
      $tmp_dir = 'C:\k8s_tmp'
      New-Item -Force -ItemType 'directory' $tmp_dir | Out-Null
      $filename = 'csi-proxy.exe'
      $urls = "${env:CSI_PROXY_STORAGE_PATH}/${env:CSI_PROXY_VERSION}/$filename"
      MustDownload-File -OutFile $tmp_dir\$filename -URLs $urls
      Move-Item -Force $tmp_dir\$filename ${env:NODE_DIR}\$filename
      # Clean up the temporary directory
      Remove-Item -Force -Recurse $tmp_dir
    }
  }
}

function Start-CSIProxy {
  if ("${env:ENABLE_CSI_PROXY}" -eq "true") {
    Log-Output "Creating CSI Proxy Service"
    $flags = "-windows-service -log_file=${env:LOGS_DIR}\csi-proxy.log -logtostderr=false"
    & sc.exe create csiproxy binPath= "${env:NODE_DIR}\csi-proxy.exe $flags"
    & sc.exe failure csiproxy reset= 0 actions= restart/10000
    Log-Output "Starting CSI Proxy Service"
    & sc.exe start csiproxy
    Write-VerboseServiceInfoToConsole -Service 'csiproxy' -Delay 1
  }
}

# TODO(pjh): this is copied from
# https://github.com/Microsoft/SDN/blob/master/Kubernetes/windows/start-kubelet.ps1#L98.
# See if there's a way to fetch or construct the "management subnet" so that
# this is not needed.
function ConvertTo_DecimalIP
{
  param(
    [parameter(Mandatory = $true, Position = 0)]
    [Net.IPAddress] $IPAddress
  )

  $i = 3; $decimal_ip = 0;
  $IPAddress.GetAddressBytes() | % {
    $decimal_ip += $_ * [Math]::Pow(256, $i); $i--
  }
  return [UInt32]$decimal_ip
}

# TODO(pjh): this is copied from
# https://github.com/Microsoft/SDN/blob/master/Kubernetes/windows/start-kubelet.ps1#L98.
# See if there's a way to fetch or construct the "management subnet" so that
# this is not needed.
function ConvertTo_DottedDecimalIP
{
  param(
    [parameter(Mandatory = $true, Position = 0)]
    [Uint32] $IPAddress
  )

  $dotted_ip = $(for ($i = 3; $i -gt -1; $i--) {
    $remainder = $IPAddress % [Math]::Pow(256, $i)
    ($IPAddress - $remainder) / [Math]::Pow(256, $i)
    $IPAddress = $remainder
  })
  return [String]::Join(".", $dotted_ip)
}

# TODO(pjh): this is copied from
# https://github.com/Microsoft/SDN/blob/master/Kubernetes/windows/start-kubelet.ps1#L98.
# See if there's a way to fetch or construct the "management subnet" so that
# this is not needed.
function ConvertTo_MaskLength
{
  param(
    [parameter(Mandatory = $True, Position = 0)]
    [Net.IPAddress] $SubnetMask
  )

  $bits = "$($SubnetMask.GetAddressBytes() | % {
    [Convert]::ToString($_, 2)
  } )" -replace "[\s0]"
  return $bits.Length
}

# Returns a network adapter object for the "management" interface via which the
# Windows pods+kubelet will communicate with the rest of the Kubernetes cluster.
#
# This function will fail if Add_InitialHnsNetwork() has not been called first.
function Get_MgmtNetAdapter {
  $net_adapter = Get-NetAdapter | Where-Object Name -like ${MGMT_ADAPTER_NAME}
  if (-not ${net_adapter}) {
    Throw ("Failed to find a suitable network adapter, check your network " +
           "settings.")
  }

  return $net_adapter
}

# Decodes the base64 $Data string and writes it as binary to $File. Does
# nothing if $File already exists and $REDO_STEPS is not set.
function Write_PkiData {
  param (
    [parameter(Mandatory=$true)] [string] $Data,
    [parameter(Mandatory=$true)] [string] $File
  )

  if (-not (ShouldWrite-File $File)) {
    return
  }

  # This command writes out a PEM certificate file, analogous to "base64
  # --decode" on Linux. See https://stackoverflow.com/a/51914136/1230197.
  [IO.File]::WriteAllBytes($File, [Convert]::FromBase64String($Data))
  Log_Todo ("need to set permissions correctly on ${File}; not sure what the " +
            "Windows equivalent of 'umask 077' is")
  # Linux: owned by root, rw by user only.
  #   -rw------- 1 root root 1.2K Oct 12 00:56 ca-certificates.crt
  #   -rw------- 1 root root 1.3K Oct 12 00:56 kubelet.crt
  #   -rw------- 1 root root 1.7K Oct 12 00:56 kubelet.key
  # Windows:
  #   https://docs.microsoft.com/en-us/dotnet/api/system.io.fileattributes
  #   https://docs.microsoft.com/en-us/dotnet/api/system.io.fileattributes
}

# Creates the node PKI files in $env:PKI_DIR.
#
# Required ${kube_env} keys:
#   CA_CERT
# ${kube_env} keys that can be omitted for nodes that do not use an
# authentication plugin:
#   KUBELET_CERT
#   KUBELET_KEY
function Create-NodePki {
  Log-Output 'Creating node pki files'

  if ($kube_env.ContainsKey('CA_CERT')) {
    $CA_CERT_BUNDLE = ${kube_env}['CA_CERT']
    Write_PkiData "${CA_CERT_BUNDLE}" ${env:CA_FILE_PATH}
  }
  else {
    Log-Output -Fatal 'CA_CERT not present in kube-env'
  }

  # On nodes that use a plugin to support authentication, KUBELET_CERT and
  # KUBELET_KEY will not be present - TPM_BOOTSTRAP_CERT and TPM_BOOTSTRAP_KEY
  # should be set instead.
  if (Test-NodeUsesAuthPlugin ${kube_env}) {
    Log-Output ('Skipping KUBELET_CERT and KUBELET_KEY, plugin will be used ' +
                'for authentication')
    return
  }

  if ($kube_env.ContainsKey('KUBELET_CERT')) {
    $KUBELET_CERT = ${kube_env}['KUBELET_CERT']
    Write_PkiData "${KUBELET_CERT}" ${env:KUBELET_CERT_PATH}
  }
  else {
    Log-Output -Fatal 'KUBELET_CERT not present in kube-env'
  }
  if ($kube_env.ContainsKey('KUBELET_KEY')) {
    $KUBELET_KEY = ${kube_env}['KUBELET_KEY']
    Write_PkiData "${KUBELET_KEY}" ${env:KUBELET_KEY_PATH}
  }
  else {
    Log-Output -Fatal 'KUBELET_KEY not present in kube-env'
  }

  Get-ChildItem ${env:PKI_DIR}
}

# Creates the bootstrap kubelet kubeconfig at $env:BOOTSTRAP_KUBECONFIG.
# https://kubernetes.io/docs/reference/command-line-tools-reference/kubelet-tls-bootstrapping/
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   KUBERNETES_MASTER_NAME: the apiserver IP address.
function Write_BootstrapKubeconfig {
  if (-not (ShouldWrite-File ${env:BOOTSTRAP_KUBECONFIG})) {
    return
  }

  # TODO(mtaufen): is user "kubelet" correct? Other examples use e.g.
  # "system:node:$(hostname)".

  $apiserverAddress = ${kube_env}['KUBERNETES_MASTER_NAME']
  New-Item -Force -ItemType file ${env:BOOTSTRAP_KUBECONFIG} | Out-Null
  Set-Content ${env:BOOTSTRAP_KUBECONFIG} `
'apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    client-certificate: KUBELET_CERT_PATH
    client-key: KUBELET_KEY_PATH
clusters:
- name: local
  cluster:
    server: https://APISERVER_ADDRESS
    certificate-authority: CA_FILE_PATH
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context'.`
    replace('KUBELET_CERT_PATH', ${env:KUBELET_CERT_PATH}).`
    replace('KUBELET_KEY_PATH', ${env:KUBELET_KEY_PATH}).`
    replace('APISERVER_ADDRESS', ${apiserverAddress}).`
    replace('CA_FILE_PATH', ${env:CA_FILE_PATH})
  Log-Output ("kubelet bootstrap kubeconfig:`n" +
              "$(Get-Content -Raw ${env:BOOTSTRAP_KUBECONFIG})")
}

# Fetches the kubelet kubeconfig from the metadata server and writes it to
# $env:KUBECONFIG.
#
# Create-NodePki() must be called first.
function Write_KubeconfigFromMetadata {
  if (-not (ShouldWrite-File ${env:KUBECONFIG})) {
    return
  }

  $kubeconfig = Get-InstanceMetadataAttribute 'kubeconfig'
  if ($kubeconfig -eq $null) {
    Log-Output `
        "kubeconfig metadata key not found, can't write ${env:KUBECONFIG}" `
        -Fatal
  }
  Set-Content ${env:KUBECONFIG} $kubeconfig
  Log-Output ("kubelet kubeconfig from metadata (non-bootstrap):`n" +
              "$(Get-Content -Raw ${env:KUBECONFIG})")
}

# Creates the kubelet kubeconfig at $env:KUBECONFIG for nodes that use an
# authentication plugin, or at $env:BOOTSTRAP_KUBECONFIG for nodes that do not.
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   KUBERNETES_MASTER_NAME: the apiserver IP address.
function Create-KubeletKubeconfig {
  if (Test-NodeUsesAuthPlugin ${kube_env}) {
    Write_KubeconfigFromMetadata
  } else {
    Write_BootstrapKubeconfig
  }
}

# Creates the kubeconfig user file for applications that communicate with Kubernetes.
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   CA_CERT
#   KUBERNETES_MASTER_NAME
function Create-Kubeconfig {
  param (
    [parameter(Mandatory=$true)] [string]$Name,
    [parameter(Mandatory=$true)] [string]$Path,
    [parameter(Mandatory=$true)] [string]$Token
  )
  if (-not (ShouldWrite-File $Path)) {
    return
  }

  New-Item -Force -ItemType file $Path | Out-Null

  # In configure-helper.sh kubelet kubeconfig uses certificate-authority while
  # kubeproxy kubeconfig uses certificate-authority-data, ugh. Does it matter?
  # Use just one or the other for consistency?
  Set-Content $Path `
'apiVersion: v1
kind: Config
users:
- name: APP_NAME
  user:
    token: APP_TOKEN
clusters:
- name: local
  cluster:
    server: https://APISERVER_ADDRESS
    certificate-authority-data: CA_CERT
contexts:
- context:
    cluster: local
    user: APP_NAME
  name: service-account-context
current-context: service-account-context'.`
  replace('APP_NAME', $Name).`
  replace('APP_TOKEN', $Token).`
  replace('CA_CERT', ${kube_env}['CA_CERT']).`
  replace('APISERVER_ADDRESS', ${kube_env}['KUBERNETES_MASTER_NAME'])

  Log-Output ("${Name} kubeconfig:`n" +
              "$(Get-Content -Raw ${Path})")
}

# Creates the kube-proxy user kubeconfig file at $env:KUBEPROXY_KUBECONFIG.
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   CA_CERT
#   KUBE_PROXY_TOKEN
function Create-KubeproxyKubeconfig {
  Create-Kubeconfig -Name 'kube-proxy' `
    -Path ${env:KUBEPROXY_KUBECONFIG} `
    -Token ${kube_env}['KUBE_PROXY_TOKEN']
}

# Returns the IP alias range configured for this GCE instance.
function Get_IpAliasRange {
  $url = ("http://${GCE_METADATA_SERVER}/computeMetadata/v1/instance/" +
          "network-interfaces/0/ip-aliases/0")
  $client = New-Object Net.WebClient
  $client.Headers.Add('Metadata-Flavor', 'Google')
  return ($client.DownloadString($url)).Trim()
}

# Retrieves the pod CIDR and sets it in $env:POD_CIDR.
function Set-PodCidr {
  while($true) {
    $pod_cidr = Get_IpAliasRange
    if (-not $?) {
      Log-Output ${pod_cIDR}
      Log-Output "Retrying Get_IpAliasRange..."
      Start-Sleep -sec 1
      continue
    }
    break
  }

  Log-Output "fetched pod CIDR (same as IP alias range): ${pod_cidr}"
  Set_MachineEnvironmentVar "POD_CIDR" ${pod_cidr}
  Set_CurrentShellEnvironmentVar "POD_CIDR" ${pod_cidr}
}

# Adds an initial HNS network on the Windows node which forces the creation of
# a virtual switch and the "management" interface that will be used to
# communicate with the rest of the Kubernetes cluster without NAT.
#
# Note that adding the initial HNS network may cause connectivity to the GCE
# metadata server to be lost due to a Windows bug.
# Configure-HostNetworkingService() restores connectivity, look there for
# details.
#
# Download-HelperScripts() must have been called first.
function Add_InitialHnsNetwork {
  $INITIAL_HNS_NETWORK = 'External'

  # This comes from
  # https://github.com/Microsoft/SDN/blob/master/Kubernetes/flannel/l2bridge/start.ps1#L74
  # (or
  # https://github.com/Microsoft/SDN/blob/master/Kubernetes/windows/start-kubelet.ps1#L206).
  #
  # daschott noted on Slack: "L2bridge networks require an external vSwitch.
  # The first network ("External") with hardcoded values in the script is just
  # a placeholder to create an external vSwitch. This is purely for convenience
  # to be able to remove/modify the actual HNS network ("cbr0") or rejoin the
  # nodes without a network blip. Creating a vSwitch takes time, causes network
  # blips, and it makes it more likely to hit the issue where flanneld is
  # stuck, so we want to do this as rarely as possible."
  $hns_network = Get-HnsNetwork | Where-Object Name -eq $INITIAL_HNS_NETWORK
  if ($hns_network) {
    if ($REDO_STEPS) {
      Log-Output ("Warning: initial '$INITIAL_HNS_NETWORK' HNS network " +
                  "already exists, removing it and recreating it")
      $hns_network | Remove-HnsNetwork
      $hns_network = $null
    }
    else {
      Log-Output ("Skip: initial '$INITIAL_HNS_NETWORK' HNS network " +
                  "already exists, not recreating it")
      return
    }
  }
  Log-Output ("Creating initial HNS network to force creation of " +
              "${MGMT_ADAPTER_NAME} interface")
  # Note: RDP connection will hiccup when running this command.
  New-HNSNetwork `
      -Type "L2Bridge" `
      -AddressPrefix "192.168.255.0/30" `
      -Gateway "192.168.255.1" `
      -Name $INITIAL_HNS_NETWORK `
      -Verbose
}

# Get the network in uint32 for the given cidr
function Get_NetworkDecimal_From_CIDR([string] $cidr) {
  $network, [int]$subnetlen = $cidr.Split('/')
  $decimal_network = ConvertTo_DecimalIP($network)
  return $decimal_network
}

# Get gateway ip string (the first address) based on pod cidr.
# For Windows nodes the pod gateway IP address is the first address in the pod
# CIDR for the host.
function Get_Gateway_From_CIDR([string] $cidr) {
  $network=Get_NetworkDecimal_From_CIDR($cidr)
  $gateway=ConvertTo_DottedDecimalIP($network+1)
  return $gateway
}

# Get endpoint gateway ip string (the second address) based on pod cidr.
# For Windows nodes the pod gateway IP address is the first address in the pod
# CIDR for the host, but from inside containers it's the second address.
function Get_Endpoint_Gateway_From_CIDR([string] $cidr) {
  $network=Get_NetworkDecimal_From_CIDR($cidr)
  $gateway=ConvertTo_DottedDecimalIP($network+2)
  return $gateway
}

# Get pod IP range start based (the third address) on pod cidr
# We reserve the first two in the cidr range for gateways. Start the cidr
# range from the third so that IPAM does not allocate those IPs to pods.
function Get_PodIP_Range_Start([string] $cidr) {
  $network=Get_NetworkDecimal_From_CIDR($cidr)
  $start=ConvertTo_DottedDecimalIP($network+3)
  return $start
}

# Configures HNS on the Windows node to enable Kubernetes networking:
#   - Creates the "management" interface associated with an initial HNS network.
#   - Creates the HNS network $env:KUBE_NETWORK for pod networking.
#   - Creates an HNS endpoint for pod networking.
#   - Adds necessary routes on the management interface.
#   - Verifies that the GCE metadata server connection remains intact.
#
# Prerequisites:
#   $env:POD_CIDR is set (by Set-PodCidr).
#   Download-HelperScripts() has been called.
function Configure-HostNetworkingService {
  Import-Module -Force ${env:K8S_DIR}\hns.psm1

  Add_InitialHnsNetwork

  $pod_gateway = Get_Gateway_From_CIDR(${env:POD_CIDR})
  $pod_endpoint_gateway = Get_Endpoint_Gateway_From_CIDR(${env:POD_CIDR})
  Log-Output ("Setting up Windows node HNS networking: " +
              "podCidr = ${env:POD_CIDR}, podGateway = ${pod_gateway}, " +
              "podEndpointGateway = ${pod_endpoint_gateway}")

  $hns_network = Get-HnsNetwork | Where-Object Name -eq ${env:KUBE_NETWORK}
  if ($hns_network) {
    if ($REDO_STEPS) {
      Log-Output ("Warning: ${env:KUBE_NETWORK} HNS network already exists, " +
                  "removing it and recreating it")
      $hns_network | Remove-HnsNetwork
      $hns_network = $null
    }
    else {
      Log-Output "Skip: ${env:KUBE_NETWORK} HNS network already exists"
    }
  }
  $created_hns_network = $false
  if (-not $hns_network) {
    # Note: RDP connection will hiccup when running this command.
    $hns_network = New-HNSNetwork `
        -Type "L2Bridge" `
        -AddressPrefix ${env:POD_CIDR} `
        -Gateway ${pod_gateway} `
        -Name ${env:KUBE_NETWORK} `
        -Verbose
    $created_hns_network = $true
  }

  $endpoint_name = "cbr0"
  $vnic_name = "vEthernet (${endpoint_name})"

  $hns_endpoint = Get-HnsEndpoint | Where-Object Name -eq $endpoint_name
  # Note: we don't expect to ever enter this block currently - while the HNS
  # network does seem to persist across reboots, the HNS endpoints do not.
  if ($hns_endpoint) {
    if ($REDO_STEPS) {
      Log-Output ("Warning: HNS endpoint $endpoint_name already exists, " +
                  "removing it and recreating it")
      $hns_endpoint | Remove-HnsEndpoint
      $hns_endpoint = $null
    }
    else {
      Log-Output "Skip: HNS endpoint $endpoint_name already exists"
    }
  }
  if (-not $hns_endpoint) {
    $hns_endpoint = New-HnsEndpoint `
        -NetworkId ${hns_network}.Id `
        -Name ${endpoint_name} `
        -IPAddress ${pod_endpoint_gateway} `
        -Gateway "0.0.0.0" `
        -Verbose
    # TODO(pjh): find out: why is this always CompartmentId 1?
    Attach-HnsHostEndpoint `
        -EndpointID ${hns_endpoint}.Id `
        -CompartmentID 1 `
        -Verbose
    netsh interface ipv4 set interface "${vnic_name}" forwarding=enabled
  }

  Try {
    Get-HNSPolicyList | Remove-HnsPolicyList
  } Catch { }

  Apply_ACLPolicyForWorkloadIdentity ${hns_endpoint}.Id

  # Add a route from the management NIC to the pod CIDR.
  #
  # When a packet from a Kubernetes service backend arrives on the destination
  # Windows node, the reverse SNAT will be applied and the source address of
  # the packet gets replaced from the pod IP to the service VIP. The packet
  # will then leave the VM and return back through hairpinning.
  #
  # When IP alias is enabled, IP forwarding is disabled for anti-spoofing;
  # the packet with the service VIP will get blocked and be lost. With this
  # route, the packet will be routed to the pod subnetwork, and not leave the
  # VM.
  $mgmt_net_adapter = Get_MgmtNetAdapter
  New-NetRoute `
      -ErrorAction Ignore `
      -InterfaceAlias ${mgmt_net_adapter}.ifAlias `
      -DestinationPrefix ${env:POD_CIDR} `
      -NextHop "0.0.0.0" `
      -Verbose

  if ($created_hns_network) {
    # There is an HNS bug where the route to the GCE metadata server will be
    # removed when the HNS network is created:
    # https://github.com/Microsoft/hcsshim/issues/299#issuecomment-425491610.
    # The behavior here is very unpredictable: the route may only be removed
    # after some delay, or it may appear to be removed then you'll add it back
    # but then it will be removed once again. So, we first wait a long
    # unfortunate amount of time to ensure that things have quiesced, then we
    # wait until we're sure the route is really gone before re-adding it again.
    Log-Output "Waiting 45 seconds for host network state to quiesce"
    Start-Sleep 45
    WaitFor_GceMetadataServerRouteToBeRemoved
    Log-Output "Re-adding the GCE metadata server route"
    Add_GceMetadataServerRoute
  }
  Verify_GceMetadataServerRouteIsPresent

  Log-Output "Host network setup complete"
}

# When workload identity is enabled, gke-metadata-server will listen on POD_GATEWAY:988
# This access control list policy should be applied on POD_GATEWAY, so that only the pod
# on this node, whose ip is in POD_CIDR, can make TCP (Protocol 6) requests to POD_GATEWAY:988.
# Those ACL policies will be evaluated from smaller Priority to bigger Priority until it
# meets one entry and then the Action will be taken (Allow or Block).
# So those policies will only affect the "In" TCP requests to POD_GATEWAY:988.
function Apply_ACLPolicyForWorkloadIdentity {
  param (
    [parameter(Mandatory=$true)] [string]$HNSEndpointID
  )
  if (-not (Test-EnableWorkloadIdentity ${kube_env})) {
    Log-Output "Skipping applying access control list policies for workload identity as it's not enabled"
  }
  $acl_json = '{
    "Policies": [
        {
            "Type": "ACL",
            "Action": "Allow",
            "Direction": "In",
            "LocalAddresses": "",
            "RemoteAddresses": "POD_CIDR",
            "LocalPorts": "988",
            "Protocol": 6,
            "Priority": 200
        },
        {
            "Type": "ACL",
            "Action": "Block",
            "Direction": "In",
            "LocalPorts": "988",
            "Protocol": 6,
            "Priority": 300
        },
        {
            "Type": "ACL",
            "Action": "Allow",
            "Direction": "In",
            "Priority": 400
        },
        {
            "Type": "ACL",
            "Action": "Allow",
            "Direction": "Out",
            "Priority": 400
        }
    ]
  }'.replace('POD_CIDR', ${env:POD_CIDR})
  Log-Output 'Apply access control policy on hns endpoint id ${HNSEndpointID} for workload identity: $acl_json'
  Invoke-HNSRequest -Method POST -Type endpoints -Id ${HNSEndpointID} -Data $acl_json
}


function Configure-GcePdTools {
  if (ShouldWrite-File ${env:K8S_DIR}\GetGcePdName.dll) {
    MustDownload-File -OutFile ${env:K8S_DIR}\GetGcePdName.dll `
      -URLs "https://storage.googleapis.com/gke-release/winnode/config/gce-tools/master/GetGcePdName/GetGcePdName.dll"
  }
  if (-not (Test-Path $PsHome\profile.ps1)) {
    New-Item -path $PsHome\profile.ps1 -type file
  }

  Add-Content $PsHome\profile.ps1 `
  '$modulePath = "K8S_DIR\GetGcePdName.dll"
  Unblock-File $modulePath
  Import-Module -Name $modulePath'.replace('K8S_DIR', ${env:K8S_DIR})
}

# Setup cni network. This function supports both Docker and containerd.
function Prepare-CniNetworking {
  if (${env:CONTAINER_RUNTIME} -eq "containerd") {
    # For containerd the CNI binaries have already been installed along with
    # the runtime.
    Configure_Containerd_CniNetworking
  } else {
    Install_Cni_Binaries
    Configure_Dockerd_CniNetworking
  }
}

# Downloads the Windows CNI binaries and puts them in $env:CNI_DIR.
function Install_Cni_Binaries {
  if (-not (ShouldWrite-File ${env:CNI_DIR}\win-bridge.exe) -and
      -not (ShouldWrite-File ${env:CNI_DIR}\host-local.exe)) {
    return
  }

  $tmp_dir = 'C:\cni_tmp'
  New-Item $tmp_dir -ItemType 'directory' -Force | Out-Null

  $release_url = "${env:WINDOWS_CNI_STORAGE_PATH}/${env:WINDOWS_CNI_VERSION}/"
  $tgz_url = ($release_url +
              "cni-plugins-windows-amd64-${env:WINDOWS_CNI_VERSION}.tgz")
  $sha_url = ($tgz_url + ".sha512")
  MustDownload-File -URLs $sha_url -OutFile $tmp_dir\cni-plugins.sha512
  $sha512_val = ($(Get-Content $tmp_dir\cni-plugins.sha512) -split ' ',2)[0]
  MustDownload-File `
      -URLs $tgz_url `
      -OutFile $tmp_dir\cni-plugins.tgz `
      -Hash $sha512_val

  tar xzvf $tmp_dir\cni-plugins.tgz -C $tmp_dir
  Move-Item -Force $tmp_dir\host-local.exe ${env:CNI_DIR}\
  Move-Item -Force $tmp_dir\win-bridge.exe ${env:CNI_DIR}\
  Remove-Item -Force -Recurse $tmp_dir

  if (-not ((Test-Path ${env:CNI_DIR}\win-bridge.exe) -and `
            (Test-Path ${env:CNI_DIR}\host-local.exe))) {
    Log-Output `
        "win-bridge.exe and host-local.exe not found in ${env:CNI_DIR}" `
        -Fatal
  }
}

# Writes a CNI config file under $env:CNI_CONFIG_DIR.
#
# Prerequisites:
#   $env:POD_CIDR is set (by Set-PodCidr).
#   The "management" interface exists (Configure-HostNetworkingService).
#   The HNS network for pod networking has been configured
#     (Configure-HostNetworkingService).
#
# Required ${kube_env} keys:
#   DNS_SERVER_IP
#   DNS_DOMAIN
#   SERVICE_CLUSTER_IP_RANGE
function Configure_Dockerd_CniNetworking {
  $l2bridge_conf = "${env:CNI_CONFIG_DIR}\l2bridge.conf"
  if (-not (ShouldWrite-File ${l2bridge_conf})) {
    return
  }

  $mgmt_ip = (Get_MgmtNetAdapter |
              Get-NetIPAddress -AddressFamily IPv4).IPAddress

  $cidr_range_start = Get_PodIP_Range_Start(${env:POD_CIDR})

  $proxy_policy_for_workload_identity = Get_ProxyPolicyForWorkloadIdentity

  # Explanation of the CNI config values:
  #   POD_CIDR: the pod CIDR assigned to this node.
  #   CIDR_RANGE_START: start of the pod CIDR range.
  #   MGMT_IP: the IP address assigned to the node's primary network interface
  #     (i.e. the internal IP of the GCE VM).
  #   SERVICE_CIDR: the CIDR used for kubernetes services.
  #   DNS_SERVER_IP: the cluster's DNS server IP address.
  #   DNS_DOMAIN: the cluster's DNS domain, e.g. "cluster.local".
  #
  # OutBoundNAT ExceptionList: No SNAT for CIDRs in the list, the same as default GKE non-masquerade destination ranges listed at https://cloud.google.com/kubernetes-engine/docs/how-to/ip-masquerade-agent#default-non-masq-dests

  New-Item -Force -ItemType file ${l2bridge_conf} | Out-Null
  Set-Content ${l2bridge_conf} `
'{
  "cniVersion":  "0.2.0",
  "name":  "l2bridge",
  "type":  "win-bridge",
  "capabilities":  {
    "portMappings":  true,
    "dns": true
  },
  "ipam":  {
    "type": "host-local",
    "subnet": "POD_CIDR",
    "rangeStart": "CIDR_RANGE_START"
  },
  "dns":  {
    "Nameservers":  [
      "DNS_SERVER_IP"
    ],
    "Search": [
      "DNS_DOMAIN"
    ]
  },
  "Policies":  [
    PROXY_POLICY_FOR_WORKLOAD_IDENTITY
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "OutBoundNAT",
        "ExceptionList":  [
          "169.254.0.0/16",
          "10.0.0.0/8",
          "172.16.0.0/12",
          "192.168.0.0/16",
          "100.64.0.0/10",
          "192.0.0.0/24",
          "192.0.2.0/24",
          "192.88.99.0/24",
          "198.18.0.0/15",
          "198.51.100.0/24",
          "203.0.113.0/24",
          "240.0.0.0/4"
        ]
      }
    },
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "ROUTE",
        "DestinationPrefix":  "SERVICE_CIDR",
        "NeedEncap":  true
      }
    },
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "ROUTE",
        "DestinationPrefix":  "MGMT_IP/32",
        "NeedEncap":  true
      }
    }
  ]
}'.replace('POD_CIDR', ${env:POD_CIDR}).`
  replace('CIDR_RANGE_START', ${cidr_range_start}).`
  replace('DNS_SERVER_IP', ${kube_env}['DNS_SERVER_IP']).`
  replace('DNS_DOMAIN', ${kube_env}['DNS_DOMAIN']).`
  replace('MGMT_IP', ${mgmt_ip}).`
  replace('SERVICE_CIDR', ${kube_env}['SERVICE_CLUSTER_IP_RANGE']).`
  replace('PROXY_POLICY_FOR_WORKLOAD_IDENTITY', ${proxy_policy_for_workload_identity})

  Log-Output "CNI config:`n$(Get-Content -Raw ${l2bridge_conf})"
}

# When workload identity is enabled, gke-metadata-server will listen on POD_GATEWAY:988
# This proxy policy will be in the CNI config, which will be applied to every pod on this node.
# The request to 169.254.169.254:80 (GCE Metadata) will be redirected to gke-metadata-server.
function Get_ProxyPolicyForWorkloadIdentity {
  if (-not (Test-EnableWorkloadIdentity ${kube_env})) {
    Log-Output "Workload identity is not enabled. No proxy policy is needed"
    return ""
  }

  $pod_gateway = Get_Endpoint_Gateway_From_CIDR ${env:POD_CIDR}

  return '{
      "Name":  "EndpointPolicy",
      "Value":  {
         "Type": "PROXY",
         "IP": "GCE_METADATA_SERVER",
         "Port": "80",
         "Destination": "POD_GATEWAY:988"
      }
    },'.replace('POD_GATEWAY', ${pod_gateway}).`
       replace('GCE_METADATA_SERVER', $GCE_METADATA_SERVER)
}

# Obtain the host dns conf and save it to a file so that kubelet/CNI
# can use it to configure dns suffix search list for pods.
# The value of DNS server is ignored right now because the pod will
# always only use cluster DNS service, but for consistency, we still
# parsed them here in the same format as Linux resolv.conf.
# This function must be called after Configure-HostNetworkingService.
function Configure-HostDnsConf {
  $net_adapter = Get_MgmtNetAdapter
  $server_ips = (Get-DnsClientServerAddress `
          -InterfaceAlias ${net_adapter}.Name).ServerAddresses
  $search_list = (Get-DnsClient).ConnectionSpecificSuffixSearchList
  $conf = ""
  ForEach ($ip in $server_ips)  {
    $conf = $conf + "nameserver $ip`r`n"
  }
  $conf = $conf + "search $search_list"
  # Do not put hostdns.conf into the CNI config directory so as to
  # avoid the container runtime treating it as CNI config.
  $hostdns_conf = "${env:CNI_DIR}\hostdns.conf"
  New-Item -Force -ItemType file ${hostdns_conf} | Out-Null
  Set-Content ${hostdns_conf} $conf
  Log-Output "HOST dns conf:`n$(Get-Content -Raw ${hostdns_conf})"
}

# Fetches the kubelet config from the instance metadata and puts it at
# $env:KUBELET_CONFIG.
function Configure-Kubelet {
  if (-not (ShouldWrite-File ${env:KUBELET_CONFIG})) {
    return
  }

  # The Kubelet config is built by build-kubelet-config() in
  # cluster/gce/util.sh, and stored in the metadata server under the
  # 'kubelet-config' key.
  $kubelet_config = Get-InstanceMetadataAttribute 'kubelet-config'
  Set-Content ${env:KUBELET_CONFIG} $kubelet_config
  Log-Output "Kubelet config:`n$(Get-Content -Raw ${env:KUBELET_CONFIG})"
}

# Sets up the kubelet and kube-proxy arguments and starts them as native
# Windows services.
#
# Required ${kube_env} keys:
#   KUBELET_ARGS
#   KUBEPROXY_ARGS
#   CLUSTER_IP_RANGE
function Start-WorkerServices {
  # Compute kubelet args
  $kubelet_args_str = ${kube_env}['KUBELET_ARGS']
  $kubelet_args = $kubelet_args_str.Split(" ")
  Log-Output "kubelet_args from metadata: ${kubelet_args}"

  # To join GCE instances to AD, we need to shorten their names, as NetBIOS name
  # must be <= 15 characters, and GKE generated names are longer than that.
  # To perform the join in an automated way, it's preferable to apply the rename
  # and domain join in the GCESysprep step. However, after sysprep is complete
  # and the machine restarts, kubelet bootstrapping should not use the shortened
  # computer name, and instead use the instance's name by using --hostname-override,
  # otherwise kubelet and kube-proxy will not be able to run properly.
  $instance_name = "$(Get-InstanceMetadata 'name' | Out-String)"
  $default_kubelet_args = @(`
      "--pod-infra-container-image=${env:INFRA_CONTAINER}",
      "--hostname-override=${instance_name}"
  )

  $kubelet_args = ${default_kubelet_args} + ${kubelet_args}
  if (-not (Test-NodeUsesAuthPlugin ${kube_env})) {
    Log-Output 'Using bootstrap kubeconfig for authentication'
    $kubelet_args = (${kubelet_args} +
                     "--bootstrap-kubeconfig=${env:BOOTSTRAP_KUBECONFIG}")
  }
  Log-Output "Final kubelet_args: ${kubelet_args}"

  # Compute kube-proxy args
  $kubeproxy_args_str = ${kube_env}['KUBEPROXY_ARGS']
  $kubeproxy_args = $kubeproxy_args_str.Split(" ")
  Log-Output "kubeproxy_args from metadata: ${kubeproxy_args}"

  # kubeproxy is started on Linux nodes using
  # kube-manifests/kubernetes/gci-trusty/kube-proxy.manifest, which is
  # generated by start-kube-proxy in configure-helper.sh and contains e.g.:
  #   kube-proxy --master=https://35.239.84.171
  #   --kubeconfig=/var/lib/kube-proxy/kubeconfig --cluster-cidr=10.64.0.0/14
  #   --oom-score-adj=-998 --v=2
  #   --iptables-sync-period=1m --iptables-min-sync-period=10s
  #   --ipvs-sync-period=1m --ipvs-min-sync-period=10s
  # And also with various volumeMounts and "securityContext: privileged: true".
  $default_kubeproxy_args = @(`
      "--kubeconfig=${env:KUBEPROXY_KUBECONFIG}",
      "--cluster-cidr=$(${kube_env}['CLUSTER_IP_RANGE'])",
      "--hostname-override=${instance_name}"
  )

  $kubeproxy_args = ${default_kubeproxy_args} + ${kubeproxy_args}
  Log-Output "Final kubeproxy_args: ${kubeproxy_args}"

  # TODO(pjh): kubelet is emitting these messages:
  # I1023 23:44:11.761915    2468 kubelet.go:274] Adding pod path:
  # C:\etc\kubernetes
  # I1023 23:44:11.775601    2468 file.go:68] Watching path
  # "C:\\etc\\kubernetes"
  # ...
  # E1023 23:44:31.794327    2468 file.go:182] Can't process manifest file
  # "C:\\etc\\kubernetes\\hns.psm1": C:\etc\kubernetes\hns.psm1: couldn't parse
  # as pod(yaml: line 10: did not find expected <document start>), please check
  # config file.
  #
  # Figure out how to change the directory that the kubelet monitors for new
  # pod manifests.

  # We configure the service to restart on failure, after 10s wait. We reset
  # the restart count to 0 each time, so we re-use our restart/10000 action on
  # each failure. Note it currently restarts even when explicitly stopped, you
  # have to delete the service entry to *really* kill it (e.g. `sc.exe delete
  # kubelet`). See issue #72900.
  if (Get-Process | Where-Object Name -eq "kubelet") {
    Log-Output -Fatal `
        "A kubelet process is already running, don't know what to do"
  }
  Log-Output "Creating kubelet service"
  & sc.exe create kubelet binPath= "${env:NODE_DIR}\kubelet.exe ${kubelet_args}" start= demand
  & sc.exe failure kubelet reset= 0 actions= restart/10000
  Log-Output "Starting kubelet service"
  & sc.exe start kubelet

  Log-Output "Waiting 10 seconds for kubelet to stabilize"
  Start-Sleep 10
  Write-VerboseServiceInfoToConsole -Service 'kubelet'

  if (Get-Process | Where-Object Name -eq "kube-proxy") {
    Log-Output -Fatal `
        "A kube-proxy process is already running, don't know what to do"
  }
  Log-Output "Creating kube-proxy service"
  & sc.exe create kube-proxy binPath= "${env:NODE_DIR}\kube-proxy.exe ${kubeproxy_args}" start= demand
  & sc.exe failure kube-proxy reset= 0 actions= restart/10000
  Log-Output "Starting kube-proxy service"
  & sc.exe start kube-proxy
  Write-VerboseServiceInfoToConsole -Service 'kube-proxy' -Delay 1

  # F1020 23:08:52.000083    9136 server.go:361] unable to load in-cluster
  # configuration, KUBERNETES_SERVICE_HOST and KUBERNETES_SERVICE_PORT must be
  # defined
  # TODO(pjh): still getting errors like these in kube-proxy log:
  # E1023 04:03:58.143449    4840 reflector.go:205] k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/factory.go:129: Failed to list *core.Endpoints: Get https://35.239.84.171/api/v1/endpoints?limit=500&resourceVersion=0: dial tcp 35.239.84.171:443: connectex: A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond.
  # E1023 04:03:58.150266    4840 reflector.go:205] k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/factory.go:129: Failed to list *core.Service: Get https://35.239.84.171/api/v1/services?limit=500&resourceVersion=0: dial tcp 35.239.84.171:443: connectex: A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond.
  WaitFor_KubeletAndKubeProxyReady
  Verify_GceMetadataServerRouteIsPresent
  Log-Output "Kubernetes components started successfully"
}

# Wait for kubelet and kube-proxy to be ready within 10s.
function WaitFor_KubeletAndKubeProxyReady {
  $waited = 0
  $timeout = 10
  while (((Get-Service kube-proxy).Status -ne 'Running' -or (Get-Service kubelet).Status -ne 'Running') -and $waited -lt $timeout) {
    Start-Sleep 1
    $waited++
  }

  # Timeout occurred
  if ($waited -ge $timeout) {
    Log-Output "$(Get-Service kube* | Out-String)"
    Throw ("Timeout while waiting ${timeout} seconds for kubelet and kube-proxy services to start")
  }
}

# Runs 'kubectl get nodes'.
# TODO(pjh): run more verification commands.
function Verify-WorkerServices {
  Log-Output ("kubectl get nodes:`n" +
              $(& "${env:NODE_DIR}\kubectl.exe" get nodes | Out-String))
  Verify_GceMetadataServerRouteIsPresent
  Log_Todo "run more verification commands."
}

# Downloads the Windows crictl package and installs its contents (e.g.
# crictl.exe) in $env:NODE_DIR.
function DownloadAndInstall-Crictl {
  if (-not (ShouldWrite-File ${env:NODE_DIR}\crictl.exe)) {
    return
  }
  $CRI_TOOLS_GCS_BUCKET = 'k8s-artifacts-cri-tools'
  $url = ('https://storage.googleapis.com/' + $CRI_TOOLS_GCS_BUCKET +
          '/release/' + $CRICTL_VERSION + '/crictl-' + $CRICTL_VERSION +
          '-windows-amd64.tar.gz')
  MustDownload-File `
      -URLs $url `
      -OutFile ${env:NODE_DIR}\crictl.tar.gz `
      -Hash $CRICTL_SHA256 `
      -Algorithm SHA256
  tar xzvf ${env:NODE_DIR}\crictl.tar.gz -C ${env:NODE_DIR}
}

# Sets crictl configuration values.
function Configure-Crictl {
  if (${env:CONTAINER_RUNTIME_ENDPOINT}) {
    & "${env:NODE_DIR}\crictl.exe" config runtime-endpoint `
        ${env:CONTAINER_RUNTIME_ENDPOINT}
  }
}

# Pulls the infra/pause container image onto the node so that it will be
# immediately available when the kubelet tries to run pods.
# TODO(pjh): downloading the container container image may take a few minutes;
# figure out how to run this in the background while perform the rest of the
# node startup steps!
# Pull-InfraContainer must be called AFTER Verify-WorkerServices.
function Pull-InfraContainer {
  $name, $label = ${env:INFRA_CONTAINER} -split ':',2
  if (-not ("$(& crictl images)" -match "$name.*$label")) {
    & crictl pull ${env:INFRA_CONTAINER}
    if (!$?) {
      throw "Error running 'crictl pull ${env:INFRA_CONTAINER}'"
    }
  }
  $inspect = "$(& crictl inspecti ${env:INFRA_CONTAINER} | Out-String)"
  Log-Output "Infra/pause container:`n$inspect"
}

# Setup the container runtime on the node. It supports both
# Docker and containerd.
function Setup-ContainerRuntime {
  Install-Pigz
  if (${env:CONTAINER_RUNTIME} -eq "containerd") {
    Install_Containerd
    Configure_Containerd
    Start_Containerd
  } else {
    Create_DockerRegistryKey
    Configure_Dockerd
  }
}

function Test-ContainersFeatureInstalled {
  return (Get-WindowsFeature Containers).Installed
}

# After this function returns, the computer must be restarted to complete
# the installation!
function Install-ContainersFeature {
  Log-Output "Installing Windows 'Containers' feature"
  Install-WindowsFeature Containers
}

function Test-DockerIsInstalled {
  return ((Get-Package `
               -ProviderName DockerMsftProvider `
               -ErrorAction SilentlyContinue |
           Where-Object Name -eq 'docker') -ne $null)
}

function Test-DockerIsRunning {
  return ((Get-Service docker).Status -eq 'Running')
}

# Installs Docker EE via the DockerMsftProvider. Ensure that the Windows
# Containers feature is installed before calling this function; otherwise,
# a restart may be needed after this function returns.
function Install-Docker {
  Log-Output 'Installing NuGet module'
  Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force

  Log-Output 'Installing DockerMsftProvider module'
  Install-Module -Name DockerMsftProvider -Repository PSGallery -Force

  Log-Output "Installing latest Docker EE version"
  Install-Package `
      -Name docker `
      -ProviderName DockerMsftProvider `
      -Force `
      -Verbose
}

# Add a registry key for docker in EventLog so that log messages are mapped
# correctly. This is a workaround since the key is missing in the base image.
# https://github.com/MicrosoftDocs/Virtualization-Documentation/pull/503
# TODO: Fix this in the base image.
# TODO(random-liu): Figure out whether we need this for containerd.
function Create_DockerRegistryKey {
  $tmp_dir = 'C:\tmp_docker_reg'
  New-Item -Force -ItemType 'directory' ${tmp_dir} | Out-Null
  $reg_file = 'docker.reg'
  Set-Content ${tmp_dir}\${reg_file} `
'Windows Registry Editor Version 5.00
 [HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\EventLog\Application\docker]
"CustomSource"=dword:00000001
"EventMessageFile"="C:\\Program Files\\docker\\dockerd.exe"
"TypesSupported"=dword:00000007'

  Log-Output "Importing registry key for Docker"
  reg import ${tmp_dir}\${reg_file}
  Remove-Item -Force -Recurse ${tmp_dir}
}

# Configure Docker daemon and restart the service.
function Configure_Dockerd {
  Set-Content "C:\ProgramData\docker\config\daemon.json" @'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "1m",
    "max-file": "5"
  }
}
'@

 Restart-Service Docker
}

# Configures the TCP/IP parameters to be in sync with the GCP recommendation.
# Not setting these values correctly can cause network issues for connections
# that live longer than 10 minutes.
# See: https://cloud.google.com/compute/docs/troubleshooting/general-tips#idle-connections
function Set-WindowsTCPParameters {
  Set-ItemProperty -Force -Confirm:$false -Path `
    'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters' `
    -Name 'KeepAliveInterval' -Type Dword -Value 1000
  Set-ItemProperty -Force -Confirm:$false `
    -Path 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters' `
    -Name 'KeepAliveTime' -Type Dword -Value 60000
  Set-ItemProperty -Force -Confirm:$false `
    -Path 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters' `
    -Name 'TcpMaxDataRetransmissions' -Type Dword -Value 10

  Log-Output 'TCP/IP Parameters'
  Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters'
}

# Writes a CNI config file under $env:CNI_CONFIG_DIR for containerd.
#
# Prerequisites:
#   $env:POD_CIDR is set (by Set-PodCidr).
#   The "management" interface exists (Configure-HostNetworkingService).
#   The HNS network for pod networking has been configured
#     (Configure-HostNetworkingService).
#   Containerd is installed (Install_Containerd).
#
# Required ${kube_env} keys:
#   DNS_SERVER_IP
#   DNS_DOMAIN
#   SERVICE_CLUSTER_IP_RANGE
function Configure_Containerd_CniNetworking {
  $l2bridge_conf = "${env:CNI_CONFIG_DIR}\l2bridge.conf"
  if (-not (ShouldWrite-File ${l2bridge_conf})) {
    return
  }

  $mgmt_ip = (Get_MgmtNetAdapter |
              Get-NetIPAddress -AddressFamily IPv4).IPAddress

  $pod_gateway = Get_Endpoint_Gateway_From_CIDR(${env:POD_CIDR})

  # Explanation of the CNI config values:
  #   POD_CIDR: the pod CIDR assigned to this node.
  #   POD_GATEWAY: the gateway IP.
  #   MGMT_IP: the IP address assigned to the node's primary network interface
  #     (i.e. the internal IP of the GCE VM).
  #   SERVICE_CIDR: the CIDR used for kubernetes services.
  #   DNS_SERVER_IP: the cluster's DNS server IP address.
  #   DNS_DOMAIN: the cluster's DNS domain, e.g. "cluster.local".
  #
  # OutBoundNAT ExceptionList: No SNAT for CIDRs in the list, the same as default GKE non-masquerade destination ranges listed at https://cloud.google.com/kubernetes-engine/docs/how-to/ip-masquerade-agent#default-non-masq-dests

  New-Item -Force -ItemType file ${l2bridge_conf} | Out-Null
  Set-Content ${l2bridge_conf} `
'{
  "cniVersion":  "0.2.0",
  "name":  "l2bridge",
  "type":  "sdnbridge",
  "master": "Ethernet",
  "capabilities":  {
    "portMappings":  true,
    "dns": true
  },
  "ipam":  {
    "subnet": "POD_CIDR",
    "routes": [
      {
        "GW": "POD_GATEWAY"
      }
    ]
  },
  "dns":  {
    "Nameservers":  [
      "DNS_SERVER_IP"
    ],
    "Search": [
      "DNS_DOMAIN"
    ]
  },
  "AdditionalArgs": [
    PROXY_POLICY_FOR_WORKLOAD_IDENTITY
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "OutBoundNAT",
        "Settings": {
          "Exceptions":  [
            "169.254.0.0/16",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "100.64.0.0/10",
            "192.0.0.0/24",
            "192.0.2.0/24",
            "192.88.99.0/24",
            "198.18.0.0/15",
            "198.51.100.0/24",
            "203.0.113.0/24",
            "240.0.0.0/4"
          ]
        }
      }
    },
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "SDNRoute",
        "Settings": {
          "DestinationPrefix":  "SERVICE_CIDR",
          "NeedEncap":  true
        }
      }
    },
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "SDNRoute",
        "Settings": {
          "DestinationPrefix":  "MGMT_IP/32",
          "NeedEncap":  true
        }
      }
    }
  ]
}'.replace('POD_CIDR', ${env:POD_CIDR}).`
  replace('POD_GATEWAY', ${pod_gateway}).`
  replace('DNS_SERVER_IP', ${kube_env}['DNS_SERVER_IP']).`
  replace('DNS_DOMAIN', ${kube_env}['DNS_DOMAIN']).`
  replace('MGMT_IP', ${mgmt_ip}).`
  replace('SERVICE_CIDR', ${kube_env}['SERVICE_CLUSTER_IP_RANGE']).`
  replace('PROXY_POLICY_FOR_WORKLOAD_IDENTITY', $(Get_ProxyPolicyForWorkloadIdentity))

  Log-Output "containerd CNI config:`n$(Get-Content -Raw ${l2bridge_conf})"
}

# Download and install containerd and CNI binaries into $env:NODE_DIR.
function Install_Containerd {
  # Assume that presence of containerd.exe indicates that all containerd
  # binaries were already previously downloaded to this node.
  if (-not (ShouldWrite-File ${env:NODE_DIR}\containerd.exe)) {
    return
  }

  $tmp_dir = 'C:\containerd_tmp'
  New-Item $tmp_dir -ItemType 'directory' -Force | Out-Null

  # TODO(ibrahimab) Change this to a gcs bucket with CI maintained and accessible by community.
  $version = '1.4.4'
  $tar_url = ("https://github.com/containerd/containerd/releases/download/v${version}/" +
              "cri-containerd-cni-${version}-windows-amd64.tar.gz")
  $sha_url = $tar_url + ".sha256sum"
  MustDownload-File -URLs $sha_url -OutFile $tmp_dir\sha256sum
  $sha = $(Get-Content $tmp_dir\sha256sum).Split(" ")[0].ToUpper()

  MustDownload-File `
      -URLs $tar_url `
      -OutFile $tmp_dir\containerd.tar.gz `
      -Hash $sha `
      -Algorithm SHA256

  tar xzvf $tmp_dir\containerd.tar.gz -C $tmp_dir
  Move-Item -Force $tmp_dir\cni\*.exe "${env:CNI_DIR}\"
  Move-Item -Force $tmp_dir\*.exe "${env:NODE_DIR}\"
  Remove-Item -Force -Recurse $tmp_dir

  # Exclusion for Defender.
  Add-MpPreference -ExclusionProcess "${env:NODE_DIR}\containerd.exe"
}

# Lookup the path of containerd config if exists, else returns a default.
function Get_Containerd_ConfigPath {
  $service = Get-WMIObject -Class Win32_Service -Filter  "Name='containerd'"
  if (!($service -eq $null) -and
      $service.PathName -match ".*\s--config\s*(\S+).*" -and
      $matches.Count -eq 2) {
    return $matches[1]
  } else {
    return 'C:\Program Files\containerd\config.toml'
  }
}

# Generates the containerd config.toml file.
function Configure_Containerd {
  $config_path = Get_Containerd_ConfigPath
  $config_dir = [System.IO.Path]::GetDirectoryName($config_path)
  New-Item $config_dir -ItemType 'directory' -Force | Out-Null
  Set-Content ${config_path} @"
[plugins.scheduler]
  schedule_delay = '0s'
  startup_delay = '0s'
[plugins.cri]
  sandbox_image = 'INFRA_CONTAINER_IMAGE'
[plugins.cri.containerd]
  snapshotter = 'windows'
  default_runtime_name = 'runhcs-wcow-process'
  disable_snapshot_annotations = true
  discard_unpacked_layers = true
[plugins.cri.cni]
  bin_dir = 'CNI_BIN_DIR'
  conf_dir = 'CNI_CONF_DIR'
"@.replace('INFRA_CONTAINER_IMAGE', ${env:INFRA_CONTAINER}).`
    replace('CNI_BIN_DIR', "${env:CNI_DIR}").`
    replace('CNI_CONF_DIR', "${env:CNI_CONFIG_DIR}")
}

# Register if needed and start containerd service.
function Start_Containerd {
  # Do the registration only if the containerd service does not exist.
  if ((Get-WMIObject -Class Win32_Service -Filter  "Name='containerd'") -eq $null) {
    Log-Output "Creating containerd service"
    & containerd.exe --register-service --log-file "${env:LOGS_DIR}/containerd.log"
  }

  Log-Output "Starting containerd service"
  Restart-Service containerd
}

# Pigz Resources
$PIGZ_ROOT = 'C:\pigz'
$PIGZ_VERSION = '2.3.1'
$PIGZ_TAR_URL = "https://storage.googleapis.com/gke-release/winnode/pigz/prod/gke_windows/pigz/release/5/20201104-134221/pigz-$PIGZ_VERSION.zip"
$PIGZ_TAR_HASH = '5a6f8f5530acc85ea51797f58c1409e5af6b69e55da243ffc608784cf14fec0cd16f74cc61c564d69e1a267750aecfc1e4c53b5219ff5f893b42a7576306f34c'

# Install Pigz (https://github.com/madler/pigz) into Windows for improved image
# extraction performance.
function Install-Pigz {
  if ("${env:WINDOWS_ENABLE_PIGZ}" -eq "true") {
    if (-not (Test-Path $PIGZ_ROOT)) {
      Log-Output "Installing Pigz $PIGZ_VERSION"
      New-Item -Path $PIGZ_ROOT -ItemType Directory
      MustDownload-File `
        -Url $PIGZ_TAR_URL `
        -OutFile "$PIGZ_ROOT\pigz-$PIGZ_VERSION.zip" `
        -Hash $PIGZ_TAR_HASH `
        -Algorithm SHA512
      Expand-Archive -Path "$PIGZ_ROOT\pigz-$PIGZ_VERSION.zip" `
        -DestinationPath $PIGZ_ROOT
      Remove-Item -Path "$PIGZ_ROOT\pigz-$PIGZ_VERSION.zip"
      # Docker and Containerd search for unpigz.exe on the first container image
      # pull request after the service is started. If unpigz.exe is in the
      # Windows path it'll use it instead of the default unzipper.
      # See: https://github.com/containerd/containerd/issues/1896
      Add-MachineEnvironmentPath -Path $PIGZ_ROOT
      # Add process exclusion for Windows Defender to boost performance.
      Add-MpPreference -ExclusionProcess "$PIGZ_ROOT\unpigz.exe"
      Log-Output "Installed Pigz $PIGZ_VERSION"
    } else {
      Log-Output "Pigz already installed."
    }
  }
}

# Node Problem Detector Resources
$NPD_SERVICE = "node-problem-detector"
$DEFAULT_NPD_VERSION = '0.8.9'
$DEFAULT_NPD_RELEASE_PATH = 'https://storage.googleapis.com/kubernetes-release'
$DEFAULT_NPD_HASH = 'f87a01d344e16d16511c5160d7d0f3b8a694e00452f50ffbb550deb10460a1ffa7e170233401ae11df5d58cf070e746feea3bd0407af95ba00d6362630da620c'

# Install Node Problem Detector (NPD).
# NPD analyzes the host for problems that can disrupt workloads.
# https://github.com/kubernetes/node-problem-detector
function DownloadAndInstall-NodeProblemDetector {
  if ("${env:ENABLE_NODE_PROBLEM_DETECTOR}" -eq "standalone") {
    if (ShouldWrite-File "${env:NODE_DIR}\node-problem-detector.exe") {
      $npd_version = $DEFAULT_NPD_VERSION
      $npd_hash = $DEFAULT_NPD_HASH
      if (-not [string]::IsNullOrEmpty(${kube_env}['NODE_PROBLEM_DETECTOR_VERSION'])) {
        $npd_version = ${kube_env}['NODE_PROBLEM_DETECTOR_VERSION']
        $npd_hash = ${kube_env}['NODE_PROBLEM_DETECTOR_TAR_HASH']
      }
      $npd_release_path = $DEFAULT_NPD_RELEASE_PATH
      if (-not [string]::IsNullOrEmpty(${kube_env}['NODE_PROBLEM_DETECTOR_RELEASE_PATH'])) {
        $npd_release_path = ${kube_env}['NODE_PROBLEM_DETECTOR_RELEASE_PATH']
      }

      $npd_tar = "node-problem-detector-v${npd_version}-windows_amd64.tar.gz"

      Log-Output "Downloading ${npd_tar}."

      $npd_dir = "${env:K8S_DIR}\node-problem-detector"
      New-Item -Path $npd_dir -ItemType Directory -Force -Confirm:$false

      MustDownload-File `
          -URLs "${npd_release_path}/node-problem-detector/${npd_tar}" `
          -Hash $npd_hash `
          -Algorithm SHA512 `
          -OutFile "${npd_dir}\${npd_tar}"

      tar xzvf "${npd_dir}\${npd_tar}" -C $npd_dir
      Move-Item "${npd_dir}\bin\*" "${env:NODE_DIR}\" -Force -Confirm:$false
      Remove-Item "${npd_dir}\bin" -Force -Confirm:$false
      Remove-Item "${npd_dir}\${npd_tar}" -Force -Confirm:$false
    }
    else {
        Log-Output "Node Problem Detector already installed."
    }
  }
}

# Creates the node-problem-detector user kubeconfig file at
# $env:NODEPROBLEMDETECTOR_KUBECONFIG_FILE (if defined).
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   CA_CERT
#   NODE_PROBLEM_DETECTOR_TOKEN
function Create-NodeProblemDetectorKubeConfig {
  if (-not [string]::IsNullOrEmpty(${env:NODEPROBLEMDETECTOR_KUBECONFIG_FILE})) {
    Create-Kubeconfig -Name 'node-problem-detector' `
      -Path ${env:NODEPROBLEMDETECTOR_KUBECONFIG_FILE} `
      -Token ${kube_env}['NODE_PROBLEM_DETECTOR_TOKEN']
  }
}

# Configures NPD to run with the bundled monitor configs and report against the Kubernetes api server.
function Configure-NodeProblemDetector {
  $npd_bin = "${env:NODE_DIR}\node-problem-detector.exe"
  if ("${env:ENABLE_NODE_PROBLEM_DETECTOR}" -eq "standalone" -and (Test-Path $npd_bin)) {
    $npd_svc = Get-Service -Name $NPD_SERVICE -ErrorAction SilentlyContinue
    if ($npd_svc -eq $null) {
      $npd_dir = "${env:K8S_DIR}\node-problem-detector"
      $npd_logs_dir = "${env:LOGS_DIR}\node-problem-detector"

      New-Item -Path $npd_logs_dir -Type Directory -Force -Confirm:$false

      $flags = ''
      if ([string]::IsNullOrEmpty(${kube_env}['NODE_PROBLEM_DETECTOR_CUSTOM_FLAGS'])) {
        $system_log_monitors = @()
        $system_stats_monitors = @()
        $custom_plugin_monitors = @()

        # Custom Plugin Monitors
        $custom_plugin_monitors += @("${npd_dir}\config\windows-health-checker-kubelet.json")
        $custom_plugin_monitors += @("${npd_dir}\config\windows-health-checker-kubeproxy.json")
        $custom_plugin_monitors += @("${npd_dir}\config\windows-defender-monitor.json")

        # System Stats Monitors
        $system_stats_monitors += @("${npd_dir}\config\windows-system-stats-monitor.json")

        # NPD Configuration for CRI monitor
        if (${env:CONTAINER_RUNTIME} -eq "containerd") {
          $system_log_monitors += @("${npd_dir}\config\windows-containerd-monitor-filelog.json")
          $custom_plugin_monitors += @("${npd_dir}\config\windows-health-checker-containerd.json")
        } else {
          $custom_plugin_monitors += @("${npd_dir}\config\windows-health-checker-docker.json")
        }

        $flags="--v=2 --port=20256 --log_dir=${npd_logs_dir}"
        if ($system_log_monitors.count -gt 0) {
          $flags+=" --config.system-log-monitor={0}" -f ($system_log_monitors -join ",")
        }
        if ($system_stats_monitors.count -gt 0) {
          $flags+=" --config.system-stats-monitor={0}" -f ($system_stats_monitors -join ",")
        }
        if ($custom_plugin_monitors.count -gt 0) {
          $flags+=" --config.custom-plugin-monitor={0}" -f ($custom_plugin_monitors -join ",")
        }
      }
      else {
        $flags = ${kube_env}['NODE_PROBLEM_DETECTOR_CUSTOM_FLAGS']
      }
      $kubernetes_master_name = ${kube_env}['KUBERNETES_MASTER_NAME']
      $flags = "${flags} --apiserver-override=`"https://${kubernetes_master_name}?inClusterConfig=false&auth=${env:NODEPROBLEMDETECTOR_KUBECONFIG_FILE}`""

      Log-Output "Creating service: ${NPD_SERVICE}"
      Log-Output "${npd_bin} ${flags}"
      sc.exe create $NPD_SERVICE binpath= "${npd_bin} ${flags}" displayName= "Node Problem Detector"
      sc.exe failure $NPD_SERVICE reset= 30 actions= restart/5000
      sc.exe start $NPD_SERVICE

      Write-VerboseServiceInfoToConsole -Service $NPD_SERVICE
    }
    else {
      Log-Output "${NPD_SERVICE} already configured."
    }
  }
}

# TODO(pjh): move the logging agent code below into a separate
# module; it was put here temporarily to avoid disrupting the file layout in
# the K8s release machinery.
$LOGGINGAGENT_VERSION = '1.7.7'
$LOGGINGAGENT_ROOT = 'C:\fluent-bit'
$LOGGINGAGENT_SERVICE = 'fluent-bit'
$LOGGINGAGENT_CMDLINE = '*fluent-bit.exe*'

$LOGGINGEXPORTER_VERSION = 'v0.17.0'
$LOGGINGEXPORTER_ROOT = 'C:\flb-exporter'
$LOGGINGEXPORTER_SERVICE = 'flb-exporter'
$LOGGINGEXPORTER_CMDLINE = '*flb-exporter.exe*'
$LOGGINGEXPORTER_HASH = 'c808c9645d84b06b89932bd707d51a9d1d0b451b5a702a5f9b2b4462c8be6502'

# Restart Logging agent or starts it if it is not currently running
function Restart-LoggingAgent {
  if (IsStackdriverAgentInstalled) {
      Restart-StackdriverAgent
      return
  }

   Restart-LogService $LOGGINGEXPORTER_SERVICE $LOGGINGEXPORTER_CMDLINE
   Restart-LogService $LOGGINGAGENT_SERVICE $LOGGINGAGENT_CMDLINE
}

# Restarts the service, or starts it if it is not currently
# running. A standard `Restart-Service` may fail because
# the process is sometimes unstoppable, so this function works around it
# by killing the processes.
function Restart-LogService([string]$service, [string]$cmdline) {
  Stop-Service -NoWait -ErrorAction Ignore $service

  # Wait (if necessary) for service to stop.
  $timeout = 10
  $stopped = (Get-service $service).Status -eq 'Stopped'
  for ($i = 0; $i -lt $timeout -and !($stopped); $i++) {
      Start-Sleep 1
      $stopped = (Get-service $service).Status -eq 'Stopped'
  }

  if ((Get-service $service).Status -ne 'Stopped') {
    # Force kill the processes.
    Stop-Process -Force -PassThru -Id (Get-WmiObject win32_process |
      Where CommandLine -Like $cmdline).ProcessId

    # Wait until process has stopped.
    $waited = 0
    $log_period = 10
    $timeout = 60
    while ((Get-service $service).Status -ne 'Stopped' -and $waited -lt $timeout) {
      Start-Sleep 1
      $waited++

      if ($waited % $log_period -eq 0) {
        Log-Output "Waiting for ${service} service to stop"
      }
    }

    # Timeout occurred
    if ($waited -ge $timeout) {
      Throw ("Timeout while waiting for ${service} service to stop")
    }
  }

  Start-Service $service
}

# Check whether the logging agent is installed by whether it's registered as service
function IsLoggingAgentInstalled {
  $logging_status = (Get-Service $LOGGINGAGENT_SERVICE -ErrorAction Ignore).Status
  return -not [string]::IsNullOrEmpty($logging_status)
}

# Installs the logging agent according to https://docs.fluentbit.io/manual/installation/windows#
# Also installs fluent bit stackdriver exporter
function Install-LoggingAgent {
  if (IsStackdriverAgentInstalled) {
    # Remove the existing storage.json file if it exists. This is a workaround
    # for the bug where the logging agent cannot start up if the file is
    # corrupted.
    Remove-Item `
      -Force `
      -ErrorAction Ignore `
      ("$STACKDRIVER_ROOT\LoggingAgent\Main\pos\winevtlog.pos\worker0\" +
       "storage.json")
    Log-Output ("Skip: Stackdriver logging agent is already installed")
    return
  }

  if (IsLoggingAgentInstalled) {
    # Note: we should reinstall the agent if $REDO_STEPS is true
    # here, but we don't know how to run the installer without it prompting
    # when logging agent is already installed. We dumped the strings in the
    # installer binary and searched for flags to do this but found nothing. Oh
    # well.
    Log-Output ("Skip: Fluentbit logging agent is already installed")
    return
  }

  DownloadAndInstall-LoggingAgents
  Create-LoggingAgentServices
}

function DownloadAndInstall-LoggingAgents {
  # Install Logging agent if not present
  if (ShouldWrite-File $LOGGINGAGENT_ROOT\td-agent-bit-${LOGGINGAGENT_VERSION}-win64) {
      $install_dir = 'C:\flb-installers'
      $url = ("https://storage.googleapis.com/gke-release/winnode/fluentbit/td-agent-bit-${LOGGINGAGENT_VERSION}-win64.zip")

      Log-Output 'Downloading Logging agent'
      New-Item $install_dir -ItemType 'directory' -Force | Out-Null
      MustDownload-File -OutFile $install_dir\td.zip -URLs $url

      cd $install_dir
      Log-Output 'Extracting Logging agent'
      Expand-Archive td.zip
      mv .\td\td-agent-bit-${LOGGINGAGENT_VERSION}-win64\ $LOGGINGAGENT_ROOT
      cd C:\
      Remove-Item -Force -Recurse $install_dir
  }

  # Download Logging exporter if needed
  if (ShouldWrite-File $LOGGINGEXPORTER_ROOT\flb-exporter.exe) {
      $url = ("https://storage.googleapis.com/gke-release/winnode/fluentbit-exporter/${LOGGINGEXPORTER_VERSION}/flb-exporter-${LOGGINGEXPORTER_VERSION}.exe")
      Log-Output 'Downloading logging exporter'
      New-Item $LOGGINGEXPORTER_ROOT -ItemType 'directory' -Force | Out-Null
      MustDownload-File `
          -OutFile $LOGGINGEXPORTER_ROOT\flb-exporter.exe `
          -URLs $url `
          -Hash $LOGGINGEXPORTER_HASH `
          -Algorithm SHA256
  }
}

function Create-LoggingAgentServices {
  cd $LOGGINGAGENT_ROOT

  Log-Output "Creating service: ${LOGGINGAGENT_SERVICE}"
  sc.exe create $LOGGINGAGENT_SERVICE binpath= "${LOGGINGAGENT_ROOT}\bin\fluent-bit.exe -c \fluent-bit\conf\fluent-bit.conf"
  sc.exe failure $LOGGINGAGENT_SERVICE reset= 30 actions= restart/5000
  Write-VerboseServiceInfoToConsole -Service $LOGGINGAGENT_SERVICE

  Log-Output "Creating service: ${LOGGINGEXPORTER_SERVICE}"
  sc.exe create  $LOGGINGEXPORTER_SERVICE  binpath= "${LOGGINGEXPORTER_ROOT}\flb-exporter.exe --kubernetes-separator=_ --stackdriver-resource-model=k8s --enable-pod-label-discovery --logtostderr --winsvc  --pod-label-dot-replacement=_"
  sc.exe failure $LOGGINGEXPORTER_SERVICE reset= 30 actions= restart/5000
  Write-VerboseServiceInfoToConsole -Service $LOGGINGEXPORTER_SERVICE
}

# Writes the logging configuration file for Logging agent. Restart-LoggingAgent
# should then be called to pick up the new configuration.
function Configure-LoggingAgent {
  if (IsStackdriverAgentInstalled) {
      Configure-StackdriverAgent
      return
  }

  $fluentbit_config_file = "$LOGGINGAGENT_ROOT\conf\fluent-bit.conf"
  $FLUENTBIT_CONFIG | Out-File -FilePath $fluentbit_config_file -Encoding ASCII
  Log-Output "Wrote logging config to $fluentbit_config_file"

  $fluentbit_parser_file = "$LOGGINGAGENT_ROOT\conf\parsers.conf"
  $PARSERS_CONFIG | Out-File -FilePath $fluentbit_parser_file -Encoding ASCII

  # Create directory for all the log position files.
  New-Item -Type Directory -Path "/var/run/google-fluentbit/pos-files/" -Force | Out-Null

  Log-Output "Wrote logging config to $fluentbit_parser_file"
}

# Fluentbit main config file
$FLUENTBIT_CONFIG = @'
[SERVICE]
    Flush         5
    Grace         120
    Log_Level     info
    Log_File      /var/log/fluentbit.log
    Daemon        off
    Parsers_File  parsers.conf
    HTTP_Server   off
    HTTP_Listen   0.0.0.0
    HTTP_PORT     2020
    plugins_file plugins.conf

    # Storage
    # =======
    # Fluent Bit can use memory and filesystem buffering based mechanisms
    #
    # - https://docs.fluentbit.io/manual/administration/buffering-and-storage
    #
    # storage metrics
    # ---------------
    # publish storage pipeline metrics in '/api/v1/storage'. The metrics are
    # exported only if the 'http_server' option is enabled.
    #
    # storage.metrics on

    # storage.path
    # ------------
    # absolute file system path to store filesystem data buffers (chunks).
    #
    # storage.path /tmp/storage

    # storage.sync
    # ------------
    # configure the synchronization mode used to store the data into the
    # filesystem. It can take the values normal or full.
    #
    # storage.sync normal

    # storage.checksum
    # ----------------
    # enable the data integrity check when writing and reading data from the
    # filesystem. The storage layer uses the CRC32 algorithm.
    #
    # storage.checksum off

    # storage.backlog.mem_limit
    # -------------------------
    # if storage.path is set, Fluent Bit will look for data chunks that were
    # not delivered and are still in the storage layer, these are called
    # backlog data. This option configure a hint of maximum value of memory
    # to use when processing these records.
    #
    # storage.backlog.mem_limit 5M

[INPUT]
    Name         winlog
    Interval_Sec 2
    # Channels Setup,Windows PowerShell
    Channels     application,system,security
    Tag          winevt.raw
    DB           /var/run/google-fluentbit/pos-files/winlog.db

# Json Log Example:
# {"log":"[info:2016-02-16T16:04:05.930-08:00] Some log text here\n","stream":"stdout","time":"2016-02-17T00:04:05.931087621Z"}
[INPUT]
    Name             tail
    Alias            kube_containers
    Tag              kube_<namespace_name>_<pod_name>_<container_name>
    Tag_Regex        (?<pod_name>[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*)_(?<namespace_name>[^_]+)_(?<container_name>.+)-
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             C:\var\log\containers\*.log
    DB               /var/run/google-fluentbit/pos-files/flb_kube.db

[FILTER]
    Name         parser
    Match        kube_*
    Key_Name     log
    Reserve_Data True
    Parser       docker
    Parser       containerd

# Log line format: [IWEF]mmdd hh:mm:ss.uuuuuu threadid file:line] msg
# Example:
# I0716 02:08:55.559351    3356 log_spam.go:42] Command line arguments:
[INPUT]
    Name             tail
    Alias            node-problem-detector
    Tag              node-problem-detector
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             C:\etc\kubernetes\logs\node-problem-detector\*.log.INFO*
    DB               /var/run/google-fluentbit/pos-files/node-problem-detector.db
    Multiline        On
    Parser_Firstline glog

# Example:
# I0928 03:15:50.440223    4880 main.go:51] Starting CSI-Proxy Server ...
[INPUT]
    Name             tail
    Alias            csi-proxy
    Tag              csi-proxy
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             /etc/kubernetes/logs/csi-proxy.log
    DB               /var/run/google-fluentbit/pos-files/csi-proxy.db
    Multiline        On
    Parser_Firstline glog

# I1118 21:26:53.975789       6 proxier.go:1096] Port "nodePort for kube-system/default-http-backend:http" (:31429/tcp) was open before and is still needed
[INPUT]
    Name             tail
    Alias            kube-proxy
    Tag              kube-proxy
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             /etc/kubernetes/logs/kube-proxy.log
    DB               /var/run/google-fluentbit/pos-files/kube-proxy.db
    Multiline        On
    Parser_Firstline glog

# Log line format: [IWEF]mmdd hh:mm:ss.uuuuuu threadid file:line] msg
# Example:
# I0716 02:08:55.559351    3356 log_spam.go:42] Command line arguments:
[INPUT]
    Name             tail
    Alias            gke-metadata-server
    Tag              gke-metadata-server
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             C:\etc\kubernetes\logs\gke-metadata-server\*.log.INFO*
    DB               /var/run/google-fluentbit/pos-files/gke-metadata-server.db
    Multiline        On
    Parser_Firstline glog

# Example:
# time="2019-12-10T21:27:59.836946700Z" level=info msg="loading plugin \"io.containerd.grpc.v1.cri\"..." type=io.containerd.grpc.v1
[INPUT]
    Name             tail
    Alias            container-runtime
    Tag              container-runtime
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             /etc/kubernetes/logs/containerd.log
    DB               /var/run/google-fluentbit/pos-files/container-runtime.db
    # TODO: Add custom parser for containerd logs once format is settled.

# Example:
# I0204 07:32:30.020537    3368 server.go:1048] POST /stats/container/: (13.972191ms) 200 [[Go-http-client/1.1] 10.244.1.3:40537]
[INPUT]
    Name             tail
    Alias            kubelet
    Tag              kubelet
    Mem_Buf_Limit    5MB
    Skip_Long_Lines  On
    Refresh_Interval 5
    Path             /etc/kubernetes/logs/kubelet.log
    DB               /var/run/google-fluentbit/pos-files/kubelet.db
    Multiline        On
    Parser_Firstline glog

[FILTER]
    Name        modify
    Match       *
    Hard_rename log message

[FILTER]
    Name        modify
    Match       winevt.raw
    Hard_rename Message message

[FILTER]
    Name         parser
    Match        kube_*
    Key_Name     message
    Reserve_Data True
    Parser       glog
    Parser       json

[OUTPUT]
    Name        http
    Match       *
    Host        127.0.0.1
    Port        2021
    URI         /logs
    header_tag  FLUENT-TAG
    Format      msgpack
    Retry_Limit 2
'@

# Fluentbit parsers config file
$PARSERS_CONFIG = @'
[PARSER]
    Name        docker
    Format      json
    Time_Key    time
    Time_Format %Y-%m-%dT%H:%M:%S.%L%z

[PARSER]
    Name        containerd
    Format      regex
    Regex       ^(?<time>.+) (?<stream>stdout|stderr) [^ ]* (?<log>.*)$
    Time_Key    time
    Time_Format %Y-%m-%dT%H:%M:%S.%L%z

[PARSER]
    Name        json
    Format      json

[PARSER]
    Name        syslog
    Format      regex
    Regex       ^\<(?<pri>[0-9]+)\>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<host>[^ ]*) (?<ident>[a-zA-Z0-9_\/\.\-]*)(?:\[(?<pid>[0-9]+)\])?(?:[^\:]*\:)? *(?<message>.*)$
    Time_Key    time
    Time_Format %b %d %H:%M:%S

[PARSER]
    Name        glog
    Format      regex
    Regex       ^(?<severity>\w)(?<time>\d{4} [^\s]*)\s+(?<pid>\d+)\s+(?<source_file>[^ \]]+)\:(?<source_line>\d+)\]\s(?<message>.*)$
    Time_Key    time
    Time_Format %m%d %H:%M:%S.%L

[PARSER]
    Name        network-log
    Format      json
    Time_Key    timestamp
    Time_Format %Y-%m-%dT%H:%M:%S.%L%z

[PARSER]
    Name        syslog-rfc5424
    Format      regex
    Regex       ^\<(?<pri>[0-9]{1,5})\>1 (?<time>[^ ]+) (?<host>[^ ]+) (?<ident>[^ ]+) (?<pid>[-0-9]+) (?<msgid>[^ ]+) (?<extradata>(\[(.*?)\]|-)) (?<message>.+)$
    Time_Key    time
    Time_Format %Y-%m-%dT%H:%M:%S.%L%z
    Time_Keep   On

[PARSER]
    Name        syslog-rfc3164-local
    Format      regex
    Regex       ^\<(?<pri>[0-9]+)\>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<ident>[a-zA-Z0-9_\/\.\-]*)(?:\[(?<pid>[0-9]+)\])?(?:[^\:]*\:)? *(?<message>.*)$
    Time_Key    time
    Time_Format %b %d %H:%M:%S
    Time_Keep   On

[PARSER]
    Name        syslog-rfc3164
    Format      regex
    Regex       /^\<(?<pri>[0-9]+)\>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<host>[^ ]*) (?<ident>[a-zA-Z0-9_\/\.\-]*)(?:\[(?<pid>[0-9]+)\])?(?:[^\:]*\:)? *(?<message>.*)$/
    Time_Key    time
    Time_Format %b %d %H:%M:%S
    Time_Keep   On

[PARSER]
    Name    kube-custom
    Format  regex
    Regex   (?<tag>[^.]+)?\.?(?<pod_name>[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*)_(?<namespace_name>[^_]+)_(?<container_name>.+)-(?<docker_id>[a-z0-9]{64})\.log$
'@


# ----------- Stackdriver logging setup --------------------------
# This section would be deprecated soon
#

$STACKDRIVER_VERSION = 'v1-11'
$STACKDRIVER_ROOT = 'C:\Program Files (x86)\Stackdriver'

# Restarts the Stackdriver logging agent, or starts it if it is not currently
# running. A standard `Restart-Service StackdriverLogging` may fail because
# StackdriverLogging sometimes is unstoppable, so this function works around it
# by killing the processes.
function Restart-StackdriverAgent {
  Stop-Service -NoWait -ErrorAction Ignore StackdriverLogging

  # Wait (if necessary) for service to stop.
  $timeout = 10
  $stopped = (Get-service StackdriverLogging).Status -eq 'Stopped'
  for ($i = 0; $i -lt $timeout -and !($stopped); $i++) {
      Start-Sleep 1
      $stopped = (Get-service StackdriverLogging).Status -eq 'Stopped'
  }

  if ((Get-service StackdriverLogging).Status -ne 'Stopped') {
    # Force kill the processes.
    Stop-Process -Force -PassThru -Id (Get-WmiObject win32_process |
      Where CommandLine -Like '*Stackdriver/logging*').ProcessId

    # Wait until process has stopped.
    $waited = 0
    $log_period = 10
    $timeout = 60
    while ((Get-service StackdriverLogging).Status -ne 'Stopped' -and $waited -lt $timeout) {
      Start-Sleep 1
      $waited++

      if ($waited % $log_period -eq 0) {
        Log-Output "Waiting for StackdriverLogging service to stop"
      }
    }

    # Timeout occurred
    if ($waited -ge $timeout) {
      Throw ("Timeout while waiting for StackdriverLogging service to stop")
    }
  }

  Start-Service StackdriverLogging
}

# Check whether the logging agent is installed by whether it's registered as service
function IsStackdriverAgentInstalled {
  $stackdriver_status = (Get-Service StackdriverLogging -ErrorAction Ignore).Status
  return -not [string]::IsNullOrEmpty($stackdriver_status)
}

# Writes the logging configuration file for Stackdriver. Restart-LoggingAgent
# should then be called to pick up the new configuration.
function Configure-StackdriverAgent {
  $fluentd_config_dir = "$STACKDRIVER_ROOT\LoggingAgent\config.d"
  $fluentd_config_file = "$fluentd_config_dir\k8s_containers.conf"

  # Create a configuration file for kubernetes containers.
  # The config.d directory should have already been created automatically, but
  # try creating again just in case.
  New-Item $fluentd_config_dir -ItemType 'directory' -Force | Out-Null

  $config = $FLUENTD_CONFIG.replace('NODE_NAME', (hostname))
  $config | Out-File -FilePath $fluentd_config_file -Encoding ASCII
  Log-Output "Wrote fluentd logging config to $fluentd_config_file"

  # Configure StackdriverLogging to automatically restart on failure after 10
  # seconds. The logging agent may die die to various disruptions but can be
  # resumed.
  sc.exe failure StackdriverLogging reset= 0 actions= restart/1000/restart/10000
  Write-VerboseServiceInfoToConsole -Service 'StackdriverLogging'
}

# The NODE_NAME placeholder must be replaced with the node's name (hostname).
$FLUENTD_CONFIG = @'
# This configuration file for Fluentd is used to watch changes to kubernetes
# container logs in the directory /var/lib/docker/containers/ and submit the
# log records to Google Cloud Logging using the cloud-logging plugin.
#
# Example
# =======
# A line in the Docker log file might look like this JSON:
#
# {"log":"2014/09/25 21:15:03 Got request with path wombat\\n",
#  "stream":"stderr",
#   "time":"2014-09-25T21:15:03.499185026Z"}
#
# The original tag is derived from the log file's location.
# For example a Docker container's logs might be in the directory:
#  /var/lib/docker/containers/997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b
# and in the file:
#  997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b-json.log
# where 997599971ee6... is the Docker ID of the running container.
# The Kubernetes kubelet makes a symbolic link to this file on the host
# machine in the /var/log/containers directory which includes the pod name,
# the namespace name and the Kubernetes container name:
#    synthetic-logger-0.25lps-pod_default_synth-lgr-997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b.log
#    ->
#    /var/lib/docker/containers/997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b/997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b-json.log
# The /var/log directory on the host is mapped to the /var/log directory in the container
# running this instance of Fluentd and we end up collecting the file:
#   /var/log/containers/synthetic-logger-0.25lps-pod_default_synth-lgr-997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b.log
# This results in the tag:
#  var.log.containers.synthetic-logger-0.25lps-pod_default_synth-lgr-997599971ee6366d4a5920d25b79286ad45ff37a74494f262e3bc98d909d0a7b.log
# where 'synthetic-logger-0.25lps-pod' is the pod name, 'default' is the
# namespace name, 'synth-lgr' is the container name and '997599971ee6..' is
# the container ID.
# The record reformer is used to extract pod_name, namespace_name and
# container_name from the tag and set them in a local_resource_id in the
# format of:
# 'k8s_container.<NAMESPACE_NAME>.<POD_NAME>.<CONTAINER_NAME>'.
# The reformer also changes the tags to 'stderr' or 'stdout' based on the
# value of 'stream'.
# local_resource_id is later used by google_cloud plugin to determine the
# monitored resource to ingest logs against.
# Json Log Example:
# {"log":"[info:2016-02-16T16:04:05.930-08:00] Some log text here\n","stream":"stdout","time":"2016-02-17T00:04:05.931087621Z"}
# CRI Log Example:
# 2016-02-17T00:04:05.931087621Z stdout F [info:2016-02-16T16:04:05.930-08:00] Some log text here
<source>
  @type tail
  path /var/log/containers/*.log
  pos_file /var/log/gcp-containers.log.pos
  # Tags at this point are in the format of:
  # reform.var.log.containers.<POD_NAME>_<NAMESPACE_NAME>_<CONTAINER_NAME>-<CONTAINER_ID>.log
  tag reform.*
  read_from_head true
  <parse>
    @type multi_format
    <pattern>
      format json
      time_key time
      time_format %Y-%m-%dT%H:%M:%S.%NZ
      keep_time_key
    </pattern>
    <pattern>
      format /^(?<time>.+) (?<stream>stdout|stderr) [^ ]* (?<log>.*)$/
      time_format %Y-%m-%dT%H:%M:%S.%N%:z
    </pattern>
  </parse>
</source>
# Example:
# I0204 07:32:30.020537    3368 server.go:1048] POST /stats/container/: (13.972191ms) 200 [[Go-http-client/1.1] 10.244.1.3:40537]
<source>
  @type tail
  format multiline
  multiline_flush_interval 5s
  format_firstline /^\w\d{4}/
  format1 /^(?<severity>\w)(?<time>\d{4} [^\s]*)\s+(?<pid>\d+)\s+(?<source>[^ \]]+)\] (?<message>.*)/
  time_format %m%d %H:%M:%S.%N
  path /etc/kubernetes/logs/kubelet.log
  pos_file /etc/kubernetes/logs/gcp-kubelet.log.pos
  tag kubelet
</source>
# Example:
# I1118 21:26:53.975789       6 proxier.go:1096] Port "nodePort for kube-system/default-http-backend:http" (:31429/tcp) was open before and is still needed
<source>
  @type tail
  format multiline
  multiline_flush_interval 5s
  format_firstline /^\w\d{4}/
  format1 /^(?<severity>\w)(?<time>\d{4} [^\s]*)\s+(?<pid>\d+)\s+(?<source>[^ \]]+)\] (?<message>.*)/
  time_format %m%d %H:%M:%S.%N
  path /etc/kubernetes/logs/kube-proxy.log
  pos_file /etc/kubernetes/logs/gcp-kube-proxy.log.pos
  tag kube-proxy
</source>
# Example:
# I0928 03:15:50.440223    4880 main.go:51] Starting CSI-Proxy Server ...
<source>
  @type tail
  format multiline
  multiline_flush_interval 5s
  format_firstline /^\w\d{4}/
  format1 /^(?<severity>\w)(?<time>\d{4} [^\s]*)\s+(?<pid>\d+)\s+(?<source>[^ \]]+)\] (?<message>.*)/
  time_format %m%d %H:%M:%S.%N
  path /etc/kubernetes/logs/csi-proxy.log
  pos_file /etc/kubernetes/logs/gcp-csi-proxy.log.pos
  tag csi-proxy
</source>
# Example:
# time="2019-12-10T21:27:59.836946700Z" level=info msg="loading plugin \"io.containerd.grpc.v1.cri\"..." type=io.containerd.grpc.v1
<source>
  @type tail
  format multiline
  multiline_flush_interval 5s
  format_firstline /^time=/
  format1 /^time="(?<time>[^ ]*)" level=(?<severity>\w*) (?<message>.*)/
  time_format %Y-%m-%dT%H:%M:%S.%N%z
  path /etc/kubernetes/logs/containerd.log
  pos_file /etc/kubernetes/logs/gcp-containerd.log.pos
  tag container-runtime
</source>
<match reform.**>
  @type record_reformer
  enable_ruby true
  <record>
    # Extract local_resource_id from tag for 'k8s_container' monitored
    # resource. The format is:
    # 'k8s_container.<namespace_name>.<pod_name>.<container_name>'.
    "logging.googleapis.com/local_resource_id" ${"k8s_container.#{tag_suffix[4].rpartition('.')[0].split('_')[1]}.#{tag_suffix[4].rpartition('.')[0].split('_')[0]}.#{tag_suffix[4].rpartition('.')[0].split('_')[2].rpartition('-')[0]}"}
    # Rename the field 'log' to a more generic field 'message'. This way the
    # fluent-plugin-google-cloud knows to flatten the field as textPayload
    # instead of jsonPayload after extracting 'time', 'severity' and
    # 'stream' from the record.
    message ${record['log']}
    # If 'severity' is not set, assume stderr is ERROR and stdout is INFO.
    severity ${record['severity'] || if record['stream'] == 'stderr' then 'ERROR' else 'INFO' end}
  </record>
  tag ${if record['stream'] == 'stderr' then 'raw.stderr' else 'raw.stdout' end}
  remove_keys stream,log
</match>
# TODO: detect exceptions and forward them as one log entry using the
# detect_exceptions plugin
# This section is exclusive for k8s_container logs. These logs come with
# 'raw.stderr' or 'raw.stdout' tags.
<match {raw.stderr,raw.stdout}>
  @type google_cloud
  # Try to detect JSON formatted log entries.
  detect_json true
  # Allow log entries from multiple containers to be sent in the same request.
  split_logs_by_tag false
  # Set the buffer type to file to improve the reliability and reduce the memory consumption
  buffer_type file
  buffer_path /var/log/fluentd-buffers/kubernetes.containers.buffer
  # Set queue_full action to block because we want to pause gracefully
  # in case of the off-the-limits load instead of throwing an exception
  buffer_queue_full_action block
  # Set the chunk limit conservatively to avoid exceeding the recommended
  # chunk size of 5MB per write request.
  buffer_chunk_limit 512k
  # Cap the combined memory usage of this buffer and the one below to
  # 512KiB/chunk * (6 + 2) chunks = 4 MiB
  buffer_queue_limit 6
  # Never wait more than 5 seconds before flushing logs in the non-error case.
  flush_interval 5s
  # Never wait longer than 30 seconds between retries.
  max_retry_wait 30
  # Disable the limit on the number of retries (retry forever).
  disable_retry_limit
  # Use multiple threads for processing.
  num_threads 2
  use_grpc true
  # Skip timestamp adjustment as this is in a controlled environment with
  # known timestamp format. This helps with CPU usage.
  adjust_invalid_timestamps false
</match>
# Attach local_resource_id for 'k8s_node' monitored resource.
<filter **>
  @type record_transformer
  enable_ruby true
  <record>
    "logging.googleapis.com/local_resource_id" ${"k8s_node.NODE_NAME"}
  </record>
</filter>
'@

# Export all public functions:
Export-ModuleMember -Function *-*
