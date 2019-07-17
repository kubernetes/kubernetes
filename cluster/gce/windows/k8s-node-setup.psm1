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

$INFRA_CONTAINER = 'mcr.microsoft.com/k8s/core/pause:1.0.0'
$GCE_METADATA_SERVER = "169.254.169.254"
# The "management" interface is used by the kubelet and by Windows pods to talk
# to the rest of the Kubernetes cluster *without NAT*. This interface does not
# exist until an initial HNS network has been created on the Windows node - see
# Add_InitialHnsNetwork().
$MGMT_ADAPTER_NAME = "vEthernet (Ethernet*"

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

# Writes debugging information, such as Windows version and patch info, to the
# console.
function Dump-DebugInfoToConsole {
  Try {
    $version = "$([System.Environment]::OSVersion.Version | Out-String)"
    $hotfixes = "$(Get-Hotfix | Out-String)"
    $image = "$(Get-InstanceMetadata 'image' | Out-String)"
    Log-Output "Windows version:`n$version"
    Log-Output "Installed hotfixes:`n$hotfixes"
    Log-Output "GCE Windows image:`n$image"
  } Catch { }
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
  # ${kube_env_table} = ConvertFrom-Yaml ${kube_env}
  # ${kube_env_table}
  # ${kube_env_table}.GetType()

  # The type of kube_env is a powershell String.
  $kube_env = Get-InstanceMetadataAttribute 'kube-env'
  $kube_env_table = ConvertFrom-Yaml ${kube_env}
  return ${kube_env_table}
}

# Sets the environment variable $Key to $Value at the Machine scope (will
# be present in the environment for all new shells after a reboot).
function Set_MachineEnvironmentVar {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$true)] [string]$Value
  )
  [Environment]::SetEnvironmentVariable($Key, $Value, "Machine")
}

# Sets the environment variable $Key to $Value in the current shell.
function Set_CurrentShellEnvironmentVar {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$true)] [string]$Value
  )
  $expression = '$env:' + $Key + ' = "' + $Value + '"'
  Invoke-Expression ${expression}
}

# Sets environment variables used by Kubernetes binaries and by other functions
# in this module. Depends on numerous ${kube_env} keys.
function Set-EnvironmentVars {
  # Turning the kube-env values into environment variables is not required but
  # it makes debugging this script easier, and it also makes the syntax a lot
  # easier (${env:K8S_DIR} can be expanded within a string but
  # ${kube_env}['K8S_DIR'] cannot be afaik).
  $env_vars = @{
    "K8S_DIR" = ${kube_env}['K8S_DIR']
    "NODE_DIR" = ${kube_env}['NODE_DIR']
    "CNI_DIR" = ${kube_env}['CNI_DIR']
    "CNI_CONFIG_DIR" = ${kube_env}['CNI_CONFIG_DIR']
    "PKI_DIR" = ${kube_env}['PKI_DIR']
    "KUBELET_CONFIG" = ${kube_env}['KUBELET_CONFIG_FILE']
    "BOOTSTRAP_KUBECONFIG" = ${kube_env}['BOOTSTRAP_KUBECONFIG_FILE']
    "KUBEPROXY_KUBECONFIG" = ${kube_env}['KUBEPROXY_KUBECONFIG_FILE']

    "Path" = ${env:Path} + ";" + ${kube_env}['NODE_DIR']
    "KUBE_NETWORK" = "l2bridge".ToLower()
    "CA_CERT_BUNDLE_PATH" = ${kube_env}['PKI_DIR'] + '\ca-certificates.crt'
    "KUBELET_CERT_PATH" = ${kube_env}['PKI_DIR'] + '\kubelet.crt'
    "KUBELET_KEY_PATH" = ${kube_env}['PKI_DIR'] + '\kubelet.key'

    # TODO(pjh): these are only in flags, can be removed from env once flags are
    # moved to util.sh:
    "LOGS_DIR" = ${kube_env}['LOGS_DIR']
    "MANIFESTS_DIR" = ${kube_env}['MANIFESTS_DIR']
    "KUBECONFIG" = ${kube_env}['KUBECONFIG_FILE']
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
  sc.exe config wuauserv start=disabled
  sc.exe stop wuauserv

  # Use TLS 1.2: needed for Invoke-WebRequest downloads from github.com.
  [Net.ServicePointManager]::SecurityProtocol = `
      [Net.SecurityProtocolType]::Tls12

  # https://github.com/cloudbase/powershell-yaml
  Log-Output "Installing powershell-yaml module from external repo"
  Install-Module -Name powershell-yaml -Force
}

# Disables Windows Defender realtime scanning.
# TODO: remove this workaround once the fix is rolled out the Windows image
# https://github.com/kubernetes/kubernetes/issues/75148
function Disable-WindowsDefender {
  # Windows Defender periodically consumes 100% of the CPU, so disable realtime
  # scanning. Uninstalling the Windows Feature will prevent the service from
  # starting after a reboot.
  # TODO(pjh): move this step to image preparation, since we don't want to do a
  # full reboot here.
  if ((Get-WindowsFeature -Name 'Windows-Defender').Installed) {
    Log-Output "Disabling Windows Defender service"
    Set-MpPreference -DisableRealtimeMonitoring $true
    Uninstall-WindowsFeature -Name 'Windows-Defender'
  }
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
    "${env:PKI_DIR}"), "C:\tmp", "C:\var\log") {
    mkdir -Force $dir
  }
}

# Downloads some external helper scripts needed by other functions in this
# module.
function Download-HelperScripts {
  if (-not (ShouldWrite-File ${env:K8S_DIR}\hns.psm1)) {
    return
  }
  MustDownload-File -OutFile ${env:K8S_DIR}\hns.psm1 `
    -URLs "https://github.com/Microsoft/SDN/raw/master/Kubernetes/windows/hns.psm1"
}

# Takes the Windows version string from the cluster bash scripts (e.g.
# 'win1809') and returns the correct label to use for containers on this
# version of Windows. Returns $null if $WinVersion is unknown.
function Get_ContainerVersionLabel {
  param (
    [parameter(Mandatory=$true)] [string]$WinVersion
  )
  # -match does regular expression matching.
  if ($WinVersion -match '1809') {
    return '1809'
  }
  elseif ($WinVersion -match '2019') {
    return 'ltsc2019'
  }
  Throw ("Unknown Windows version $WinVersion, don't know its container " +
         "version label")
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
  MustDownload-File -Hash $hash -OutFile ${tmp_dir}\${filename} -URLs $urls

  # Change the directory to the parent directory of ${env:K8S_DIR} and untar.
  # This (over-)writes ${dest_dir}/kubernetes/node/bin/*.exe files.
  $dest_dir = (Get-Item ${env:K8S_DIR}).Parent.Fullname
  tar xzf ${tmp_dir}\${filename} -C ${dest_dir}

  # Clean up the temporary directory
  Remove-Item -Force -Recurse $tmp_dir
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

# Returns the "management" subnet on which the Windows pods+kubelet will
# communicate with the rest of the Kubernetes cluster without NAT. In GCE this
# is the subnet that VM internal IPs are allocated from.
#
# This function will fail if Add_InitialHnsNetwork() has not been called first.
function Get_MgmtSubnet {
  $net_adapter = Get_MgmtNetAdapter

  # TODO(pjh): applying the primary interface's subnet mask to its IP address
  # *should* give us the GCE network subnet that VM IP addresses are being
  # allocated from... however it might be more accurate or straightforward to
  # just fetch the IP address range for the VPC subnet that the kube-up script
  # creates (kubernetes-subnet-default).
  $addr = (Get-NetIPAddress `
      -InterfaceAlias ${net_adapter}.ifAlias `
      -AddressFamily IPv4).IPAddress
  $mask = (Get-WmiObject Win32_NetworkAdapterConfiguration |
      Where-Object InterfaceIndex -eq $(${net_adapter}.ifIndex)).IPSubnet[0]
  $mgmt_subnet = `
    (ConvertTo_DecimalIP ${addr}) -band (ConvertTo_DecimalIP ${mask})
  $mgmt_subnet = ConvertTo_DottedDecimalIP ${mgmt_subnet}
  return "${mgmt_subnet}/$(ConvertTo_MaskLength $mask)"
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
#   KUBELET_CERT
#   KUBELET_KEY
function Create-NodePki {
  Log-Output "Creating node pki files"

  $CA_CERT_BUNDLE = ${kube_env}['CA_CERT']
  $KUBELET_CERT = ${kube_env}['KUBELET_CERT']
  $KUBELET_KEY = ${kube_env}['KUBELET_KEY']

  Write_PkiData "${CA_CERT_BUNDLE}" ${env:CA_CERT_BUNDLE_PATH}
  Write_PkiData "${KUBELET_CERT}" ${env:KUBELET_CERT_PATH}
  Write_PkiData "${KUBELET_KEY}" ${env:KUBELET_KEY_PATH}
  Get-ChildItem ${env:PKI_DIR}
}

# Creates the kubelet kubeconfig at $env:BOOTSTRAP_KUBECONFIG.
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   KUBERNETES_MASTER_NAME: the apiserver IP address.
function Create-KubeletKubeconfig {
  # The API server IP address comes from KUBERNETES_MASTER_NAME in kube-env, I
  # think. cluster/gce/gci/configure-helper.sh?l=2801
  $apiserverAddress = ${kube_env}['KUBERNETES_MASTER_NAME']

  # TODO(pjh): set these using kube-env values.
  $createBootstrapConfig = $true
  $fetchBootstrapConfig = $false

  if (${createBootstrapConfig}) {
    if (-not (ShouldWrite-File ${env:BOOTSTRAP_KUBECONFIG})) {
      return
    }
    New-Item -Force -ItemType file ${env:BOOTSTRAP_KUBECONFIG} | Out-Null
    # TODO(mtaufen): is user "kubelet" correct? Other examples use e.g.
    #   "system:node:$(hostname)".
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
    certificate-authority: CA_CERT_BUNDLE_PATH
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context'.`
      replace('KUBELET_CERT_PATH', ${env:KUBELET_CERT_PATH}).`
      replace('KUBELET_KEY_PATH', ${env:KUBELET_KEY_PATH}).`
      replace('APISERVER_ADDRESS', ${apiserverAddress}).`
      replace('CA_CERT_BUNDLE_PATH', ${env:CA_CERT_BUNDLE_PATH})
    Log-Output ("kubelet bootstrap kubeconfig:`n" +
                "$(Get-Content -Raw ${env:BOOTSTRAP_KUBECONFIG})")
  }
  elseif (${fetchBootstrapConfig}) {
    Log_NotImplemented `
        "fetching kubelet bootstrap-kubeconfig file from metadata"
    # get-metadata-value "instance/attributes/bootstrap-kubeconfig" >
    #   /var/lib/kubelet/bootstrap-kubeconfig
    Log-Output ("kubelet bootstrap kubeconfig:`n" +
                "$(Get-Content -Raw ${env:BOOTSTRAP_KUBECONFIG})")
  }
  else {
    Log_NotImplemented "fetching kubelet kubeconfig file from metadata"
  }
}

# Creates the kube-proxy user kubeconfig file at $env:KUBEPROXY_KUBECONFIG.
#
# Create-NodePki() must be called first.
#
# Required ${kube_env} keys:
#   CA_CERT
#   KUBE_PROXY_TOKEN
function Create-KubeproxyKubeconfig {
  if (-not (ShouldWrite-File ${env:KUBEPROXY_KUBECONFIG})) {
    return
  }

  New-Item -Force -ItemType file ${env:KUBEPROXY_KUBECONFIG} | Out-Null

  # In configure-helper.sh kubelet kubeconfig uses certificate-authority while
  # kubeproxy kubeconfig uses certificate-authority-data, ugh. Does it matter?
  # Use just one or the other for consistency?
  Set-Content ${env:KUBEPROXY_KUBECONFIG} `
'apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: KUBEPROXY_TOKEN
clusters:
- name: local
  cluster:
    server: https://APISERVER_ADDRESS
    certificate-authority-data: CA_CERT
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context'.`
    replace('KUBEPROXY_TOKEN', ${kube_env}['KUBE_PROXY_TOKEN']).`
    replace('CA_CERT', ${kube_env}['CA_CERT']).`
    replace('APISERVER_ADDRESS', ${kube_env}['KUBERNETES_MASTER_NAME'])

  Log-Output ("kubeproxy kubeconfig:`n" +
              "$(Get-Content -Raw ${env:KUBEPROXY_KUBECONFIG})")
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

  # For Windows nodes the pod gateway IP address is the .1 address in the pod
  # CIDR for the host, but from inside containers it's the .2 address.
  $pod_gateway = `
      ${env:POD_CIDR}.substring(0, ${env:POD_CIDR}.lastIndexOf('.')) + '.1'
  $pod_endpoint_gateway = `
      ${env:POD_CIDR}.substring(0, ${env:POD_CIDR}.lastIndexOf('.')) + '.2'
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

  Get-HNSPolicyList | Remove-HnsPolicyList

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

function Configure-GcePdTools {
  if (ShouldWrite-File ${env:K8S_DIR}\GetGcePdName.dll) {
    MustDownload-File -OutFile ${env:K8S_DIR}\GetGcePdName.dll `
      -URLs "https://github.com/pjh/gce-tools/raw/master/GceTools/GetGcePdName/GetGcePdName.dll"
  }
  if (-not (Test-Path $PsHome\profile.ps1)) {
    New-Item -path $PsHome\profile.ps1 -type file
  }

    Add-Content $PsHome\profile.ps1 `
'$modulePath = "K8S_DIR\GetGcePdName.dll"
Unblock-File $modulePath
Import-Module -Name $modulePath'.replace('K8S_DIR', ${env:K8S_DIR})
}

# Downloads the Windows CNI binaries and writes a CNI config file under
# $env:CNI_CONFIG_DIR.
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
#   CLUSTER_IP_RANGE
#   SERVICE_CLUSTER_IP_RANGE
function Configure-CniNetworking {
  $CNI_RELEASE_VERSION = 'v0.8.0'
  if ((ShouldWrite-File ${env:CNI_DIR}\win-bridge.exe) -or
      (ShouldWrite-File ${env:CNI_DIR}\host-local.exe)) {
    $tmp_dir = 'C:\cni_tmp'
    New-Item $tmp_dir -ItemType 'directory' -Force | Out-Null

    $release_url = ('https://github.com/containernetworking/plugins/releases/' +
        'download/' + $CNI_RELEASE_VERSION + '/')
    $sha_url = ($release_url +
        "cni-plugins-windows-amd64-$CNI_RELEASE_VERSION.tgz.sha1")
    $tgz_url = ($release_url +
        "cni-plugins-windows-amd64-$CNI_RELEASE_VERSION.tgz")
    MustDownload-File -URLs $sha_url -OutFile $tmp_dir\cni-plugins.sha1
    $sha1_val = ($(Get-Content $tmp_dir\cni-plugins.sha1) -split ' ',2)[0]
    MustDownload-File `
        -URLs $tgz_url `
        -OutFile $tmp_dir\cni-plugins.tgz `
        -Hash $sha1_val

    Push-Location $tmp_dir
    # tar can only extract in the current directory.
    tar -xvf $tmp_dir\cni-plugins.tgz
    Move-Item -Force host-local.exe ${env:CNI_DIR}\
    Move-Item -Force win-bridge.exe ${env:CNI_DIR}\
    Pop-Location
    Remove-Item -Force -Recurse $tmp_dir
  }
  if (-not ((Test-Path ${env:CNI_DIR}\win-bridge.exe) -and `
            (Test-Path ${env:CNI_DIR}\host-local.exe))) {
    Log-Output `
        "win-bridge.exe and host-local.exe not found in ${env:CNI_DIR}" `
        -Fatal
  }

  $l2bridge_conf = "${env:CNI_CONFIG_DIR}\l2bridge.conf"
  if (-not (ShouldWrite-File ${l2bridge_conf})) {
    return
  }

  $mgmt_ip = (Get_MgmtNetAdapter |
              Get-NetIPAddress -AddressFamily IPv4).IPAddress
  $mgmt_subnet = Get_MgmtSubnet
  Log-Output ("using mgmt IP ${mgmt_ip} and mgmt subnet ${mgmt_subnet} for " +
              "CNI config")

  # We reserve .1 and .2 for gateways. Start the CIDR range from ".3" so that
  # IPAM does not allocate those IPs to pods.
  $cidr_range_start = `
      ${env:POD_CIDR}.substring(0, ${env:POD_CIDR}.lastIndexOf('.')) + '.3'

  # Explanation of the CNI config values:
  #   CLUSTER_CIDR: the cluster CIDR from which pod CIDRs are allocated.
  #   POD_CIDR: the pod CIDR assigned to this node.
  #   CIDR_RANGE_START: start of the pod CIDR range.
  #   MGMT_SUBNET: the subnet on which the Windows pods + kubelet will
  #     communicate with the rest of the cluster without NAT (i.e. the subnet
  #     that VM internal IPs are allocated from).
  #   MGMT_IP: the IP address assigned to the node's primary network interface
  #     (i.e. the internal IP of the GCE VM).
  #   SERVICE_CIDR: the CIDR used for kubernetes services.
  #   DNS_SERVER_IP: the cluster's DNS server IP address.
  #   DNS_DOMAIN: the cluster's DNS domain, e.g. "cluster.local".
  New-Item -Force -ItemType file ${l2bridge_conf} | Out-Null
  Set-Content ${l2bridge_conf} `
'{
  "cniVersion":  "0.2.0",
  "name":  "l2bridge",
  "type":  "win-bridge",
  "capabilities":  {
    "portMappings":  true
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
    {
      "Name":  "EndpointPolicy",
      "Value":  {
        "Type":  "OutBoundNAT",
        "ExceptionList":  [
          "CLUSTER_CIDR",
          "SERVICE_CIDR",
          "MGMT_SUBNET"
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
  replace('CLUSTER_CIDR', ${kube_env}['CLUSTER_IP_RANGE']).`
  replace('SERVICE_CIDR', ${kube_env}['SERVICE_CLUSTER_IP_RANGE']).`
  replace('MGMT_SUBNET', ${mgmt_subnet})

  Log-Output "CNI config:`n$(Get-Content -Raw ${l2bridge_conf})"
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
  $default_kubelet_args = @(`
      "--pod-infra-container-image=${INFRA_CONTAINER}"
  )
  $kubelet_args = ${default_kubelet_args} + ${kubelet_args}
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
      "--cluster-cidr=$(${kube_env}['CLUSTER_IP_RANGE'])"
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
  sc.exe create kubelet binPath= "${env:NODE_DIR}\kubelet.exe ${kubelet_args}" start= demand
  sc.exe failure kubelet reset= 0 actions= restart/10000
  Log-Output "Starting kubelet service"
  sc.exe start kubelet

  Log-Output "Waiting 10 seconds for kubelet to stabilize"
  Start-Sleep 10

  if (Get-Process | Where-Object Name -eq "kube-proxy") {
    Log-Output -Fatal `
        "A kube-proxy process is already running, don't know what to do"
  }
  Log-Output "Creating kube-proxy service"
  sc.exe create kube-proxy binPath= "${env:NODE_DIR}\kube-proxy.exe ${kubeproxy_args}" start= demand
  sc.exe failure kube-proxy reset= 0 actions= restart/10000
  Log-Output "Starting kube-proxy service"
  sc.exe start kube-proxy

  # F1020 23:08:52.000083    9136 server.go:361] unable to load in-cluster
  # configuration, KUBERNETES_SERVICE_HOST and KUBERNETES_SERVICE_PORT must be
  # defined
  # TODO(pjh): still getting errors like these in kube-proxy log:
  # E1023 04:03:58.143449    4840 reflector.go:205] k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/factory.go:129: Failed to list *core.Endpoints: Get https://35.239.84.171/api/v1/endpoints?limit=500&resourceVersion=0: dial tcp 35.239.84.171:443: connectex: A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond.
  # E1023 04:03:58.150266    4840 reflector.go:205] k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion/factory.go:129: Failed to list *core.Service: Get https://35.239.84.171/api/v1/services?limit=500&resourceVersion=0: dial tcp 35.239.84.171:443: connectex: A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond.

  Log_Todo ("verify that jobs are still running; print more details about " +
            "the background jobs.")
  Log-Output "$(Get-Service kube* | Out-String)"
  Verify_GceMetadataServerRouteIsPresent
  Log-Output "Kubernetes components started successfully"
}

# Runs 'kubectl get nodes'.
# TODO(pjh): run more verification commands.
function Verify-WorkerServices {
  Log-Output ("kubectl get nodes:`n" +
              "$(& ${env:NODE_DIR}\kubectl.exe get nodes | Out-String)")
  Verify_GceMetadataServerRouteIsPresent
  Log_Todo "run more verification commands."
}

# Pulls the infra/pause container image onto the node so that it will be
# immediately available when the kubelet tries to run pods.
# TODO(pjh): downloading the container container image may take a few minutes;
# figure out how to run this in the background while perform the rest of the
# node startup steps!
function Pull-InfraContainer {
  $name, $label = $INFRA_CONTAINER -split ':',2
  if (-not ("$(& docker image list)" -match "$name.*$label")) {
    & docker pull $INFRA_CONTAINER
    if (!$?) {
      throw "Error running 'docker pull $INFRA_CONTAINER'"
    }
  }
  $inspect = "$(& docker inspect $INFRA_CONTAINER | Out-String)"
  Log-Output "Infra/pause container:`n$inspect"
}

# Add a registry key for docker in EventLog so that log messages are mapped
# correctly. This is a workaround since the key is missing in the base image.
# https://github.com/MicrosoftDocs/Virtualization-Documentation/pull/503
# TODO: Fix this in the base image.
function Create-DockerRegistryKey {
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
function Configure-Dockerd {
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

# TODO(pjh): move the Stackdriver logging agent code below into a separate
# module; it was put here temporarily to avoid disrupting the file layout in
# the K8s release machinery.
$STACKDRIVER_VERSION = 'v1-9'
$STACKDRIVER_ROOT = 'C:\Program Files (x86)\Stackdriver'


# Restart the Stackdriver logging agent
# `Restart-Service StackdriverLogging` may fail because StackdriverLogging
# sometimes is unstoppable, so we work around it by killing the processes.
function Restart-StackdriverLoggingAgent {
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

# Install and start the Stackdriver logging agent according to
# https://cloud.google.com/logging/docs/agent/installation.
# TODO(yujuhong): Update to a newer Stackdriver agent once it is released to
# support kubernetes metadata properly. The current version does not recognizes
# the local resource key "logging.googleapis.com/local_resource_id", and fails
# to label namespace, pod and container names on the logs.
function InstallAndStart-LoggingAgent {
  # Remove the existing storage.json file if it exists. This is a workaround
  # for the bug where the logging agent cannot start up if the file is
  # corrupted.
  Remove-Item `
      -Force `
      -ErrorAction Ignore `
      ("$STACKDRIVER_ROOT\LoggingAgent\Main\pos\winevtlog.pos\worker0\" +
       "storage.json")

  if (Test-Path $STACKDRIVER_ROOT) {
    # Note: we should reinstall the Stackdriver agent if $REDO_STEPS is true
    # here, but we don't know how to run the installer without it prompting
    # when Stackdriver is already installed. We dumped the strings in the
    # installer binary and searched for flags to do this but found nothing. Oh
    # well.
    Log-Output ("Skip: $STACKDRIVER_ROOT is already present, assuming that " +
                "Stackdriver logging agent is already installed")
    # Restart-Service restarts a running service or starts a not-running
    # service.
    Restart-StackdriverLoggingAgent
    return
  }

  $url = ("https://dl.google.com/cloudagents/windows/" +
          "StackdriverLogging-${STACKDRIVER_VERSION}.exe")
  $tmp_dir = 'C:\stackdriver_tmp'
  New-Item $tmp_dir -ItemType 'directory' -Force | Out-Null
  $installer_file = "${tmp_dir}\StackdriverLogging-${STACKDRIVER_VERSION}.exe"
  MustDownload-File -OutFile $installer_file -URLs $url

  # Start the installer silently. This automatically starts the
  # "StackdriverLogging" service.
  Log-Output 'Invoking Stackdriver installer'
  Start-Process $installer_file -ArgumentList "/S" -Wait

  Start-Process "$STACKDRIVER_ROOT\LoggingAgent\Main\bin\fluent-gem" `
      -ArgumentList "install","fluent-plugin-record-reformer" `
      -Wait

  # Create a configuration file for kubernetes containers.
  # The config.d directory should have already been created automatically, but
  # try creating again just in case.
  New-Item "$STACKDRIVER_ROOT\LoggingAgent\config.d" `
      -ItemType 'directory' `
      -Force | Out-Null
  $FLUENTD_CONFIG | Out-File `
      -FilePath "$STACKDRIVER_ROOT\LoggingAgent\config.d\k8s_containers.conf" `
      -Encoding ASCII

  # Restart the service to pick up the new configurations.
  Restart-StackdriverLoggingAgent
  Remove-Item -Force -Recurse $tmp_dir
}

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
# TODO: Support CRI log format, which requires the multi_format plugin.
<source>
  @type tail
  path /var/log/containers/*.log
  pos_file /var/log/gcp-containers.log.pos
  # Tags at this point are in the format of:
  # reform.var.log.containers.<POD_NAME>_<NAMESPACE_NAME>_<CONTAINER_NAME>-<CONTAINER_ID>.log
  tag reform.*
  format json
  time_key time
  time_format %Y-%m-%dT%H:%M:%S.%NZ
  read_from_head true
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
'@.replace('NODE_NAME', (hostname))


# Export all public functions:
Export-ModuleMember -Function *-*
