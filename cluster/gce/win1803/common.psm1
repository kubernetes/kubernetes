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
  Library containing common variables and code used by other PowerShell modules
  and scripts for configuring Windows nodes.
#>

# REDO_STEPS affects the behavior of a node that is rebooted after initial
# bringup. When true, on a reboot the scripts will redo steps that were
# determined to have already been completed once (e.g. to overwrite
# already-existing config files). When false the scripts will perform the
# minimum required steps to re-join this node to the cluster.
$REDO_STEPS = $false
Export-ModuleMember -Variable REDO_STEPS

# Writes $Message to the console. Terminates the script if $Fatal is set.
function Log-Output {
  param (
    [parameter(Mandatory=$true)] [string]$Message,
    [switch]$Fatal
  )
  Write-Host "${Message}"
  if (${Fatal}) {
    Exit 1
  }
}

# Checks if a file should be written or overwritten by testing if it already
# exists and checking the value of the global $REDO_STEPS variable. Emits an
# informative message if the file already exists.
#
# Returns $true if the file does not exist, or if it does but the global
# $REDO_STEPS variable is set to $true. Returns $false if the file exists and
# the caller should not overwrite it.
function ShouldWrite-File {
  param (
    [parameter(Mandatory=$true)] [string]$Filename
  )
  if (Test-Path $Filename) {
    if ($REDO_STEPS) {
      Log-Output "Warning: $Filename already exists, will overwrite it"
      return $true
    }
    Log-Output "Skip: $Filename already exists, not overwriting it"
    return $false
  }
  return $true
}

# Returns the GCE instance metadata value for $Key. If the key is not present
# in the instance metadata returns $Default if set, otherwise returns $null.
function Get-InstanceMetadataValue {
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
      Log-Output "Failed to retrieve value for $Key."
      return $null
    }
  }
}

# Export all public functions:
Export-ModuleMember -Function *-*
