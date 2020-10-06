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

# IMPORTANT PLEASE NOTE:
# Any time the file structure in the `windows` directory changes,
# `windows/BUILD` and `k8s.io/release/lib/releaselib.sh` must be manually
# updated with the changes.
# We HIGHLY recommend not changing the file structure, because consumers of
# Kubernetes releases depend on the release structure remaining stable.

# Disable progress bar to increase download speed.
$ProgressPreference = 'SilentlyContinue'

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
function Get-InstanceMetadata {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$false)] [string]$Default
  )

  $url = "http://metadata.google.internal/computeMetadata/v1/instance/$Key"
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

# Returns the GCE instance metadata value for $Key where key is an "attribute"
# of the instance. If the key is not present in the instance metadata returns
# $Default if set, otherwise returns $null.
function Get-InstanceMetadataAttribute {
  param (
    [parameter(Mandatory=$true)] [string]$Key,
    [parameter(Mandatory=$false)] [string]$Default
  )

  return Get-InstanceMetadata "attributes/$Key" $Default
}

function Validate-SHA {
  param(
    [parameter(Mandatory=$true)] [string]$Hash,
    [parameter(Mandatory=$true)] [string]$Path,
    [parameter(Mandatory=$true)] [string]$Algorithm
  )
  $actual = Get-FileHash -Path $Path -Algorithm $Algorithm
  # Note: Powershell string comparisons are case-insensitive by default, and this
  # is important here because Linux shell scripts produce lowercase hashes but
  # Powershell Get-FileHash produces uppercase hashes. This must be case-insensitive
  # to work.
  if ($actual.Hash -ne $Hash) {
    Log-Output "$Path corrupted, $Algorithm $actual doesn't match expected $Hash"
    Throw ("$Path corrupted, $Algorithm $actual doesn't match expected $Hash")
  }
}

# Attempts to download the file from URLs, trying each URL until it succeeds.
# It will loop through the URLs list forever until it has a success. If
# successful, it will write the file to OutFile. You can optionally provide a
# Hash argument with an optional Algorithm, in which case it will attempt to
# validate the downloaded file against the hash. SHA512 will be used if Algorithm
# is not provided.
function MustDownload-File {
  param (
    [parameter(Mandatory=$false)] [string]$Hash,
    [parameter(Mandatory=$false)] [string]$Algorithm = 'SHA512',
    [parameter(Mandatory=$true)] [string]$OutFile,
    [parameter(Mandatory=$true)] [System.Collections.Generic.List[String]]$URLs,
    [parameter(Mandatory=$false)] [System.Collections.IDictionary]$Headers = @{}
  )

  While($true) {
    ForEach($url in $URLs) {
      # If the URL is for GCS and the node has dev storage scope, add the
      # service account token to the request headers.
      if (($url -match "^https://storage`.googleapis`.com.*") -and $(Check-StorageScope)) {
        $Headers["Authorization"] = "Bearer $(Get-Credentials)"
      }

      # Attempt to download the file
      Try {
        # TODO(mtaufen): When we finally get a Windows version that has Powershell 6
        # installed we can set `-MaximumRetryCount 6 -RetryIntervalSec 10` to make this even more robust.
        $result = Invoke-WebRequest $url -Headers $Headers -OutFile $OutFile -TimeoutSec 300
      } Catch {
        $message = $_.Exception.ToString()
        Log-Output "Failed to download file from $url. Will retry. Error: $message"
        continue
      }
      # Attempt to validate the hash
      if ($Hash) {
        Try {
            Validate-SHA -Hash $Hash -Path $OutFile -Algorithm $Algorithm
        } Catch {
            $message = $_.Exception.ToString()
            Log-Output "Hash validation of $url failed. Will retry. Error: $message"
            continue
        }
        Log-Output "Downloaded $url ($Algorithm = $Hash)"
        return
      }
      Log-Output "Downloaded $url"
      return
    }
  }
}

# Returns the default service account token for the VM, retrieved from
# the instance metadata.
function Get-Credentials {
  While($true) {
    $data = Get-InstanceMetadata -Key "service-accounts/default/token"
    if ($data) {
      return ($data | ConvertFrom-Json).access_token
    }
    Start-Sleep -Seconds 1
  }
}

# Returns True if the VM has the dev storage scope, False otherwise.
function Check-StorageScope {
  While($true) {
    $data = Get-InstanceMetadata -Key "service-accounts/default/scopes"
    if ($data) {
      return ($data -match "auth/devstorage")
    }
    Start-Sleep -Seconds 1
  }
}

# This compiles some C# code that can make syscalls, and pulls the
# result into our powershell environment so we can make syscalls from this script.
# We make syscalls directly, because whatever the powershell cmdlets do under the hood,
# they can't seem to open the log files concurrently with writers.
# See https://docs.microsoft.com/en-us/dotnet/framework/interop/marshaling-data-with-platform-invoke
# for details on which unmanaged types map to managed types.
$SyscallDefinitions = @'
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern IntPtr CreateFileW(
  String lpFileName,
  UInt32 dwDesiredAccess,
  UInt32 dwShareMode,
  IntPtr lpSecurityAttributes,
  UInt32 dwCreationDisposition,
  UInt32 dwFlagsAndAttributes,
  IntPtr hTemplateFile
);

[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern bool SetFilePointer(
  IntPtr hFile,
  Int32  lDistanceToMove,
  IntPtr lpDistanceToMoveHigh,
  UInt32 dwMoveMethod
);

[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern bool SetEndOfFile(
  IntPtr hFile
);

[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern bool CloseHandle(
  IntPtr hObject
);
'@
$Kernel32 = Add-Type -MemberDefinition $SyscallDefinitions -Name 'Kernel32' -Namespace 'Win32' -PassThru

# Close-Handle closes the specified open file handle.
# On failure, throws an exception.
function Close-Handle {
  param (
    [parameter(Mandatory=$true)] [System.IntPtr]$Handle
  )
  $ret = $Kernel32::CloseHandle($Handle)
  $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
  if (-not $ret) {
    throw "Failed to close open file handle ${Handle}, system error code: ${err}"
  }
}

# Open-File tries to open the file at the specified path with ReadWrite access mode and ReadWrite file share mode.
# On success, returns an open file handle.
# On failure, throws an exception.
function Open-File {
  param (
    [parameter(Mandatory=$true)] [string]$Path
  )

  $lpFileName = $Path
  $dwDesiredAccess = [System.IO.FileAccess]::ReadWrite
  $dwShareMode = [System.IO.FileShare]::ReadWrite # Fortunately golang also passes these same flags when it creates the log files, so we can open it concurrently.
  $lpSecurityAttributes = [System.IntPtr]::Zero
  $dwCreationDisposition = [System.IO.FileMode]::Open
  $dwFlagsAndAttributes = [System.IO.FileAttributes]::Normal
  $hTemplateFile = [System.IntPtr]::Zero

  $handle = $Kernel32::CreateFileW($lpFileName, $dwDesiredAccess, $dwShareMode, $lpSecurityAttributes, $dwCreationDisposition, $dwFlagsAndAttributes, $hTemplateFile)
  $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
  if ($handle -eq -1) {
    throw "Failed to open file ${Path}, system error code: ${err}"
  }

  return $handle
}

# Truncate-File truncates the file in-place by opening it, moving the file pointer to the beginning,
# and setting the end of file to the file pointer's location.
# On failure, throws an exception.
# The file must have been originally created with FILE_SHARE_WRITE for this to be possible.
# Fortunately Go creates files with FILE_SHARE_READ|FILE_SHARE_WRITE by for all os.Open calls,
# so our log writers should be doing the right thing.
function Truncate-File {
  param (
    [parameter(Mandatory=$true)] [string]$Path
  )
  $INVALID_SET_FILE_POINTER = 0xffffffff
  $NO_ERROR = 0
  $FILE_BEGIN = 0

  $handle = Open-File -Path $Path

  # https://docs.microsoft.com/en-us/windows/desktop/api/fileapi/nf-fileapi-setfilepointer
  # Docs: Because INVALID_SET_FILE_POINTER is a valid value for the low-order DWORD of the new file pointer,
  # you must check both the return value of the function and the error code returned by GetLastError to
  # determine whether or not an error has occurred. If an error has occurred, the return value of SetFilePointer
  # is INVALID_SET_FILE_POINTER and GetLastError returns a value other than NO_ERROR.
  $ret = $Kernel32::SetFilePointer($handle, 0, [System.IntPtr]::Zero, $FILE_BEGIN)
  $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
  if ($ret -eq $INVALID_SET_FILE_POINTER -and $err -ne $NO_ERROR) {
    Close-Handle -Handle $handle
    throw "Failed to set file pointer for handle ${handle}, system error code: ${err}"
  }

  $ret = $Kernel32::SetEndOfFile($handle)
  $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
  if ($ret -eq 0) {
    Close-Handle -Handle $handle
    throw "Failed to set end of file for handle ${handle}, system error code: ${err}"
  }
  Close-Handle -Handle $handle
}

# FileRotationConfig defines the common options for file rotation.
class FileRotationConfig {
  # Force rotation, ignoring $MaxBackupInterval and $MaxSize criteria.
  [bool]$Force
  # Maximum time since last backup, after which file will be rotated.
  # When no backups exist, Rotate-File acts as if -MaxBackupInterval has not elapsed,
  # instead relying on the other criteria.
  [TimeSpan]$MaxBackupInterval
  # Maximum file size, after which file will be rotated.
  [int]$MaxSize
  # Maximum number of backup archives to maintain.
  [int]$MaxBackups
}

# New-FileRotationConfig constructs a FileRotationConfig with default options.
function New-FileRotationConfig {
  param (
    # Force rotation, ignoring $MaxBackupInterval and $MaxSize criteria.
    [parameter(Mandatory=$false)] [switch]$Force,
    # Maximum time since last backup, after which file will be rotated.
    # When no backups exist, Rotate-File acts as if -MaxBackupInterval has not elapsed,
    # instead relying on the other criteria.
    # Defaults to daily rotations.
    [parameter(Mandatory=$false)] [TimeSpan]$MaxBackupInterval = $(New-TimeSpan -Day 1),
    # Maximum file size, after which file will be rotated.
    [parameter(Mandatory=$false)] [int]$MaxSize = 100mb,
    # Maximum number of backup archives to maintain.
    [parameter(Mandatory=$false)] [int]$MaxBackups = 5
  )
  $config = [FileRotationConfig]::new()
  $config.Force = $Force
  $config.MaxBackupInterval = $MaxBackupInterval
  $config.MaxSize = $MaxSize
  $config.MaxBackups = $MaxBackups
  return $config
}

# Get-Backups returns a list of paths to backup files for the original file path -Path,
# assuming that backup files are in the same directory, with a prefix matching
# the original file name and a .zip suffix.
function Get-Backups {
  param (
    # Original path of the file for which backups were created (no suffix).
    [parameter(Mandatory=$true)] [string]$Path
  )
  $parent = Split-Path -Parent -Path $Path
  $leaf = Split-Path -Leaf -Path $Path
  $files = Get-ChildItem -File -Path $parent |
           Where-Object Name -like "${leaf}*.zip"
  return $files
}

# Trim-Backups deletes old backups for the log file identified by -Path until only -Count remain.
# Deletes backups with the oldest CreationTime first.
function Trim-Backups {
  param (
    [parameter(Mandatory=$true)] [int]$Count,
    [parameter(Mandatory=$true)] [string]$Path
  )
  if ($Count -lt 0) {
    $Count = 0
  }
  # If creating a new backup will exceed $Count, delete the oldest files
  # until we have one less than $Count, leaving room for the new one.
  # If the pipe results in zero items, $backups is $null, and if it results
  # in only one item, PowerShell doesn't wrap in an array, so we check both cases.
  # In the latter case, this actually caused it to often trim all backups, because
  # .Length is also a property of FileInfo (size of the file)!
  $backups = Get-Backups -Path $Path | Sort-Object -Property CreationTime
  if ($backups -and $backups.GetType() -eq @().GetType() -and $backups.Length -gt $Count) {
    $num = $backups.Length - $Count
    $rmFiles = $backups | Select-Object -First $num
    ForEach ($file in $rmFiles) {
      Remove-Item $file.FullName
    }
  }
}

# Backup-File creates a copy of the file at -Path.
# The name of the backup is the same as the file,
# with the suffix "-%Y%m%d-%s" to identify the time of the backup.
# Returns the path to the backup file.
function Backup-File {
  param (
    [parameter(Mandatory=$true)] [string]$Path
  )
  $date = Get-Date -UFormat "%Y%m%d-%s"
  $dest = "${Path}-${date}"
  Copy-Item -Path $Path -Destination $dest
  return $dest
}

# Compress-BackupFile creates a compressed archive containing the file
# at -Path and subsequently deletes the file at -Path. We split backup
# and compression steps to minimize time between backup and truncation,
# which helps minimize log loss.
function Compress-BackupFile {
  param (
    [parameter(Mandatory=$true)] [string]$Path
  )
  Compress-Archive -Path $Path -DestinationPath "${Path}.zip"
  Remove-Item -Path $Path
}

# Rotate-File rotates the log file at -Path by first making a compressed copy of the original
# log file with the suffix "-%Y%m%d-%s" to identify the time of the backup, then truncating
# the original file in-place. Rotation is performed according to the options in -Config.
function Rotate-File {
  param (
    # Path to the log file to rotate.
    [parameter(Mandatory=$true)] [string]$Path,
    # Config for file rotation.
    [parameter(Mandatory=$true)] [FileRotationConfig]$Config
  )
  function rotate {
    # If creating a new backup will exceed $MaxBackups, delete the oldest files
    # until we have one less than $MaxBackups, leaving room for the new one.
    Trim-Backups -Count ($Config.MaxBackups - 1) -Path $Path

    $backupPath = Backup-File -Path $Path
    Truncate-File -Path $Path
    Compress-BackupFile -Path $backupPath
  }

  # Check Force
  if ($Config.Force) {
    rotate
    return
  }

  # Check MaxSize.
  $file = Get-Item $Path
  if ($file.Length -gt $Config.MaxSize) {
    rotate
    return
  }

  # Check MaxBackupInterval.
  $backups = Get-Backups -Path $Path | Sort-Object -Property CreationTime
  if ($backups.Length -ge 1) {
    $lastBackupTime = $backups[0].CreationTime
    $now = Get-Date
    if ($now - $lastBackupTime -gt $Config.MaxBackupInterval) {
      rotate
      return
    }
  }
}

# Rotate-Files rotates the log files in directory -Path that match -Pattern.
# Rotation is performed by Rotate-File, according to -Config.
function Rotate-Files {
  param (
    # Pattern that file names must match to be rotated. Does not include parent path.
    [parameter(Mandatory=$true)] [string]$Pattern,
    # Path to the log directory containing files to rotate.
    [parameter(Mandatory=$true)] [string]$Path,
    # Config for file rotation.
    [parameter(Mandatory=$true)] [FileRotationConfig]$Config

  )
  $files = Get-ChildItem -File -Path $Path | Where-Object Name -match $Pattern
  ForEach ($file in $files) {
    try {
      Rotate-File -Path $file.FullName -Config $Config
    } catch {
      Log-Output "Caught exception rotating $($file.FullName): $($_.Exception)"
    }
  }
}

# Schedule-LogRotation schedules periodic log rotation with the Windows Task Scheduler.
# Rotation is performed by Rotate-Files, according to -Pattern and -Config.
# The system will check whether log files need to be rotated at -RepetitionInterval.
function Schedule-LogRotation {
  param (
    # Pattern that file names must match to be rotated. Does not include parent path.
    [parameter(Mandatory=$true)] [string]$Pattern,
    # Path to the log directory containing files to rotate.
    [parameter(Mandatory=$true)] [string]$Path,
    # Interval at which to check logs against rotation criteria.
    # Minimum 1 minute, maximum 31 days (see https://docs.microsoft.com/en-us/windows/desktop/taskschd/taskschedulerschema-interval-repetitiontype-element).
    [parameter(Mandatory=$true)] [TimeSpan]$RepetitionInterval,
    # Config for file rotation.
    [parameter(Mandatory=$true)] [FileRotationConfig]$Config
  )
  # Write a powershell script to a file that imports this module ($PSCommandPath)
  # and calls Rotate-Files with the configured arguments.
  $scriptPath = "C:\rotate-kube-logs.ps1"
  New-Item -Force -ItemType file -Path $scriptPath | Out-Null
  Set-Content -Path $scriptPath @"
`$ErrorActionPreference = 'Stop'
Import-Module -Force ${PSCommandPath}
`$maxBackupInterval = New-Timespan -Days $($Config.MaxBackupInterval.Days) -Hours $($Config.MaxBackupInterval.Hours) -Minutes $($Config.MaxBackupInterval.Minutes) -Seconds $($Config.MaxBackupInterval.Seconds)
`$config = New-FileRotationConfig -Force:`$$($Config.Force) -MaxBackupInterval `$maxBackupInterval -MaxSize $($Config.MaxSize) -MaxBackups $($Config.MaxBackups)
Rotate-Files -Pattern '${Pattern}' -Path '${Path}' -Config `$config
"@
  # The task will execute the rotate-kube-logs.ps1 script created above.
  # We explicitly set -WorkingDirectory to $Path for safety's sake, otherwise
  # it runs in %windir%\system32 by default, which sounds dangerous.
  $action = New-ScheduledTaskAction -Execute "powershell" -Argument "-NoLogo -NonInteractive -File ${scriptPath}" -WorkingDirectory $Path
  # Start the task immediately, and trigger the task once every $RepetitionInterval.
  $trigger = New-ScheduledTaskTrigger -Once -At $(Get-Date) -RepetitionInterval $RepetitionInterval
  # Run the task as the same user who is currently running this script.
  $principal = New-ScheduledTaskPrincipal $([System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
  # Just use the default task settings.
  $settings = New-ScheduledTaskSettingsSet
  # Create the ScheduledTask object from the above parameters.
  $task = New-ScheduledTask -Action $action -Principal $principal -Trigger $trigger -Settings $settings -Description "Rotate Kubernetes logs"
  # Register the new ScheduledTask with the Task Scheduler.
  # Always try to unregister and re-register, in case it already exists (e.g. across reboots).
  $name = "RotateKubeLogs"
  try {
    Unregister-ScheduledTask -Confirm:$false -TaskName $name
  } catch {} finally {
    Register-ScheduledTask -TaskName $name -InputObject $task
  }
}

# Returns true if this node is part of a test cluster (see
# cluster/gce/config-test.sh). $KubeEnv is a hash table containing the kube-env
# metadata keys+values.
function Test-IsTestCluster {
  param (
    [parameter(Mandatory=$true)] [hashtable]$KubeEnv
  )

  if ($KubeEnv.Contains('TEST_CLUSTER') -and `
      ($KubeEnv['TEST_CLUSTER'] -eq 'true')) {
    return $true
  }
  return $false
}

# Returns true if this node uses a plugin to support authentication to the
# master, e.g. for TPM-based authentication. $KubeEnv is a hash table
# containing the kube-env metadata keys+values.
function Test-NodeUsesAuthPlugin {
  param (
    [parameter(Mandatory=$true)] [hashtable]$KubeEnv
  )

  return $KubeEnv.Contains('EXEC_AUTH_PLUGIN_URL')
}

# Export all public functions:
Export-ModuleMember -Function *-*
