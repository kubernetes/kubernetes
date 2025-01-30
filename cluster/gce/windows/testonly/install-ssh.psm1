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
  Library for installing and running Win64-OpenSSH. NOT FOR PRODUCTION USE.

.NOTES
  This module depends on common.psm1. This module depends on third-party code
  which has not been security-reviewed, so it should only be used for test
  clusters. DO NOT USE THIS MODULE FOR PRODUCTION.
#>

# IMPORTANT PLEASE NOTE:
# Any time the file structure in the `windows` directory changes, `windows/BUILD`
# and `k8s.io/release/lib/releaselib.sh` must be manually updated with the changes.
# We HIGHLY recommend not changing the file structure, because consumers of
# Kubernetes releases depend on the release structure remaining stable.

Import-Module -Force C:\common.psm1

$OPENSSH_ROOT = 'C:\Program Files\OpenSSH'
$USER_PROFILE_MODULE = 'C:\user-profile.psm1'
$WRITE_SSH_KEYS_SCRIPT = 'C:\write-ssh-keys.ps1'

# Starts the Win64-OpenSSH services and configures them to automatically start
# on subsequent boots.
function Start_OpenSshServices {
  ForEach ($service in ("sshd", "ssh-agent")) {
    net start ${service}
    Set-Service ${service} -StartupType Automatic
  }
}

# Installs open-ssh using the instructions in
# https://github.com/PowerShell/Win32-OpenSSH/wiki/Install-Win32-OpenSSH.
#
# After installation run StartProcess-WriteSshKeys to fetch ssh keys from the
# metadata server.
function InstallAndStart-OpenSsh {
  if (-not (ShouldWrite-File $OPENSSH_ROOT)) {
    Log-Output "Starting already-installed OpenSSH services"
    Start_OpenSshServices
    return
  }
  elseif (Test-Path $OPENSSH_ROOT) {
    Log-Output ("OpenSSH directory already exists, attempting to run its " +
                "uninstaller before reinstalling")
    powershell.exe `
        -ExecutionPolicy Bypass `
        -File "$OPENSSH_ROOT\OpenSSH-Win64\uninstall-sshd.ps1"
    rm -Force -Recurse $OPENSSH_ROOT\OpenSSH-Win64
  }

  # Download open-ssh.
  # Use TLS 1.2: needed for Invoke-WebRequest downloads from github.com.
  [Net.ServicePointManager]::SecurityProtocol = `
      [Net.SecurityProtocolType]::Tls12
  $url = ("https://github.com/PowerShell/Win32-OpenSSH/releases/download/" +
          "v7.9.0.0p1-Beta/OpenSSH-Win64.zip")
  $ProgressPreference = 'SilentlyContinue'
  Invoke-WebRequest $url -OutFile C:\openssh-win64.zip

  # Unzip and install open-ssh
  Expand-Archive -Force C:\openssh-win64.zip -DestinationPath $OPENSSH_ROOT
  powershell.exe `
      -ExecutionPolicy Bypass `
      -File "$OPENSSH_ROOT\OpenSSH-Win64\install-sshd.ps1"

  # Disable password-based authentication.
  $sshd_config_default = "$OPENSSH_ROOT\OpenSSH-Win64\sshd_config_default"
  $sshd_config = 'C:\ProgramData\ssh\sshd_config'
  New-Item -Force -ItemType Directory -Path "C:\ProgramData\ssh\" | Out-Null
  # SSH config files must be UTF-8 encoded:
  # https://github.com/PowerShell/Win32-OpenSSH/issues/862
  # https://github.com/PowerShell/Win32-OpenSSH/wiki/Various-Considerations
  (Get-Content $sshd_config_default).`
      replace('#PasswordAuthentication yes', 'PasswordAuthentication no') |
      Set-Content -Encoding UTF8 $sshd_config

  # Configure the firewall to allow inbound SSH connections
  if (Get-NetFirewallRule -ErrorAction SilentlyContinue sshd) {
    Get-NetFirewallRule sshd | Remove-NetFirewallRule
  }
  New-NetFirewallRule `
      -Name sshd `
      -DisplayName 'OpenSSH Server (sshd)' `
      -Enabled True `
      -Direction Inbound `
      -Protocol TCP `
      -Action Allow `
      -LocalPort 22

  Start_OpenSshServices
}

function Setup_WriteSshKeysScript {
  if (-not (ShouldWrite-File $WRITE_SSH_KEYS_SCRIPT)) {
    return
  }

  # Fetch helper module for manipulating Windows user profiles.
  if (ShouldWrite-File $USER_PROFILE_MODULE) {
    $module = Get-InstanceMetadataAttribute 'user-profile-psm1'
    New-Item -ItemType file -Force $USER_PROFILE_MODULE | Out-Null
    Set-Content $USER_PROFILE_MODULE $module
  }

  # TODO(pjh): check if we still need to write authorized_keys to users-specific
  # directories, or if just writing to the centralized keys file for
  # Administrators on the system is sufficient (does our log-dump user have
  # Administrator rights?).
  New-Item -Force -ItemType file ${WRITE_SSH_KEYS_SCRIPT} | Out-Null
  Set-Content ${WRITE_SSH_KEYS_SCRIPT} `
'Import-Module -Force USER_PROFILE_MODULE
# For [System.Web.Security.Membership]::GeneratePassword():
Add-Type -AssemblyName System.Web

$poll_interval = 10

# New for v7.9.0.0: administrators_authorized_keys file. For permission
# information see
# https://github.com/PowerShell/Win32-OpenSSH/wiki/Security-protection-of-various-files-in-Win32-OpenSSH#administrators_authorized_keys.
# this file is created only once, each valid user will be added here
$administrator_keys_file = ${env:ProgramData} + `
    "\ssh\administrators_authorized_keys"
New-Item -ItemType file -Force $administrator_keys_file | Out-Null
icacls $administrator_keys_file /inheritance:r | Out-Null
icacls $administrator_keys_file /grant SYSTEM:`(F`) | Out-Null
icacls $administrator_keys_file /grant BUILTIN\Administrators:`(F`) | `
    Out-Null

while($true) {
  $r1 = ""
  $r2 = ""
  # Try both the new "ssh-keys" and the legacy "sshSkeys" attributes for
  # compatibility. The Invoke-RestMethods calls will fail when these attributes
  # do not exist, or they may fail when the connection to the metadata server
  # gets disrupted while we set up container networking on the node.
  try {
    $r1 = Invoke-RestMethod -Headers @{"Metadata-Flavor"="Google"} -Uri `
        "http://metadata.google.internal/computeMetadata/v1/project/attributes/ssh-keys"
  } catch {}
  try {
    $r2 = Invoke-RestMethod -Headers @{"Metadata-Flavor"="Google"} -Uri `
        "http://metadata.google.internal/computeMetadata/v1/project/attributes/sshKeys"
  } catch {}
  $response= $r1 + $r2

  # Split the response into lines; handle both \r\n and \n line breaks.
  $tuples = $response -split "\r?\n"

  $users_to_keys = @{}
  foreach($line in $tuples) {
    if ([string]::IsNullOrEmpty($line)) {
      continue
    }
    # The final parameter to -Split is the max number of strings to return, so
    # this only splits on the first colon.
    $username, $key = $line -Split ":",2

    # Detect and skip keys without associated usernames, which may come back
    # from the legacy sshKeys metadata.
    if (($username -like "ssh-*") -or ($username -like "ecdsa-*")) {
      Write-Error "Skipping key without username: $username"
      continue
    }
    if (-not $users_to_keys.ContainsKey($username)) {
      $users_to_keys[$username] = @($key)
    }
    else {
      $keyList = $users_to_keys[$username]
      $users_to_keys[$username] = $keyList + $key
    }
  }
  $users_to_keys.GetEnumerator() | ForEach-Object {
    $username = $_.key

    # We want to create an authorized_keys file in the user profile directory
    # for each user, but if we create the directory before that user profile
    # has been created first by Windows, then Windows will create a different
    # user profile directory that looks like "<user>.KUBERNETES-MINI" and sshd
    # will look for the authorized_keys file in THAT directory. In other words,
    # we need to create the user first before we can put the authorized_keys
    # file in that user profile directory. The user-profile.psm1 module (NOT
    # FOR PRODUCTION USE!) has Create-NewProfile which achieves this.
    #
    # Run "Get-Command -Module Microsoft.PowerShell.LocalAccounts" to see the
    # build-in commands for users and groups. For some reason the New-LocalUser
    # command does not create the user profile directory, so we use the
    # auxiliary user-profile.psm1 instead.

    $pw = [System.Web.Security.Membership]::GeneratePassword(16,2)
    try {
      # Create-NewProfile will throw these errors:
      #
      # - if the username already exists:
      #
      #   Create-NewProfile : Exception calling "SetInfo" with "0" argument(s):
      #   "The account already exists."
      #
      # - if the username is invalid (e.g. gke-29bd5e8d9ea0446f829d)
      #
      #   Create-NewProfile : Exception calling "SetInfo" with "0" argument(s): "The specified username is invalid.
      #
      # Just catch them and ignore them.
      Create-NewProfile $username $pw -ErrorAction Stop

      # Add the user to the Administrators group, otherwise we will not have
      # privilege when we ssh.
      Add-LocalGroupMember -Group Administrators -Member $username
    } catch {}

    $user_dir = "C:\Users\" + $username
    if (-not (Test-Path $user_dir)) {
      # If for some reason Create-NewProfile failed to create the user profile
      # directory just continue on to the next user.
      return
    }

    # the authorized_keys file is created only once per user
    $user_keys_file = -join($user_dir, "\.ssh\authorized_keys")
    if (-not (Test-Path $user_keys_file)) {
      New-Item -ItemType file -Force $user_keys_file | Out-Null
    }

    ForEach ($ssh_key in $_.value) {
      # authorized_keys and other ssh config files must be UTF-8 encoded:
      # https://github.com/PowerShell/Win32-OpenSSH/issues/862
      # https://github.com/PowerShell/Win32-OpenSSH/wiki/Various-Considerations
      #
      # these files will be append only, only new keys will be added
      $found = Select-String -Path $user_keys_file -Pattern $ssh_key -SimpleMatch
      if ($found -eq $null) {
        Add-Content -Encoding UTF8 $user_keys_file $ssh_key
      }
      $found = Select-String -Path $administrator_keys_file -Pattern $ssh_key -SimpleMatch
      if ($found -eq $null) {
        Add-Content -Encoding UTF8 $administrator_keys_file $ssh_key
      }
    }
  }
  Start-Sleep -sec $poll_interval
}'.replace('USER_PROFILE_MODULE', $USER_PROFILE_MODULE)
  Log-Output ("${WRITE_SSH_KEYS_SCRIPT}:`n" +
              "$(Get-Content -Raw ${WRITE_SSH_KEYS_SCRIPT})")
}

# Starts a background process that retrieves ssh keys from the metadata server
# and writes them to user-specific directories. Intended for use only by test
# clusters!!
#
# While this is running it should be possible to SSH to the Windows node using:
#   gcloud compute ssh <username>@<instance> --zone=<zone>
# or:
#   ssh -i ~/.ssh/google_compute_engine -o 'IdentitiesOnly yes' \
#     <username>@<instance_external_ip>
# or copy files using:
#   gcloud compute scp <username>@<instance>:C:\\path\\to\\file.txt \
#     path/to/destination/ --zone=<zone>
#
# If the username you're using does not already have a project-level SSH key
# (run "gcloud compute project-info describe --flatten
# commonInstanceMetadata.items.ssh-keys" to check), run gcloud compute ssh with
# that username once to add a new project-level SSH key, wait one minute for
# StartProcess-WriteSshKeys to pick it up, then try to ssh/scp again.
function StartProcess-WriteSshKeys {
  Setup_WriteSshKeysScript

  # TODO(pjh): check if such a process is already running before starting
  # another one.
  $write_keys_process = Start-Process `
      -FilePath "powershell.exe" `
      -ArgumentList @("-Command", ${WRITE_SSH_KEYS_SCRIPT}) `
      -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput "NUL" `
      -RedirectStandardError C:\write-ssh-keys.err
  Log-Output "Started background process to write SSH keys"
  Log-Output "$(${write_keys_process} | Out-String)"
}

# Export all public functions:
Export-ModuleMember -Function *-*
