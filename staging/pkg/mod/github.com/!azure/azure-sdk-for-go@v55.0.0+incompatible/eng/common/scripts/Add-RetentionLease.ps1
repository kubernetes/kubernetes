[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [Parameter(Mandatory = $true)]
  [string]$Organization,

  [Parameter(Mandatory = $true)]
  [string]$Project,

  [Parameter(Mandatory = $true)]
  [int]$DefinitionId,

  [Parameter(Mandatory = $true)]
  [int]$RunId,

  [Parameter(Mandatory = $true)]
  [string]$OwnerId,

  [Parameter(Mandatory = $true)]
  [int]$DaysValid,

  [Parameter(Mandatory = $true)]
  [string]$AccessToken
)

$unencodedAuthToken = "nobody:$AccessToken"
$unencodedAuthTokenBytes = [System.Text.Encoding]::UTF8.GetBytes($unencodedAuthToken)
$encodedAuthToken = [System.Convert]::ToBase64String($unencodedAuthTokenBytes)

# We are doing this here so that there is zero chance that this token is emitted in Azure Pipelines
# build logs. Azure Pipelines will see this text and register the secret as a value it should *** out
# before being transmitted to the server (and shown in logs). It means if the value is accidentally
# leaked anywhere else that it won't be visible. The downside is that when the script is executed
# on a local development box, it will be visible.
Write-Host "##vso[task.setvariable variable=_throwawayencodedaccesstoken;issecret=true;]$($encodedAuthToken)"

. (Join-Path $PSScriptRoot common.ps1)

LogDebug "Checking for existing leases on run: $RunId"
$existingLeases = Get-RetentionLeases -Organization $Organization -Project $Project -DefinitionId $DefinitionId -RunId $RunId -OwnerId $OwnerId -Base64EncodedAuthToken $encodedAuthToken

if ($existingLeases.count -ne 0) {
    LogDebug "Found $($existingLeases.count) leases, will delete them first."

    foreach ($lease in $existingLeases.value) {
        LogDebug "Deleting lease: $($lease.leaseId)"
        Delete-RetentionLease -Organization $Organization -Project $Project -LeaseId $lease.leaseId -Base64EncodedAuthToken $encodedAuthToken
    }

}

LogDebug "Creating new lease on run: $RunId"
$lease = Add-RetentionLease -Organization $Organization -Project $Project -DefinitionId $DefinitionId -RunId $RunId -OwnerId $OwnerId -DaysValid $DaysValid -Base64EncodedAuthToken $encodedAuthToken
LogDebug "Lease ID is: $($lease.value.leaseId)"