[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [Parameter(Mandatory = $true)]
  [string]$Organization,

  [Parameter(Mandatory = $true)]
  [string]$Project,

  [Parameter(Mandatory = $true)]
  [string]$SourceBranch,

  [Parameter(Mandatory = $true)]
  [int]$DefinitionId,

  [boolean]$CancelPreviousBuilds=$false,

  [Parameter(Mandatory = $false)]
  [string]$VsoQueuedPipelines,

  [Parameter(Mandatory = $true)]
  [string]$Base64EncodedAuthToken
)

. (Join-Path $PSScriptRoot common.ps1)

if ($CancelPreviousBuilds)
{
  try {
    $queuedBuilds = Get-DevOpsBuilds -BranchName "refs/heads/$SourceBranch" -Definitions $DefinitionId `
    -StatusFilter "inProgress, notStarted" -Base64EncodedAuthToken $Base64EncodedAuthToken

    if ($queuedBuilds.count -eq 0) {
      LogDebug "There is no previous build still inprogress or about to start."
    }

    foreach ($build in $queuedBuilds.Value) {
      $buildID = $build.id
      LogDebug "Canceling build [ $($build._links.web.href) ]"
      Update-DevOpsBuild -BuildId $buildID -Status "cancelling" -Base64EncodedAuthToken $Base64EncodedAuthToken
    }
  }
  catch {
    LogError "Call to DevOps API failed with exception:`n$_"
    exit 1
  }
}

try {
  $resp = Start-DevOpsBuild -SourceBranch $SourceBranch -DefinitionId $DefinitionId -Base64EncodedAuthToken $Base64EncodedAuthToken
}
catch {
  LogError "Start-DevOpsBuild failed with exception:`n$_"
  exit 1
}

LogDebug "Pipeline [ $($resp.definition.name) ] queued at [ $($resp._links.web.href) ]"

if ($VsoQueuedPipelines) {
  $enVarValue = [System.Environment]::GetEnvironmentVariable($VsoQueuedPipelines)
  $QueuedPipelineLinks = if ($enVarValue) { 
    "$enVarValue<br>[$($resp.definition.name)]($($resp._links.web.href))"
  }else {
    "[$($resp.definition.name)]($($resp._links.web.href))"
  }
  $QueuedPipelineLinks
  Write-Host "##vso[task.setvariable variable=$VsoQueuedPipelines]$QueuedPipelineLinks"
}