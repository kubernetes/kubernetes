[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [Parameter(Mandatory = $true)]
  [string]$RepoOwner,

  [Parameter(Mandatory = $true)]
  [string]$RepoName,

  [Parameter(Mandatory = $true)]
  [string]$IssueNumber,

  [Parameter(Mandatory = $true)]
  [string]$Comment,

  [Parameter(Mandatory = $true)]
  [string]$AuthToken
)

. (Join-Path $PSScriptRoot common.ps1)

try {
  Add-GithubIssueComment -RepoOwner $RepoOwner -RepoName $RepoName `
  -IssueNumber $IssueNumber -Comment $Comment -AuthToken $AuthToken
}
catch {
  LogError "Add-GithubIssueComment failed with exception:`n$_"
  exit 1
}