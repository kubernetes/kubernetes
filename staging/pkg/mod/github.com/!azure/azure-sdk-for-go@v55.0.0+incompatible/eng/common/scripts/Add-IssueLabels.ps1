[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [Parameter(Mandatory = $true)]
  [string]$RepoOwner,

  [Parameter(Mandatory = $true)]
  [string]$RepoName,

  [Parameter(Mandatory = $true)]
  [string]$IssueNumber,

  [Parameter(Mandatory = $true)]
  [string]$Labels,

  [Parameter(Mandatory = $true)]
  [string]$AuthToken
)

. (Join-Path $PSScriptRoot common.ps1)

try {
  Add-GithubIssueLabels -RepoOwner $RepoOwner -RepoName $RepoName `
  -IssueNumber $IssueNumber -Labels $Labels -AuthToken $AuthToken
}
catch {
  LogError "Add-GithubIssueLabels failed with exception:`n$_"
  exit 1
}