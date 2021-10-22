param (
    [Parameter(Mandatory = $true)]
    [string]$RepoOwner,

    [Parameter(Mandatory = $true)]
    [string]$RepoName,

    [Parameter(Mandatory = $true)]
    $PullRequestNumber,

    [Parameter(Mandatory = $true)]
    [string]$AuthToken
)

. (Join-Path $PSScriptRoot common.ps1)

try
{
    $pullRequest = Get-GithubPullRequest -RepoOwner $RepoOwner -RepoName $RepoName `
    -PullRequestNumber $PullRequestNumber -AuthToken $AuthToken
    Write-Host "##vso[task.setvariable variable=System.PullRequest.Creator;]$($pullRequest.user.login)"
}
catch
{
    Write-Error "Get-PullRequest failed with exception:`n$_"
    exit 1
}

