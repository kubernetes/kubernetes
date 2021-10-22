param(
  $RepoOwner,
  $RepoName,
  $BranchPrefix,
  $AuthToken
)

. (Join-Path $PSScriptRoot common.ps1)

LogDebug "Operating on Repo [ $RepoName ]"
try{
  $branches = (Get-GitHubSourceReferences -RepoOwner $RepoOwner -RepoName $RepoName -Ref "heads/$BranchPrefix" -AuthToken $AuthToken).ref 
}
catch {
  LogError "Get-GitHubSourceReferences failed with exception:`n$_"
  exit 1
}

foreach ($branch in $branches)
{
  try {
    $branchName = $branch.Replace("refs/heads/","")
    $head = "${RepoOwner}/${RepoName}:${branchName}"
    LogDebug "Operating on branch [ $branchName ]"
    $pullRequests = Get-GitHubPullRequests -RepoOwner $RepoOwner -RepoName $RepoName -State "all" -Head $head -AuthToken $AuthToken
  }
  catch
  {
    LogError "Get-GitHubPullRequests failed with exception:`n$_"
    exit 1
  }

  $openPullRequests = $pullRequests | ? { $_.State -eq "open" }

  if ($openPullRequests.Count -eq 0)
  {
    LogDebug "Branch [ $branchName ] in repo [ $RepoName ] has no associated open Pull Request. Deleting Branch"
    try{
      Remove-GitHubSourceReferences -RepoOwner $RepoOwner -RepoName $RepoName -Ref ($branch.Remove(0,5)) -AuthToken $AuthToken
    }
    catch {
      LogError "Remove-GitHubSourceReferences failed with exception:`n$_"
      exit 1
    }
  }
}