. (Join-Path $PSScriptRoot common.ps1)
Install-Module -Name powershell-yaml -RequiredVersion 0.4.1 -Force -Scope CurrentUser
$ymlfiles = Get-ChildItem $RepoRoot -recurse | Where-Object {$_ -like '*.yml'}
$affectedRepos = [System.Collections.ArrayList]::new()

foreach ($file in $ymlfiles)
{
  Write-Host "Verifying '${file}'"
  $ymlContent = Get-Content $file.FullName -Raw
  $ymlObject = ConvertFrom-Yaml $ymlContent -Ordered

  if ($ymlObject.Contains("resources"))
  {
    if ($ymlObject["resources"]["repositories"])
    {
      $repositories = $ymlObject["resources"]["repositories"]
      foreach ($repo in $repositories)
      {
        $repoName = $repo["repository"]
        if (-not ($repo.Contains("ref")))
        {
          $errorMessage = "File: ${file}, Repository: ${repoName}."
          [void]$affectedRepos.Add($errorMessage)
        }
      }
    }
  }
}

if ($affectedRepos.Count -gt 0)
{
    Write-Output "Ref not found in the following Repository Resources."
    foreach ($errorMessage in $affectedRepos)
    {
        Write-Output "`t$errorMessage"
    }
    Write-Output "Please ensure you add a Ref: when using repository resources"
    Write-Output "More Info at https://aka.ms/azsdk/engsys/tools-versioning"
    exit 1
}

Write-Output "All repository resources in yaml files reference a valid tag"