# ASSUMPTIONS
# * that `npm` cli is present for querying available npm packages
# * that an environment variable $env:GH_TOKEN is populated with the appropriate PAT to allow pushing of github releases

param (
  # used by VerifyPackages
  $artifactLocation, # the root of the artifact folder. DevOps $(System.ArtifactsDirectory)
  $workingDirectory, # directory that package artifacts will be extracted into for examination (if necessary)
  $packageRepository, # used to indicate destination against which we will check the existing version.
  # valid options: PyPI, Nuget, NPM, Maven, C, CPP
  # used by CreateTags
  $releaseSha, # the SHA for the artifacts. DevOps: $(Release.Artifacts.<artifactAlias>.SourceVersion) or $(Build.SourceVersion)

  # used by Git Release
  $repoOwner = "", # the owning organization of the repository. EG "Azure"
  $repoName = "", # the name of the repository. EG "azure-sdk-for-java"
  $repoId = "$repoOwner/$repoName", # full repo id. EG azure/azure-sdk-for-net  DevOps: $(Build.Repository.Id),
  [switch]$continueOnError = $false
)

. (Join-Path $PSScriptRoot common.ps1)

$apiUrl = "https://api.github.com/repos/$repoId"
Write-Host "Using API URL $apiUrl"

# VERIFY PACKAGES
$pkgList = VerifyPackages -artifactLocation $artifactLocation -workingDirectory $workingDirectory -apiUrl $apiUrl -releaseSha $releaseSha -continueOnError $continueOnError

if ($pkgList) {
  Write-Host "Given the visible artifacts, github releases will be created for the following:"

  foreach ($packageInfo in $pkgList) {
    Write-Host $packageInfo.Tag
  }

  # CREATE TAGS and RELEASES
  CreateReleases -pkgList $pkgList -releaseApiUrl $apiUrl/releases -releaseSha $releaseSha
}
else {
  Write-Host "After processing, no packages required release."
}
