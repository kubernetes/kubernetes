. (Join-Path $EngCommonScriptsDir SemVer.ps1)

$SDIST_PACKAGE_REGEX = "^(?<package>.*)\-(?<versionstring>$([AzureEngSemanticVersion]::SEMVER_REGEX))"

# Posts a github release for each item of the pkgList variable. SilentlyContinue
function CreateReleases($pkgList, $releaseApiUrl, $releaseSha) {
  foreach ($pkgInfo in $pkgList) {
    Write-Host "Creating release $($pkgInfo.Tag)"

    $releaseNotes = ""
    if ($pkgInfo.ReleaseNotes -ne $null) {
      $releaseNotes = $pkgInfo.ReleaseNotes
    }

    $isPrerelease = $False

    $parsedSemver = [AzureEngSemanticVersion]::ParseVersionString($pkgInfo.PackageVersion)

    if ($parsedSemver) {
      $isPrerelease = $parsedSemver.IsPrerelease
    }

    $url = $releaseApiUrl
    $body = ConvertTo-Json @{
      tag_name         = $pkgInfo.Tag
      target_commitish = $releaseSha
      name             = $pkgInfo.Tag
      draft            = $False
      prerelease       = $isPrerelease
      body             = $releaseNotes
    }

    $headers = @{
      "Content-Type"  = "application/json"
      "Authorization" = "token $($env:GH_TOKEN)"
    }

    Invoke-RestMethod -Uri $url -Body $body -Headers $headers -Method "Post" -MaximumRetryCount 3 -RetryIntervalSec 10
  }
}

# Retrieves the list of all tags that exist on the target repository
function GetExistingTags($apiUrl) {
  try {
    return (Invoke-RestMethod -Method "GET" -Uri "$apiUrl/git/refs/tags" -MaximumRetryCount 3 -RetryIntervalSec 10) | % { $_.ref.Replace("refs/tags/", "") }
  }
  catch {
    Write-Host $_
    $statusCode = $_.Exception.Response.StatusCode.value__
    $statusDescription = $_.Exception.Response.StatusDescription

    Write-Host "Failed to retrieve tags from repository."
    Write-Host "StatusCode:" $statusCode
    Write-Host "StatusDescription:" $statusDescription

    # Return an empty list if there are no tags in the repo
    if ($statusCode -eq 404) {
      return ,@()
    }

    exit(1)
  }
}

# Retrieve release tag for artiface package. If multiple packages, then output the first one.
function RetrieveReleaseTag($artifactLocation, $continueOnError = $true) {
  if (!$artifactLocation) {
    return ""
  }
  try {
    $pkgs, $parsePkgInfoFn = RetrievePackages -artifactLocation $artifactLocation
    if (!$pkgs -or !$pkgs[0]) {
      Write-Host "No packages retrieved from artifact location."
      return ""
    }
    if ($pkgs.Count -gt 1) {
      Write-Host "There are more than 1 packages retieved from artifact location."
      foreach ($pkg in $pkgs) {
        Write-Host "The package name is $($pkg.BaseName)"
      }
      return ""
    }
    $parsedPackage = &$parsePkgInfoFn -pkg $pkgs[0] -workingDirectory $artifactLocation
    return $parsedPackage.ReleaseTag
  }
  catch {
    if ($continueOnError) {
      return ""
    }
    Write-Error "No release tag retrieved from $artifactLocation"
  }
}

function RetrievePackages($artifactLocation) {
  $pkgs = Get-ChildItem -Path $artifactLocation -Include $packagePattern -Recurse -File
  if ($GetPackageInfoFromPackageFileFn -and (Test-Path "Function:$GetPackageInfoFromPackageFileFn"))
  {
    return $pkgs, $GetPackageInfoFromPackageFileFn
  }
  else
  {
    LogError "The function for '$GetPackageInfoFromPackageFileFn' was not found.`
    Make sure it is present in eng/scripts/Language-Settings.ps1 and referenced in eng/common/scripts/common.ps1.`
    See https://github.com/Azure/azure-sdk-tools/blob/master/doc/common/common_engsys.md#code-structure"
  }
}

# Walk across all build artifacts, check them against the appropriate repository, return a list of tags/releases
function VerifyPackages($artifactLocation, $workingDirectory, $apiUrl, $releaseSha,  $continueOnError = $false) {
  $pkgList = [array]@()
  $pkgs, $parsePkgInfoFn = RetrievePackages -artifactLocation $artifactLocation

  foreach ($pkg in $pkgs) {
    try {
      $parsedPackage = &$parsePkgInfoFn -pkg $pkg -workingDirectory $workingDirectory

      if ($parsedPackage -eq $null) {
        continue
      }

      if ($parsedPackage.Deployable -ne $True -and !$continueOnError) {
        Write-Host "Package $($parsedPackage.PackageId) is marked with version $($parsedPackage.PackageVersion), the version $($parsedPackage.PackageVersion) has already been deployed to the target repository."
        Write-Host "Maybe a pkg version wasn't updated properly?"
        exit(1)
      }
      $docsReadMeName = $parsedPackage.PackageId
      if ($parsedPackage.DocsReadMeName) {
        $docsReadMeName = $parsedPackage.DocsReadMeName
      }
      $pkgList += New-Object PSObject -Property @{
        PackageId      = $parsedPackage.PackageId
        PackageVersion = $parsedPackage.PackageVersion
        GroupId        = $parsedPackage.GroupId
        Tag            = $parsedPackage.ReleaseTag
        ReleaseNotes   = $parsedPackage.ReleaseNotes
        ReadmeContent  = $parsedPackage.ReadmeContent
        DocsReadMeName = $docsReadMeName
        IsPrerelease   = [AzureEngSemanticVersion]::ParseVersionString($parsedPackage.PackageVersion).IsPrerelease
      }
    }
    catch {
      Write-Host $_.Exception.Message
      exit(1)
    }
  }

  $results = @([array]$pkgList | Sort-Object -Property Tag -uniq)

  $existingTags = GetExistingTags($apiUrl)
  
  $intersect = $results | % { $_.Tag } | ? { $existingTags -contains $_ }

  if ($intersect.Length -gt 0 -and !$continueOnError) {
    CheckArtifactShaAgainstTagsList -priorExistingTagList $intersect -releaseSha $releaseSha -apiUrl $apiUrl -continueOnError $continueOnError

    # all the tags are clean. remove them from the list of releases we will publish.
    $results = $results | ? { -not ($intersect -contains $_.Tag ) }
  }

  return $results
}

# given a set of tags that we want to release, we need to ensure that if they already DO exist.
# if they DO exist, quietly exit if the commit sha of the artifact matches that of the tag
# if the commit sha does not match, exit with error and report both problem shas
function CheckArtifactShaAgainstTagsList($priorExistingTagList, $releaseSha, $apiUrl, $continueOnError) {
  $headers = @{
    "Content-Type"  = "application/json"
    "Authorization" = "token $($env:GH_TOKEN)"
  }

  $unmatchedTags = @()

  foreach ($tag in $priorExistingTagList) {
    $tagSha = (Invoke-RestMethod -Method "Get" -Uri "$apiUrl/git/refs/tags/$tag" -Headers $headers -MaximumRetryCount 3 -RetryIntervalSec 10)."object".sha

    if ($tagSha -eq $releaseSha) {
      Write-Host "This package has already been released. The existing tag commit SHA $releaseSha matches the artifact SHA being processed. Skipping release step for this tag."
    }
    else {
      Write-Host "The artifact SHA $releaseSha does not match that of the currently existing tag."
      Write-Host "Tag with issues is $tag with commit SHA $tagSha"

      $unmatchedTags += $tag
    }
  }

  if ($unmatchedTags.Length -gt 0 -and !$continueOnError) {
    Write-Host "Tags already existing with different SHA versions. Exiting."
    exit(1)
  }
}
