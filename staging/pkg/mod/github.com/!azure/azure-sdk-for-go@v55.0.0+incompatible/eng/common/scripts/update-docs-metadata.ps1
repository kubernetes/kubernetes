# Note, due to how `Expand-Archive` is leveraged in this script,
# powershell core is a requirement for successful execution.
param (
  # arguments leveraged to parse and identify artifacts
  [Parameter(Mandatory = $true)]
  $ArtifactLocation, # the root of the artifact folder. DevOps $(System.ArtifactsDirectory)
  [Parameter(Mandatory = $true)]
  $WorkDirectory, # a clean folder that we can work in
  [Parameter(Mandatory = $true)]
  $ReleaseSHA, # the SHA for the artifacts. DevOps: $(Release.Artifacts.<artifactAlias>.SourceVersion) or $(Build.SourceVersion)
  [Parameter(Mandatory = $true)]
  $RepoId, # full repo id. EG azure/azure-sdk-for-net  DevOps: $(Build.Repository.Id). Used as a part of VerifyPackages
  [Parameter(Mandatory = $true)]
  $Repository, # EG: "Maven", "PyPI", "NPM"

  # arguments necessary to power the docs release
  [Parameter(Mandatory = $true)]
  $DocRepoLocation, # the location on disk where we have cloned the documentation repository
  [Parameter(Mandatory = $true)]
  $Language, # EG: js, java, dotnet. Used in language for the embedded readme.
  [Parameter(Mandatory = $true)]
  $Configs # The configuration elements informing important locations within the cloned doc repo
)

. (Join-Path $PSScriptRoot common.ps1)

$releaseReplaceRegex = "(https://github.com/$RepoId/(?:blob|tree)/)master"

function GetMetaData {
  if (Test-Path Variable:MetadataUri) {
    $metadataResponse = Invoke-RestMethod -Uri $MetadataUri -method "GET" -MaximumRetryCount 3 -RetryIntervalSec 10 | ConvertFrom-Csv
  }
  else {
    LogError "The variable '$MetadataUri' was not found."
  }

  return $metadataResponse
}

function GetAdjustedReadmeContent($pkgInfo){
    $date = Get-Date -Format "MM/dd/yyyy"
    $service = ""

    # the namespace is not expected to be present for js.
    $pkgId = $pkgInfo.PackageId.Replace("@azure/", "")

    try {
      $metadata = GetMetaData

      $service = $metadata | ? { $_.Package -eq $pkgId }

      if ($service) {
        $service = "$($service.ServiceName)".ToLower().Replace(" ", "")
      }
    }
    catch {
      Write-Host $_
      Write-Host "Unable to retrieve service metadata for packageId $($pkgInfo.PackageId)"
    }

    $fileContent = $pkgInfo.ReadmeContent

    # only replace the version if the formatted header can be found
    $titleRegex = "(\#\s+(?<filetitle>Azure .+? (?:client|plugin|shared) library for (?:JavaScript|Java|Python|\.NET|C)))"
    $foundTitle = ""
    if ($fileContent -match $titleRegex) {
      $fileContent = $fileContent -replace $titleRegex, "`${0} - Version $($pkgInfo.PackageVersion) `n"
      $foundTitle = $matches["filetitle"]
    }
    # Replace github master link with release tag.
    $ReplacementPattern = "`${1}$($pkgInfo.Tag)"
    $fileContent = $fileContent -replace $releaseReplaceRegex, $ReplacementPattern
  
    $header = "---`ntitle: $foundTitle`nkeywords: Azure, $Language, SDK, API, $($pkgInfo.PackageId), $service`nauthor: maggiepint`nms.author: magpint`nms.date: $date`nms.topic: article`nms.prod: azure`nms.technology: azure`nms.devlang: $Language`nms.service: $service`n---`n"

    if ($fileContent) {
      return "$header`n$fileContent"
    }
    else {
      return ""
    }
}

$apiUrl = "https://api.github.com/repos/$repoId"
$pkgs = VerifyPackages -artifactLocation $ArtifactLocation `
  -workingDirectory $WorkDirectory `
  -apiUrl $apiUrl `
  -releaseSha $ReleaseSHA `
  -continueOnError $True

$targets = ($Configs | ConvertFrom-Json).targets

foreach ($config in $targets) {
  if ($config.mode -eq "Preview") { $includePreview = $true } else { $includePreview = $false }
  $pkgsFiltered = $pkgs | ? { $_.IsPrerelease -eq $includePreview}
  $suffix = ""
  if ($config.suffix) {
    $suffix = $config.suffix
  }

  if ($pkgsFiltered) {
    Write-Host "Given the visible artifacts, $($config.mode) Readme updates against $($config.path_to_config) will be processed for the following packages."
    Write-Host ($pkgsFiltered | % { $_.PackageId + " " + $_.PackageVersion })
  
    foreach ($packageInfo in $pkgsFiltered) {
      $readmeName = "$($packageInfo.DocsReadMeName.ToLower())-readme${suffix}.md"
      $readmeFolder = Join-Path $DocRepoLocation $config.content_folder
      $readmeLocation = Join-Path $readmeFolder $readmeName

      # what happens if this is the first time we've written to this folder? It won't exist. Resolve that.
      if(!(Test-Path $readmeFolder)) {
        New-Item -ItemType Directory -Force -Path $readmeFolder
      }

      if ($packageInfo.ReadmeContent) {
        $adjustedContent = GetAdjustedReadmeContent -pkgInfo $packageInfo
      }
  
      if ($adjustedContent) {
        try {
          Push-Location $DocRepoLocation
          Set-Content -Path $readmeLocation -Value $adjustedContent -Force
  
          Write-Host "Updated readme for $readmeName."
        } catch {
          Write-Host $_
        } finally {
          Pop-Location
        }
      } else {
        Write-Host "Unable to parse a header out of the readmecontent for PackageId $($packageInfo.PackageId)"
      }
    }
  }
  else {
    Write-Host "No readmes discovered for $($config.mode) doc release under folder $ArtifactLocation."
  }


}



