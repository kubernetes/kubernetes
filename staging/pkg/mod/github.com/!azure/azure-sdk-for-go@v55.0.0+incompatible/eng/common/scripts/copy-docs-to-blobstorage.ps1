# Note, due to how `Expand-Archive` is leveraged in this script,
# powershell core is a requirement for successful execution.
param (
  $AzCopy,
  $DocLocation,
  $SASKey,
  $BlobName,
  $ExitOnError=1,
  $UploadLatest=1,
  $PublicArtifactLocation = "",
  $RepoReplaceRegex = ""
)

. (Join-Path $PSScriptRoot common.ps1)

# Regex inspired but simplified from https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
$SEMVER_REGEX = "^(?<major>0|[1-9]\d*)\.(?<minor>0|[1-9]\d*)\.(?<patch>0|[1-9]\d*)(?:-?(?<prelabel>[a-zA-Z-]*)(?:\.?(?<prenumber>0|[1-9]\d*))?)?$"

function ToSemVer($version){
    if ($version -match $SEMVER_REGEX)
    {
        if(-not $matches['prelabel']) {
            # artifically provide these values for non-prereleases to enable easy sorting of them later than prereleases.
            $prelabel = "zzz"
            $prenumber = 999;
            $isPre = $false;
        }
        else {
            $prelabel = $matches["prelabel"]
            $prenumber = 0

            # some older packages don't have a prenumber, should handle this
            if($matches["prenumber"]){
                $prenumber = [int]$matches["prenumber"]
            }

            $isPre = $true;
        }

        New-Object PSObject -Property @{
            Major = [int]$matches['major']
            Minor = [int]$matches['minor']
            Patch = [int]$matches['patch']
            PrereleaseLabel = $prelabel
            PrereleaseNumber = $prenumber
            IsPrerelease = $isPre
            RawVersion = $version
        }
    }
    else
    {
        if ($ExitOnError)
        {
            throw "Unable to convert $version to valid semver and hard exit on error is enabled. Exiting."
        }
        else
        {
            return $null
        }
    }
}

function SortSemVersions($versions)
{
    return $versions | Sort -Property Major, Minor, Patch, PrereleaseLabel, PrereleaseNumber -Descending
}

function Sort-Versions
{
    Param (
        [Parameter(Mandatory=$true)] [string[]]$VersionArray
    )

    # standard init and sorting existing
    $versionsObject = New-Object PSObject -Property @{
        OriginalVersionArray = $VersionArray
        SortedVersionArray = @()
        LatestGAPackage = ""
        RawVersionsList = ""
        LatestPreviewPackage = ""
    }

    if ($VersionArray.Count -eq 0)
    {
        return $versionsObject
    }

    $versionsObject.SortedVersionArray = @(SortSemVersions -versions ($VersionArray | % { ToSemVer $_}))
    $versionsObject.RawVersionsList = $versionsObject.SortedVersionArray | % { $_.RawVersion }

    # handle latest and preview
    # we only want to hold onto the latest preview if its NEWER than the latest GA.
    # this means that the latest preview package either A) has to be the latest value in the VersionArray
    # or B) set to nothing. We'll handle the set to nothing case a bit later.
    $versionsObject.LatestPreviewPackage = $versionsObject.SortedVersionArray[0].RawVersion
    $gaVersions = $versionsObject.SortedVersionArray | ? { !$_.IsPrerelease }

    # we have a GA package
    if ($gaVersions.Count -ne 0)
    {
        # GA is the newest non-preview package
        $versionsObject.LatestGAPackage = $gaVersions[0].RawVersion

        # in the case where latest preview == latestGA (because of our default selection earlier)
        if ($versionsObject.LatestGAPackage -eq $versionsObject.LatestPreviewPackage)
        {
            # latest is newest, unset latest preview
            $versionsObject.LatestPreviewPackage = ""
        }
    }

    return $versionsObject
}

function Get-Existing-Versions
{
    Param (
        [Parameter(Mandatory=$true)] [String]$PkgName
    )
    $versionUri = "$($BlobName)/`$web/$($Language)/$($PkgName)/versioning/versions"
    LogDebug "Heading to $versionUri to retrieve known versions"

    try {
        return ((Invoke-RestMethod -Uri $versionUri -MaximumRetryCount 3 -RetryIntervalSec 5) -Split "\n" | % {$_.Trim()} | ? { return $_ })
    }
    catch {
        # Handle 404. If it's 404, this is the first time we've published this package.
        if ($_.Exception.Response.StatusCode.value__ -eq 404){
            LogDebug "Version file does not exist. This is the first time we have published this package."
        }
        else {
            # If it's not a 404. exit. We don't know what's gone wrong.
            LogError "Exception getting version file. Aborting"
            LogError $_
            exit(1)
        }
    }
}

function Update-Existing-Versions
{
    Param (
        [Parameter(Mandatory=$true)] [String]$PkgName,
        [Parameter(Mandatory=$true)] [String]$PkgVersion,
        [Parameter(Mandatory=$true)] [String]$DocDest
    )
    $existingVersions = @(Get-Existing-Versions -PkgName $PkgName)

    LogDebug "Before I update anything, I am seeing $existingVersions"

    if (!$existingVersions)
    {
        $existingVersions = @()
        $existingVersions += $PkgVersion
        LogDebug "No existing versions. Adding $PkgVersion."
    }
    else
    {
        $existingVersions += $pkgVersion
        LogDebug "Already Existing Versions. Adding $PkgVersion."
    }

    $existingVersions = $existingVersions | Select-Object -Unique

    # newest first
    $sortedVersionObj = (Sort-Versions -VersionArray $existingVersions)

    LogDebug $sortedVersionObj
    LogDebug $sortedVersionObj.LatestGAPackage
    LogDebug $sortedVersionObj.LatestPreviewPackage

    # write to file. to get the correct performance with "actually empty" files, we gotta do the newline
    # join ourselves. This way we have absolute control over the trailing whitespace.
    $sortedVersionObj.RawVersionsList -join "`n" | Out-File -File "$($DocLocation)/versions" -Force -NoNewLine
    $sortedVersionObj.LatestGAPackage | Out-File -File "$($DocLocation)/latest-ga" -Force -NoNewLine
    $sortedVersionObj.LatestPreviewPackage | Out-File -File "$($DocLocation)/latest-preview" -Force -NoNewLine

    & $($AzCopy) cp "$($DocLocation)/versions" "$($DocDest)/$($PkgName)/versioning/versions$($SASKey)" --cache-control "max-age=300, must-revalidate"
    & $($AzCopy) cp "$($DocLocation)/latest-preview" "$($DocDest)/$($PkgName)/versioning/latest-preview$($SASKey)" --cache-control "max-age=300, must-revalidate"
    & $($AzCopy) cp "$($DocLocation)/latest-ga" "$($DocDest)/$($PkgName)/versioning/latest-ga$($SASKey)" --cache-control "max-age=300, must-revalidate"
    return $sortedVersionObj
}

function Upload-Blobs
{
    Param (
        [Parameter(Mandatory=$true)] [String]$DocDir,
        [Parameter(Mandatory=$true)] [String]$PkgName,
        [Parameter(Mandatory=$true)] [String]$DocVersion,
        [Parameter(Mandatory=$false)] [String]$ReleaseTag
    )
    #eg : $BlobName = "https://azuresdkdocs.blob.core.windows.net"
    $DocDest = "$($BlobName)/`$web/$($Language)"

    LogDebug "DocDest $($DocDest)"
    LogDebug "PkgName $($PkgName)"
    LogDebug "DocVersion $($DocVersion)"
    LogDebug "DocDir $($DocDir)"
    LogDebug "Final Dest $($DocDest)/$($PkgName)/$($DocVersion)"
    LogDebug "Release Tag $($ReleaseTag)"

    # Use the step to replace default branch link to release tag link 
    if ($ReleaseTag) {
        foreach ($htmlFile in (Get-ChildItem $DocDir -include *.html -r)) 
        {
            $fileContent = Get-Content -Path $htmlFile -Raw
            $updatedFileContent = $fileContent -replace $RepoReplaceRegex, "`${1}$ReleaseTag"
            if ($updatedFileContent -ne $fileContent) {
                Set-Content -Path $htmlFile -Value $updatedFileContent -NoNewLine
            }
        }
    } 
    else {
        LogWarning "Not able to do the default branch link replacement, since no release tag found for the release. Please manually check."
    } 
   
    LogDebug "Uploading $($PkgName)/$($DocVersion) to $($DocDest)..."
    & $($AzCopy) cp "$($DocDir)/**" "$($DocDest)/$($PkgName)/$($DocVersion)$($SASKey)" --recursive=true --cache-control "max-age=300, must-revalidate"
    
    LogDebug "Handling versioning files under $($DocDest)/$($PkgName)/versioning/"
    $versionsObj = (Update-Existing-Versions -PkgName $PkgName -PkgVersion $DocVersion -DocDest $DocDest)
    $latestVersion = $versionsObj.LatestGAPackage 
    if (!$latestVersion) {
        $latestVersion = $versionsObj.LatestPreviewPackage 
    }
    LogDebug "Fetching the latest version $latestVersion"
    
    if ($UploadLatest -and ($latestVersion -eq $DocVersion))
    {
        LogDebug "Uploading $($PkgName) to latest folder in $($DocDest)..."
        & $($AzCopy) cp "$($DocDir)/**" "$($DocDest)/$($PkgName)/latest$($SASKey)" --recursive=true --cache-control "max-age=300, must-revalidate"
    }
}

if ($PublishGithubIODocsFn -and (Test-Path "Function:$PublishGithubIODocsFn"))
{
    &$PublishGithubIODocsFn -DocLocation $DocLocation -PublicArtifactLocation $PublicArtifactLocation
}
else
{
    LogWarning "The function for '$PublishGithubIODocsFn' was not found.`
    Make sure it is present in eng/scripts/Language-Settings.ps1 and referenced in eng/common/scripts/common.ps1.`
    See https://github.com/Azure/azure-sdk-tools/blob/master/doc/common/common_engsys.md#code-structure"
}

