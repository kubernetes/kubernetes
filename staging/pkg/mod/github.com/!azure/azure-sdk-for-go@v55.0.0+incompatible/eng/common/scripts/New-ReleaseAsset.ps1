<#
.SYNOPSIS
Uploads the release asset and returns the resulting object from the upload

.PARAMETER ReleaseTag
Tag to look up release

.PARAMETER AssetPath
Location of the asset file to upload

.PARAMETER GitHubRepo
Name of the GitHub repo to search (of the form Azure/azure-sdk-for-cpp)

#>

param (
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ReleaseTag,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $AssetPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $GitHubRepo,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $GitHubPat
)

# Get information about release at $ReleaseTag
$releaseInfoUrl = "https://api.github.com/repos/$GitHubRepo/releases/tags/$ReleaseTag"
Write-Verbose "Requesting release info from $releaseInfoUrl"
$release = Invoke-RestMethod `
    -Uri  $releaseInfoUrl `
    -Method GET

$assetFilename = Split-Path $AssetPath -Leaf

# Upload URL comes in the literal form (yes, those curly braces) of:
# https://uploads.github.com/repos/Azure/azure-sdk-for-cpp/releases/123/assets{?name,label}
# Converts to something like:
# https://uploads.github.com/repos/Azure/azure-sdk-for-cpp/releases/123/assets?name=foo.tar.gz
# Docs: https://docs.github.com/en/rest/reference/repos#get-a-release-by-tag-name
$uploadUrl = $release.upload_url.Split('{')[0] + "?name=$assetFilename"

Write-Verbose "Uploading $assetFilename to $uploadUrl"

$asset = Invoke-RestMethod `
    -Uri $uploadUrl `
    -Method POST `
    -InFile $AssetPath `
    -Credential $credentials `
    -Headers @{ Authorization = "token $GitHubPat" } `
    -ContentType "application/gzip"

Write-Verbose "Upload complete. Browser download URL: $($asset.browser_download_url)"

return $asset
