[CmdletBinding()]
Param (
  [Parameter(Mandatory=$True)]
  [string] $ArtifactPath,
  [Parameter(Mandatory=$True)]
  [string] $APIViewUri,
  [Parameter(Mandatory=$True)]
  [string] $APIKey,
  [Parameter(Mandatory=$True)]
  [string] $APILabel,
  [string] $PackageName,
  [string] $SourceBranch,
  [string] $DefaultBranch,
  [string] $ConfigFileDir = ""
)

# Submit API review request and return status whether current revision is approved or pending or failed to create review
function Submit-APIReview($packagename, $filePath, $uri, $apiKey, $apiLabel)
{
    $multipartContent = [System.Net.Http.MultipartFormDataContent]::new()
    $FileStream = [System.IO.FileStream]::new($filePath, [System.IO.FileMode]::Open)
    $fileHeader = [System.Net.Http.Headers.ContentDispositionHeaderValue]::new("form-data")
    $fileHeader.Name = "file"
    $fileHeader.FileName = $packagename
    $fileContent = [System.Net.Http.StreamContent]::new($FileStream)
    $fileContent.Headers.ContentDisposition = $fileHeader
    $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("application/octet-stream")
    $multipartContent.Add($fileContent)


    $stringHeader = [System.Net.Http.Headers.ContentDispositionHeaderValue]::new("form-data")
    $stringHeader.Name = "label"
    $StringContent = [System.Net.Http.StringContent]::new($apiLabel)
    $StringContent.Headers.ContentDisposition = $stringHeader
    $multipartContent.Add($stringContent)

    $headers = @{
        "ApiKey" = $apiKey;
        "content-type" = "multipart/form-data"
    }

    try
    {
        $Response = Invoke-WebRequest -Method 'POST' -Uri $uri -Body $multipartContent -Headers $headers
        Write-Host "API Review URL: $($Response.Content)"
        $StatusCode = $Response.StatusCode
    }
    catch
    {
        Write-Host "Exception details: $($_.Exception.Response)"
        $StatusCode = $_.Exception.Response.StatusCode
    }

    return $StatusCode
}


. (Join-Path $PSScriptRoot common.ps1)

Write-Host "Artifact path: $($ArtifactPath)"
Write-Host "Package Name: $($PackageName)"
Write-Host "Source branch: $($SourceBranch)"
Write-Host "Config File directory: $($ConfigFileDir)"

$packages = @{}
if ($FindArtifactForApiReviewFn -and (Test-Path "Function:$FindArtifactForApiReviewFn"))
{
    $packages = &$FindArtifactForApiReviewFn $ArtifactPath $PackageName
}
else
{
    Write-Host "The function for 'FindArtifactForApiReviewFn' was not found.`
    Make sure it is present in eng/scripts/Language-Settings.ps1 and referenced in eng/common/scripts/common.ps1.`
    See https://github.com/Azure/azure-sdk-tools/blob/master/doc/common/common_engsys.md#code-structure"
    exit(1)
}

# Check if package config file is present. This file has package version, SDK type etc info.
if (-not $ConfigFileDir)
{
    $ConfigFileDir = Join-Path -Path $ArtifactPath "PackageInfo"
}

if ($packages)
{
    foreach($pkgPath in $packages.Values)
    {
        $pkg = Split-Path -Leaf $pkgPath
        $pkgPropPath = Join-Path -Path $ConfigFileDir "$PackageName.json"
        if (-Not (Test-Path $pkgPropPath))
        {
            Write-Host " Package property file path $($pkgPropPath) is invalid."
            continue
        }
        # Get package info from json file created before updating version to daily dev
        $pkgInfo = Get-Content $pkgPropPath | ConvertFrom-Json
        $version = [AzureEngSemanticVersion]::ParseVersionString($pkgInfo.Version)
        if ($version -eq $null)
        {
            Write-Host "Version info is not available for package $PackageName, because version '$(pkgInfo.Version)' is invalid. Please check if the version follows Azure SDK package versioning guidelines."
            exit 1
        }

        Write-Host "Version: $($version)"
        Write-Host "SDK Type: $($pkgInfo.SdkType)"

        # Run create review step only if build is triggered from master branch or if version is GA.
        # This is to avoid invalidating review status by a build triggered from feature branch
        if ( ($SourceBranch -eq $DefaultBranch) -or (-not $version.IsPrerelease))
        {
            Write-Host "Submitting API Review for package $($pkg)"
            $respCode = Submit-APIReview -packagename $pkg -filePath $pkgPath -uri $APIViewUri -apiKey $APIKey -apiLabel $APILabel
            Write-Host "HTTP Response code: $($respCode)"
            # HTTP status 200 means API is in approved status
            if ($respCode -eq '200')
            {
                Write-Host "API review is in approved status."
            }
            elseif ($version.IsPrerelease -or ($version.Major -eq 0))
            {
                # Ignore API review status for prerelease version
                Write-Host "Package version is not GA. Ignoring API view approval status"
            }
            else
            {
                # Return error code if status code is 201 for new data plane package
                if ($pkgInfo.SdkType -eq "client" -and $pkgInfo.IsNewSdk)
                {
                    if ($respCode -eq '201')
                    {
                        Write-Host "Package version $($version) is GA and automatic API Review is not yet approved for package $($PackageName)."
                        Write-Host "Build and release is not allowed for GA package without API review approval."
                        Write-Host "You will need to queue another build to proceed further after API review is approved"
                        Write-Host "You can check http://aka.ms/azsdk/engsys/apireview/faq for more details on API Approval."
                    }
                    else
                    {
                        Write-Host "Failed to create API Review for package $($PackageName). Please reach out to Azure SDK engineering systems on teams channel and share this build details."
                    }
                    exit 1
                }
                else
                {
                    Write-Host "API review is not approved for package $($PackageName), however it is not required for this package type so it can still be released without API review approval."
                }
            } 
        }
        else
        {
            Write-Host "Build is triggered from $($SourceBranch) with prerelease version. Skipping API review status check."
        }
    }
}
else
{
    Write-Host "No package is found in artifact path to submit review request"
}
