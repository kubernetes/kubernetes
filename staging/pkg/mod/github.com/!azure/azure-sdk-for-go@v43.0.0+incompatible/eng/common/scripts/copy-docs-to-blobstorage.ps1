# Note, due to how `Expand-Archive` is leveraged in this script, 
# powershell core is a requirement for successful execution. 
param (
  $AzCopy,
  $DocLocation,
  $SASKey,
  $Language,
  $BlobName,
  $ExitOnError=1
)
$Language = $Language.ToLower()

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

    $versionsObject.SortedVersionArray = SortSemVersions -versions ($VersionArray | % { ToSemVer $_})
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
    Write-Host "Heading to $versionUri to retrieve known versions"

    try {
        return ((Invoke-RestMethod -Uri $versionUri -MaximumRetryCount 3 -RetryIntervalSec 5) -Split "\n" | % {$_.Trim()} | ? { return $_ })
    }
    catch {
        # Handle 404. If it's 404, this is the first time we've published this package.
        if ($_.Exception.Response.StatusCode.value__ -eq 404){
            Write-Host "Version file does not exist. This is the first time we have published this package."
        }
        else {
            # If it's not a 404. exit. We don't know what's gone wrong.
            Write-Host "Exception getting version file. Aborting"
            Write-Host $_
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

    Write-Host "Before I update anything, I am seeing $existingVersions"

    if (!$existingVersions)
    {
        $existingVersions = @()
        $existingVersions += $PkgVersion
        Write-Host "No existing versions. Adding $PkgVersion."
    }
    else 
    {
        $existingVersions += $pkgVersion
        Write-Host "Already Existing Versions. Adding $PkgVersion."
    }

    $existingVersions = $existingVersions | Select-Object -Unique

    # newest first
    $sortedVersionObj = (Sort-Versions -VersionArray $existingVersions)

    Write-Host $sortedVersionObj
    Write-Host $sortedVersionObj.LatestGAPackage
    Write-Host $sortedVersionObj.LatestPreviewPackage

    # write to file. to get the correct performance with "actually empty" files, we gotta do the newline 
    # join ourselves. This way we have absolute control over the trailing whitespace.
    $sortedVersionObj.RawVersionsList -join "`n" | Out-File -File "$($DocLocation)/versions" -Force -NoNewLine
    $sortedVersionObj.LatestGAPackage | Out-File -File "$($DocLocation)/latest-ga" -Force -NoNewLine
    $sortedVersionObj.LatestPreviewPackage | Out-File -File "$($DocLocation)/latest-preview" -Force -NoNewLine

    & $($AzCopy) cp "$($DocLocation)/versions" "$($DocDest)/$($PkgName)/versioning/versions$($SASKey)"
    & $($AzCopy) cp "$($DocLocation)/latest-preview" "$($DocDest)/$($PkgName)/versioning/latest-preview$($SASKey)"
    & $($AzCopy) cp "$($DocLocation)/latest-ga" "$($DocDest)/$($PkgName)/versioning/latest-ga$($SASKey)"
}

function Upload-Blobs
{
    Param (
        [Parameter(Mandatory=$true)] [String]$DocDir,
        [Parameter(Mandatory=$true)] [String]$PkgName,
        [Parameter(Mandatory=$true)] [String]$DocVersion
    )
    #eg : $BlobName = "https://azuresdkdocs.blob.core.windows.net"
    $DocDest = "$($BlobName)/`$web/$($Language)"

    Write-Host "DocDest $($DocDest)"
    Write-Host "PkgName $($PkgName)"
    Write-Host "DocVersion $($DocVersion)"
    Write-Host "DocDir $($DocDir)"
    Write-Host "Final Dest $($DocDest)/$($PkgName)/$($DocVersion)"

    Write-Host "Uploading $($PkgName)/$($DocVersion) to $($DocDest)..."
    & $($AzCopy) cp "$($DocDir)/**" "$($DocDest)/$($PkgName)/$($DocVersion)$($SASKey)" --recursive=true

    Write-Host "Handling versioning files under $($DocDest)/$($PkgName)/versioning/"
    Update-Existing-Versions -PkgName $PkgName -PkgVersion $DocVersion -DocDest $DocDest
}


if ($Language -eq "javascript")
{
    $PublishedDocs = Get-ChildItem "$($DocLocation)/documentation" | Where-Object -FilterScript {$_.Name.EndsWith(".zip")}

    foreach ($Item in $PublishedDocs) {
        $PkgName = "azure-$($Item.BaseName)"
        Write-Host $PkgName
        Expand-Archive -Force -Path "$($DocLocation)/documentation/$($Item.Name)" -DestinationPath "$($DocLocation)/documentation/$($Item.BaseName)"
        $dirList = Get-ChildItem "$($DocLocation)/documentation/$($Item.BaseName)/$($Item.BaseName)" -Attributes Directory
    
        if($dirList.Length -eq 1){
            $DocVersion = $dirList[0].Name
            Write-Host "Uploading Doc for $($PkgName) Version:- $($DocVersion)..."
            Upload-Blobs -DocDir "$($DocLocation)/documentation/$($Item.BaseName)/$($Item.BaseName)/$($DocVersion)" -PkgName $PkgName -DocVersion $DocVersion
        }
        else{
            Write-Host "found more than 1 folder under the documentation for package - $($Item.Name)"
        }
    }
}

if ($Language -eq "dotnet")
{
    $PublishedPkgs = Get-ChildItem "$($DocLocation)/packages" | Where-Object -FilterScript {$_.Name.EndsWith(".nupkg") -and -not $_.Name.EndsWith(".symbols.nupkg")}
    $PublishedDocs = Get-ChildItem "$($DocLocation)" | Where-Object -FilterScript {$_.Name.StartsWith("Docs.")}

    foreach ($Item in $PublishedDocs) {
        $PkgName = $Item.Name.Remove(0, 5)
        $PkgFullName = $PublishedPkgs | Where-Object -FilterScript {$_.Name -match "$($PkgName).\d"}

        if (($PkgFullName | Measure-Object).count -eq 1) 
        {
            $DocVersion = $PkgFullName[0].BaseName.Remove(0, $PkgName.Length + 1)

            Write-Host "Start Upload for $($PkgName)/$($DocVersion)"
            Write-Host "DocDir $($Item)"
            Write-Host "PkgName $($PkgName)"
            Write-Host "DocVersion $($DocVersion)"
            Upload-Blobs -DocDir "$($Item)" -PkgName $PkgName -DocVersion $DocVersion
        }
        else
        {
            Write-Host "Package with the same name Exists. Upload Skipped"
            continue
        }
    }
}

if ($Language -eq "python")
{
    $PublishedDocs = Get-ChildItem "$DocLocation" | Where-Object -FilterScript {$_.Name.EndsWith(".zip")}
    
    foreach ($Item in $PublishedDocs) {
        $PkgName = $Item.BaseName
        $ZippedDocumentationPath = Join-Path -Path $DocLocation -ChildPath $Item.Name
        $UnzippedDocumentationPath = Join-Path -Path $DocLocation -ChildPath $PkgName
        $VersionFileLocation = Join-Path -Path $UnzippedDocumentationPath -ChildPath "version.txt"        

        Expand-Archive -Force -Path $ZippedDocumentationPath -DestinationPath $UnzippedDocumentationPath

        $Version = $(Get-Content $VersionFileLocation).Trim()

        Write-Host "Discovered Package Name: $PkgName"
        Write-Host "Discovered Package Version: $Version"
        Write-Host "Directory for Upload: $UnzippedDocumentationPath"

        Upload-Blobs -DocDir $UnzippedDocumentationPath -PkgName $PkgName -DocVersion $Version
    }
}

if ($Language -eq "java")
{
    $PublishedDocs = Get-ChildItem "$DocLocation" | Where-Object -FilterScript {$_.Name.EndsWith("-javadoc.jar")}
    foreach ($Item in $PublishedDocs) {
        $UnjarredDocumentationPath = ""
        try {
            $PkgName = $Item.BaseName
            # The jar's unpacking command doesn't allow specifying a target directory
            # and will unjar all of the files in whatever the current directory is.
            # Create a subdirectory to unjar into, set the location, unjar and then
            # set the location back to its original location.
            $UnjarredDocumentationPath = Join-Path -Path $DocLocation -ChildPath $PkgName
            New-Item -ItemType directory -Path "$UnjarredDocumentationPath"
            $CurrentLocation = Get-Location
            Set-Location $UnjarredDocumentationPath
            jar -xf "$($Item.FullName)"
            Set-Location $CurrentLocation

            # Get the POM file for the artifact we're processing
            $PomFile = $Item.FullName.Substring(0,$Item.FullName.LastIndexOf(("-javadoc.jar"))) + ".pom"
            Write-Host "PomFile $($PomFile)"

            # Pull the version from the POM
            [xml]$PomXml = Get-Content $PomFile
            $Version = $PomXml.project.version
            $ArtifactId = $PomXml.project.artifactId

            Write-Host "Start Upload for $($PkgName)/$($Version)"
            Write-Host "DocDir $($UnjarredDocumentationPath)"
            Write-Host "PkgName $($ArtifactId)"
            Write-Host "DocVersion $($Version)"

            Upload-Blobs -DocDir $UnjarredDocumentationPath -PkgName $ArtifactId -DocVersion $Version

        } Finally {
            if (![string]::IsNullOrEmpty($UnjarredDocumentationPath)) {
                if (Test-Path -Path $UnjarredDocumentationPath) {
                    Write-Host "Cleaning up $UnjarredDocumentationPath"
                    Remove-Item -Recurse -Force $UnjarredDocumentationPath
                }
            }
        }
    }
}

if ($Language -eq "c")
{
    # The documentation publishing process for C differs from the other
    # langauges in this file because this script is invoked once per library
    # publishing. It is not, for example, invoked once per service publishing.
    # This is also the case for other langauge publishing steps above... Those
    # loops are left over from previous versions of this script which were used
    # to publish multiple docs packages in a single invocation.
    $pkgInfo = Get-Content $DocLocation/package-info.json | ConvertFrom-Json
    $pkgName = $pkgInfo.name
    $pkgVersion = $pkgInfo.version

    Upload-Blobs -DocDir $DocLocation -PkgName $pkgName -DocVersion $pkgVersion
}