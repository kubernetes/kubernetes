<#
  .SYNOPSIS
  Check broken links.

  .DESCRIPTION
  The Verify-Links.ps1 script will check whether the files contain any broken links.

  .PARAMETER urls
  Specify url list to verify links. Can either be a http address or a local file request. Local file paths support md and html files.

  .PARAMETER ignoreLinksFile
  Specifies the file that contains a set of links to ignore when verifying.

  .PARAMETER devOpsLogging
  Switch that will enable devops specific logging for warnings

  .PARAMETER recursive
  Check the links recurisvely based on recursivePattern.

  .PARAMETER baseUrl
  Recursively check links for all links verified that begin with this baseUrl, defaults to the folder the url is contained in.

  .PARAMETER rootUrl
  Path to the root of the site for resolving rooted relative links, defaults to host root for http and file directory for local files.

  .PARAMETER errorStatusCodes
  List of http status codes that count as broken links. Defaults to 400, 401, 404, SocketError.HostNotFound = 11001, SocketError.NoData = 11004.

  .PARAMETER branchReplaceRegex
  Regex to check if the link needs to be replaced. E.g. ^(https://github.com/.*/(?:blob|tree)/)master(/.*)$

  .PARAMETER branchReplacementName
  The substitute branch name or SHA commit.

  .PARAMETER checkLinkGuidance
  Flag to allow checking against azure sdk link guidance. Check link guidance here: https://aka.ms/azsdk/guideline/links.

  .PARAMETER userAgent
  UserAgent to be configured for web requests. Defaults to current Chrome version.

  .PARAMETER inputCacheFile
  Path to a file that contains a list of links that are known valid so we can skip checking them.

  .PARAMETER outputCacheFile
  Path to a file that the script will output all the validated links after running all checks.

  .EXAMPLE
  PS> .\Verify-Links.ps1 C:\README.md

  .EXAMPLE
  PS> .\Verify-Links.ps1 https://azure.github.io/azure-sdk/index.html

  .EXAMPLE
  PS> .\Verify-Links C:\README.md -checkLinkGuidance $true
#>
[CmdletBinding()]
param (
  [string[]] $urls,
  [string] $ignoreLinksFile = "$PSScriptRoot/ignore-links.txt",
  [switch] $devOpsLogging = $false,
  [switch] $recursive = $true,
  [string] $baseUrl = "",
  [string] $rootUrl = "",
  [array] $errorStatusCodes = @(400, 401, 404, 11001, 11004),
  [string] $branchReplaceRegex = "",
  [string] $branchReplacementName = "",
  [bool] $checkLinkGuidance = $false,
  [string] $userAgent,
  [string] $inputCacheFile,
  [string] $outputCacheFile
)

$ProgressPreference = "SilentlyContinue"; # Disable invoke-webrequest progress dialog
# Regex of the locale keywords.
$locale = "/en-us/"
$emptyLinkMessage = "There is at least one empty link in the page. Please replace with absolute link. Check here for more information: https://aka.ms/azsdk/guideline/links"
if (!$userAgent) {
  $userAgent = "Chrome/87.0.4280.88"
}
function NormalizeUrl([string]$url){
  if (Test-Path $url) {
    $url = "file://" + (Resolve-Path $url).ToString();
  }

  Write-Verbose "The url to check against: $url."
  $uri = [System.Uri]$url;

  if ($script:baseUrl -eq "") {
    # for base url default to containing directory
    $script:baseUrl = (new-object System.Uri($uri, ".")).ToString();
  }

  if ($script:rootUrl -eq "") {
    if ($uri.IsFile) {
      # for files default to the containing directory
      $script:rootUrl = $script:baseUrl;
    }
    else {
      # for http links default to the root path
      $script:rootUrl = new-object System.Uri($uri, "/");
    }
  }
  return $uri
}

function LogWarning
{
  if ($devOpsLogging)
  {
    Write-Host "##vso[task.LogIssue type=warning;]$args"
  }
  else
  {
    Write-Warning "$args"
  }
}

function LogError
{
  if ($devOpsLogging)
  {
    Write-Host "##vso[task.logissue type=error]$args"
  }
  else
  {
    Write-Error "$args"
  }
}

function ResolveUri ([System.Uri]$referralUri, [string]$link)
{
  # If the link is mailto, skip it.
  if ($link.StartsWith("mailto:")) {
    Write-Verbose "Skipping $link because it is a mailto link."
    return
  }

  $linkUri = [System.Uri]$link;
  # Our link guidelines do not allow relative links so only resolve them when we are not
  # validating links against our link guidelines (i.e. !$checkLinkGuideance)
  if ($checkLinkGuidance -and !$linkUri.IsAbsoluteUri) {
    return $linkUri
  }

  if (!$linkUri.IsAbsoluteUri) {
    # For rooted paths resolve from the baseUrl
    if ($link.StartsWith("/")) {
      Write-Verbose "rooturl = $rootUrl"
      $linkUri = new-object System.Uri([System.Uri]$rootUrl, ".$link");
    }
    else {
      $linkUri = new-object System.Uri($referralUri, $link);
    }
  }

  $linkUri = [System.Uri]$linkUri.GetComponents([System.UriComponents]::HttpRequestUrl, [System.UriFormat]::SafeUnescaped)
  Write-Verbose "ResolvedUri $link to $linkUri"

  # If the link is not a web request, like mailto, skip it.
  if (!$linkUri.Scheme.StartsWith("http") -and !$linkUri.IsFile) {
    Write-Verbose "Skipping $linkUri because it is not http or file based."
    return
  }

  if ($null -ne $ignoreLinks -and ($ignoreLinks.Contains($link) -or $ignoreLinks.Contains($linkUri.ToString()))) {
    Write-Verbose "Ignoring invalid link $linkUri because it is in the ignore file."
    return
  }

  return $linkUri;
}

function ParseLinks([string]$baseUri, [string]$htmlContent)
{
  $hrefRegex = "<a[^>]+href\s*=\s*[""']?(?<href>[^""']*)[""']?"
  $regexOptions = [System.Text.RegularExpressions.RegexOptions]"Singleline, IgnoreCase";

  $hrefs = [RegEx]::Matches($htmlContent, $hrefRegex, $regexOptions);

  #$hrefs | Foreach-Object { Write-Host $_ }

  Write-Verbose "Found $($hrefs.Count) raw href's in page $baseUri";
  $links = $hrefs | ForEach-Object { ResolveUri $baseUri $_.Groups["href"].Value }

  #$links | Foreach-Object { Write-Host $_ }

  return $links
}

function CheckLink ([System.Uri]$linkUri, $allowRetry=$true)
{
  if(!$linkUri.ToString().Trim()) {
    LogWarning "Found Empty link. Please use absolute link instead. Check here for more information: https://aka.ms/azsdk/guideline/links"
    return $false
  }

  $originalLinkUri = $linkUri
  $linkUri = ReplaceGithubLink $linkUri

  $link = $linkUri.ToString()

  if ($checkedLinks.ContainsKey($link)) {
    if (!$checkedLinks[$link]) {
      LogWarning "broken link $link"
    }
    return $checkedLinks[$link]
  }

  $linkValid = $true
  Write-Verbose "Checking link $linkUri..."

  if ($linkUri.IsFile) {
    if (!(Test-Path $linkUri.LocalPath)) {
      LogWarning "Link to file does not exist $($linkUri.LocalPath)"
      $linkValid = $false
    }
  }
  elseif ($linkUri.IsAbsoluteUri) {
    try {
      $headRequestSucceeded = $true
      try {
        # Attempt HEAD request first
        $response = Invoke-WebRequest -Uri $linkUri -Method HEAD -UserAgent $userAgent
      }
      catch {
        $headRequestSucceeded = $false
      }
      if (!$headRequestSucceeded) {
        # Attempt a GET request if the HEAD request failed.
        $response = Invoke-WebRequest -Uri $linkUri -Method GET -UserAgent $userAgent
      }
      $statusCode = $response.StatusCode
      if ($statusCode -ne 200) {
        Write-Host "[$statusCode] while requesting $linkUri"
      }
    }
    catch {
      $statusCode = $_.Exception.Response.StatusCode.value__

      if(!$statusCode) {
        # Try to pull the error code from any inner SocketException we might hit
        $statusCode = $_.Exception.InnerException.ErrorCode
      }

      if ($statusCode -in $errorStatusCodes) {
        if ($originalLinkUri -ne $linkUri) {
          LogWarning "[$statusCode] broken link $originalLinkUri (resolved to $linkUri)"
        }
        else {
          LogWarning "[$statusCode] broken link $linkUri"
        }

        $linkValid = $false
      }
      else {
        if ($null -ne $statusCode) {
          # For 429 rate-limiting try to pause if possible
          if ($allowRetry -and $_.Exception.Response -and $statusCode -eq 429) {
            $retryAfter = $_.Exception.Response.Headers.RetryAfter.Delta.TotalSeconds

            # Default retry after 60 (arbitrary) seconds if no header given
            if (!$retryAfter -or $retryAfter -gt 60) { $retryAfter = 60 }
            Write-Host "Rate-Limited for $retryAfter seconds while requesting $linkUri"

            Start-Sleep -Seconds $retryAfter
            $linkValid = CheckLink $originalLinkUri -allowRetry $false
          }
          else {
            Write-Host "[$statusCode] handled while requesting $linkUri"
            # Override and set status code in the cache so it is truthy
            # so we don't keep checking but we don't think it is valid either
            $linkValid = $statusCode
          }
        }
        else {
          Write-Host "Exception while requesting $linkUri"
          Write-Host $_.Exception.ToString()
          # Override and set exception in the cache so it is truthy
          # so we don't keep checking but we don't think it is valid either
          $linkValid = "Exception"
        }
      }
    }
  }
  elseif ($link.StartsWith("#")) {
    # Ignore anchor links as we don't have a great way to check them.
  }
  else {
    LogWarning "Link has invalid format $linkUri"
    $linkValid = $false
  }

  if ($checkLinkGuidance) {
    if ($linkUri.Scheme -eq 'http') {
      LogWarning "DO NOT use 'http' in $linkUri. Please use secure link with https instead. Check here for more information: https://aka.ms/azsdk/guideline/links"
      $linkValid = $false
    }
    # Check if the url is relative links, suppress the archor link validation.
    if (!$linkUri.IsAbsoluteUri -and !$link.StartsWith("#")) {
      LogWarning "DO NOT use relative link $linkUri. Please use absolute link instead. Check here for more information: https://aka.ms/azsdk/guideline/links"
      $linkValid = $false
    }
    # Check if the url is anchor link has any uppercase.
    if ($link -cmatch '#[^?]*[A-Z]') {
      LogWarning "Please lower case your anchor tags (i.e. anything after '#' in your link '$linkUri'. Check here for more information: https://aka.ms/azsdk/guideline/links"
      $linkValid = $false
    }
     # Check if link uri includes locale info.
    if ($linkUri -match $locale) {
      LogWarning "DO NOT include locale $locale information in links: $linkUri. Check here for more information: https://aka.ms/azsdk/guideline/links"
      $linkValid = $false
    }
  }

  $checkedLinks[$link] = $linkValid
  return $linkValid
}

function ReplaceGithubLink([string]$originLink) {
  if (!$branchReplacementName -or !$branchReplaceRegex) {
    return $originLink
  }
  $ReplacementPattern = "`${1}$branchReplacementName`$2"
  return $originLink -replace $branchReplaceRegex, $ReplacementPattern
}

function GetLinks([System.Uri]$pageUri)
{
  if ($pageUri.Scheme.StartsWith("http")) {
    try {
      $response = Invoke-WebRequest -Uri $pageUri -UserAgent $userAgent
      $content = $response.Content

      if ($pageUri.ToString().EndsWith(".md")) {
        $content = (ConvertFrom-MarkDown -InputObject $content).html
      }
    }
    catch {
      $statusCode = $_.Exception.Response.StatusCode.value__
      Write-Error "Invalid page [$statusCode] $pageUri"
    }
  }
  elseif ($pageUri.IsFile -and (Test-Path $pageUri.LocalPath)) {
    $file = $pageUri.LocalPath
    if ($file.EndsWith(".md")) {
      $content = (ConvertFrom-MarkDown $file).html
    }
    elseif ($file.EndsWith(".html")) {
      $content = Get-Content $file
    }
    else {
      if (Test-Path ($file + "index.html")) {
        $content = Get-Content ($file + "index.html")
      }
      else {
        # Fallback to just reading the content directly
        $content = Get-Content $file
      }
    }
  }
  else {
    Write-Error "Don't know how to process uri $pageUri"
  }

  $links = ParseLinks $pageUri $content

  return $links;
}

if ($urls) {
  if ($urls.Count -eq 0) {
    Write-Host "Usage $($MyInvocation.MyCommand.Name) <urls>";
    exit 1;
  }
}

if ($PSVersionTable.PSVersion.Major -lt 6)
{
  LogWarning "Some web requests will not work in versions of PS earlier then 6. You are running version $($PSVersionTable.PSVersion)."
}
$ignoreLinks = @();
if (Test-Path $ignoreLinksFile) {
  $ignoreLinks = (Get-Content $ignoreLinksFile).Where({ $_.Trim() -ne "" -and !$_.StartsWith("#") })
}

# Use default hashtable constructor instead of @{} because we need them to be case sensitive
$checkedPages = New-Object Hashtable
$checkedLinks = New-Object Hashtable

if ($inputCacheFile)
{
  $cacheContent = ""
  if ($inputCacheFile.StartsWith("http")) {
    try {
      $response = Invoke-WebRequest -Uri $inputCacheFile
      $cacheContent = $response.Content
    }
    catch {
      $statusCode = $_.Exception.Response.StatusCode.value__
      Write-Error "Failed to read cache file from  page [$statusCode] $inputCacheFile"
    }
  }
  elseif (Test-Path $inputCacheFile) {
    $cacheContent = Get-Content $inputCacheFile -Raw
  }
  $goodLinks = $cacheContent.Split("`n").Where({ $_.Trim() -ne "" -and !$_.StartsWith("#") })

  foreach ($goodLink in $goodLinks) {
    $checkedLinks[$goodLink] = $true
  }
}

$cachedLinksCount = $checkedLinks.Count

if ($cachedLinksCount) {
  Write-Host "Skipping checks on $cachedLinksCount links found in the given cache of known good links."
}

$badLinks = New-Object Hashtable
$pageUrisToCheck = new-object System.Collections.Queue
foreach ($url in $urls) {
  $uri = NormalizeUrl $url
  $pageUrisToCheck.Enqueue($uri);
}

while ($pageUrisToCheck.Count -ne 0)
{
  $pageUri = $pageUrisToCheck.Dequeue();
  if ($checkedPages.ContainsKey($pageUri)) { continue }
  $checkedPages[$pageUri] = $true;

  $linkUris = GetLinks $pageUri
  Write-Host "Found $($linkUris.Count) links on page $pageUri";
  $badLinksPerPage = @();
  foreach ($linkUri in $linkUris) {
    $isLinkValid = CheckLink $linkUri
    if (!$isLinkValid -and !$badLinksPerPage.Contains($linkUri)) {
      if (!$linkUri.ToString().Trim()) {
        $linkUri = $emptyLinkMessage
      }
      $badLinksPerPage += $linkUri
    }
    if ($recursive -and $isLinkValid) {
      if ($linkUri.ToString().StartsWith($baseUrl) -and !$checkedPages.ContainsKey($linkUri)) {
        $pageUrisToCheck.Enqueue($linkUri);
      }
    }
  }
  if ($badLinksPerPage.Count -gt 0) {
    $badLinks[$pageUri] = $badLinksPerPage
  }
}

if ($badLinks.Count -gt 0) {
  Write-Host "Summary of broken links:"
}
foreach ($pageLink in $badLinks.Keys) {
  Write-Host "'$pageLink' has $($badLinks[$pageLink].Count) broken link(s):"
  foreach ($brokenLink in $badLinks[$pageLink]) {
    Write-Host "  $brokenLink"
  }
}

$linksChecked = $checkedLinks.Count - $cachedLinksCount

if ($badLinks.Count -gt 0) {
  LogError "Checked $linksChecked links with $($badLinks.Count) page(s) broken."
}
else {
  Write-Host "Checked $linksChecked links. No broken links found."
}

if ($outputCacheFile)
{
  $goodLinks = $checkedLinks.Keys.Where({ "True" -eq $checkedLinks[$_].ToString() }) | Sort-Object

  Write-Host "Writing the list of validated links to $outputCacheFile"
  $goodLinks | Set-Content $outputCacheFile
}

exit $badLinks.Count
