<#
.DESCRIPTION
Parses a semver version string into its components and supports operations around it that we use for versioning our packages.

See https://azure.github.io/azure-sdk/policies_releases.html#package-versioning

Example: 1.2.3-beta.4
Components: Major.Minor.Patch-PrereleaseLabel.PrereleaseNumber

Example: 1.2.3-alpha.20200828.4
Components: Major.Minor.Patch-PrereleaseLabel.PrereleaseNumber.BuildNumber

Note: A builtin Powershell version of SemVer exists in 'System.Management.Automation'. At this time, it does not parsing of PrereleaseNumber. It's name is also type accelerated to 'SemVer'.
#>

class AzureEngSemanticVersion : IComparable {
  [int] $Major
  [int] $Minor
  [int] $Patch
  [string] $PrereleaseLabelSeparator
  [string] $PrereleaseLabel
  [string] $PrereleaseNumberSeparator
  [string] $BuildNumberSeparator
  # BuildNumber is string to preserve zero-padding where applicable
  [string] $BuildNumber
  [int] $PrereleaseNumber
  [bool] $IsPrerelease
  [string] $VersionType
  [string] $RawVersion
  [bool] $IsSemVerFormat
  [string] $DefaultPrereleaseLabel
  [string] $DefaultAlphaReleaseLabel

  # Regex inspired but simplified from https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
  # Validation: https://regex101.com/r/vkijKf/426
  static [string] $SEMVER_REGEX = "(?<major>0|[1-9]\d*)\.(?<minor>0|[1-9]\d*)\.(?<patch>0|[1-9]\d*)(?:(?<presep>-?)(?<prelabel>[a-zA-Z]+)(?:(?<prenumsep>\.?)(?<prenumber>[0-9]{1,8})(?:(?<buildnumsep>\.?)(?<buildnumber>\d{1,3}))?)?)?"

  static [AzureEngSemanticVersion] ParseVersionString([string] $versionString)
  {
    $version = [AzureEngSemanticVersion]::new($versionString)

    if (!$version.IsSemVerFormat) {
      return $null
    }
    return $version
  }

  static [AzureEngSemanticVersion] ParsePythonVersionString([string] $versionString)
  {
    $version = [AzureEngSemanticVersion]::ParseVersionString($versionString)

    if (!$version) {
      return $null
    }

    $version.SetupPythonConventions()
    return $version
  }
  
  AzureEngSemanticVersion([string] $versionString)
  {
    if ($versionString -match "^$([AzureEngSemanticVersion]::SEMVER_REGEX)$")
    {
      $this.IsSemVerFormat = $true
      $this.RawVersion = $versionString
      $this.Major = [int]$matches.Major
      $this.Minor = [int]$matches.Minor
      $this.Patch = [int]$matches.Patch

      # If Language exists and is set to python setup the python conventions.
      $parseLanguage = (Get-Variable -Name "Language" -ValueOnly -ErrorAction "Ignore")
      if ($parseLanguage -eq "python") {
        $this.SetupPythonConventions()
      }
      else {
        $this.SetupDefaultConventions()
      }

      if ($null -eq $matches['prelabel']) 
      {
        # artifically provide these values for non-prereleases to enable easy sorting of them later than prereleases.
        $this.PrereleaseLabel = "zzz"
        $this.PrereleaseNumber = 99999999
        $this.IsPrerelease = $false
        $this.VersionType = "GA"
        if ($this.Patch -ne 0) {
          $this.VersionType = "Patch"
        }
      }
      else
      {
        $this.PrereleaseLabel = $matches["prelabel"]
        $this.PrereleaseLabelSeparator = $matches["presep"]
        $this.PrereleaseNumber = [int]$matches["prenumber"]
        $this.PrereleaseNumberSeparator = $matches["prenumsep"]
        $this.IsPrerelease = $true
        $this.VersionType = "Beta"

        $this.BuildNumberSeparator = $matches["buildnumsep"]
        $this.BuildNumber = $matches["buildnumber"]
      }
    }
    else
    {
      $this.RawVersion = $versionString
      $this.IsSemVerFormat = $false
    }
  }

  # If a prerelease label exists, it must be 'beta', and similar semantics used in our release guidelines
  # See https://azure.github.io/azure-sdk/policies_releases.html#package-versioning
  [bool] HasValidPrereleaseLabel()
  {
    if ($this.IsPrerelease -eq $true) {
      if ($this.PrereleaseLabel -ne $this.DefaultPrereleaseLabel -and $this.PrereleaseLabel -ne $this.DefaultAlphaReleaseLabel) {
        Write-Host "Unexpected pre-release identifier '$($this.PrereleaseLabel)', "`
                   "should be '$($this.DefaultPrereleaseLabel)' or '$($this.DefaultAlphaReleaseLabel)'"
        return $false;
      }
      if ($this.PrereleaseNumber -lt 1)
      {
        Write-Host "Unexpected pre-release version '$($this.PrereleaseNumber)', should be >= '1'"
        return $false;
      }
    }

    return $true;
  }

  [string] ToString()
  {
    $versionString = "{0}.{1}.{2}" -F $this.Major, $this.Minor, $this.Patch

    if ($this.IsPrerelease)
    {
      $versionString += $this.PrereleaseLabelSeparator + $this.PrereleaseLabel + `
                        $this.PrereleaseNumberSeparator + $this.PrereleaseNumber
      if ($this.BuildNumber) {
          $versionString += $this.BuildNumberSeparator + $this.BuildNumber
      }
    }
    return $versionString;
  }

  [void] IncrementAndSetToPrerelease() {
    if ($this.IsPrerelease -eq $false)
    {
      $this.PrereleaseLabel = $this.DefaultPrereleaseLabel
      $this.PrereleaseNumber = 1
      $this.Minor++
      $this.Patch = 0
      $this.IsPrerelease = $true
    }
    else
    {
      if ($this.BuildNumber) {
        throw "Cannot increment releases tagged with azure pipelines build numbers"
      }
      $this.PrereleaseNumber++
    }
  }

  [void] SetupPythonConventions() 
  {
    # Python uses no separators and "b" for beta so this sets up the the object to work with those conventions
    $this.PrereleaseLabelSeparator = $this.PrereleaseNumberSeparator = $this.BuildNumberSeparator = ""
    $this.DefaultPrereleaseLabel = "b"
    $this.DefaultAlphaReleaseLabel = "a"
  }

  [void] SetupDefaultConventions() 
  {
    # Use the default common conventions
    $this.PrereleaseLabelSeparator = "-"
    $this.PrereleaseNumberSeparator = "."
    $this.BuildNumberSeparator = "."
    $this.DefaultPrereleaseLabel = "beta"
    $this.DefaultAlphaReleaseLabel = "alpha"
  }

  [int] CompareTo($other)
  {
    if ($other -isnot [AzureEngSemanticVersion]) {
      throw "Cannot compare $other with $this"
    }

    $ret = $this.Major.CompareTo($other.Major)
    if ($ret) { return $ret }

    $ret = $this.Minor.CompareTo($other.Minor)
    if ($ret) { return $ret }

    $ret = $this.Patch.CompareTo($other.Patch)
    if ($ret) { return $ret }

    # Mimic PowerShell that uses case-insensitive comparisons by default.
    $ret = [string]::Compare($this.PrereleaseLabel, $other.PrereleaseLabel, $true)
    if ($ret) { return $ret }

    $ret = $this.PrereleaseNumber.CompareTo($other.PrereleaseNumber)
    if ($ret) { return $ret }

    return ([int] $this.BuildNumber).CompareTo([int] $other.BuildNumber)
  }

  static [string[]] SortVersionStrings([string[]] $versionStrings)
  {
    $versions = $versionStrings | ForEach-Object { [AzureEngSemanticVersion]::ParseVersionString($_) }
    $sortedVersions = [AzureEngSemanticVersion]::SortVersions($versions)
    return ($sortedVersions | ForEach-Object { $_.RawVersion })
  }

  static [AzureEngSemanticVersion[]] SortVersions([AzureEngSemanticVersion[]] $versions)
  {
    return $versions | Sort-Object -Descending
  }

  static [void] QuickTests()
  {
    $global:Language = ""
    $versions = @(
      "1.0.1", 
      "2.0.0", 
      "2.0.0-alpha.20200920",
      "2.0.0-alpha.20200920.1",
      "2.0.0-beta.2", 
      "1.0.10", 
      "2.0.0-alpha.20201221.03",
      "2.0.0-alpha.20201221.1",
      "2.0.0-alpha.20201221.5",
      "2.0.0-alpha.20201221.2",
      "2.0.0-alpha.20201221.10",
      "2.0.0-beta.1", 
      "2.0.0-beta.10", 
      "1.0.0", 
      "1.0.0b2",
      "1.0.2")

    $expectedSort = @(
      "2.0.0",
      "2.0.0-beta.10",
      "2.0.0-beta.2",
      "2.0.0-beta.1",
      "2.0.0-alpha.20201221.10",
      "2.0.0-alpha.20201221.5",
      "2.0.0-alpha.20201221.03",
      "2.0.0-alpha.20201221.2",
      "2.0.0-alpha.20201221.1",
      "2.0.0-alpha.20200920.1",
      "2.0.0-alpha.20200920",
      "1.0.10",
      "1.0.2",
      "1.0.1",
      "1.0.0",
      "1.0.0b2")

    $sort = [AzureEngSemanticVersion]::SortVersionStrings($versions)

    for ($i = 0; $i -lt $expectedSort.Count; $i++)
    {
      if ($sort[$i] -ne $expectedSort[$i]) { 
        Write-Host "Error: Incorrect version sort:"
        Write-Host "Expected: "
        Write-Host $expectedSort
        Write-Host "Actual:"
        Write-Host $sort
        break
      }
    }

    $alphaVerString = "1.2.3-alpha.20200828.9"
    $alphaVer = [AzureEngSemanticVersion]::new($alphaVerString)
    if (!$alphaVer.IsPrerelease) {
      Write-Host "Expected alpha version to be marked as prerelease"
    }
    if ($alphaVer.Major -ne 1 -or $alphaVer.Minor -ne 2 -or $alphaVer.Patch -ne 3 -or `
        $alphaVer.PrereleaseLabel -ne "alpha" -or $alphaVer.PrereleaseNumber -ne 20200828 -or $alphaVer.BuildNumber -ne 9) {
      Write-Host "Error: Didn't correctly parse alpha version string $alphaVerString"
    }
    if ($alphaVerString -ne $alphaVer.ToString()) {
      Write-Host "Error: alpha string did not correctly round trip with ToString. Expected: $($alphaVerString), Actual: $($alphaVer)"
    }

    $global:Language = "python"
    $pythonAlphaVerString = "1.2.3a20200828009"
    $pythonAlphaVer = [AzureEngSemanticVersion]::new($pythonAlphaVerString)
    if (!$pythonAlphaVer.IsPrerelease) {
      Write-Host "Expected python alpha version to be marked as prerelease"
    }
    # Note: For python we lump build number into prerelease number, since it simplifies the code and regex, and is behaviorally the same
    if ($pythonAlphaVer.Major -ne 1 -or $pythonAlphaVer.Minor -ne 2 -or $pythonAlphaVer.Patch -ne 3 `
        -or $pythonAlphaVer.PrereleaseLabel -ne "a" -or $pythonAlphaVer.PrereleaseNumber -ne 20200828 `
        -or $pythonAlphaVer.BuildNumber -ne "009") {
      Write-Host "Error: Didn't correctly parse python alpha version string $pythonAlphaVerString"
    }
    if ($pythonAlphaVerString -ne $pythonAlphaVer.ToString()) {
      Write-Host "Error: python alpha string did not correctly round trip with ToString. Expected: $($pythonAlphaVerString), Actual: $($pythonAlphaVer)"
    }

    $versions = @("1.0.1", "2.0.0", "2.0.0a20201208001", "2.0.0a20201105020", "2.0.0a20201208012", `
                  "2.0.0b2", "1.0.10", "2.0.0b1", "2.0.0b10", "1.0.0", "1.0.0b2", "1.0.2")
    $expectedSort = @("2.0.0", "2.0.0b10", "2.0.0b2", "2.0.0b1", "2.0.0a20201208012", "2.0.0a20201208001", `
                      "2.0.0a20201105020", "1.0.10", "1.0.2", "1.0.1", "1.0.0", "1.0.0b2")
    $sort = [AzureEngSemanticVersion]::SortVersionStrings($versions)
    for ($i = 0; $i -lt $expectedSort.Count; $i++)
    {
      if ($sort[$i] -ne $expectedSort[$i]) { 
        Write-Host "Error: Incorrect python version sort:"
        Write-Host "Expected: "
        Write-Host $expectedSort
        Write-Host "Actual:"
        Write-Host $sort
        break
      }
    }

    $global:Language = ""

    $gaVerString = "1.2.3"
    $gaVer = [AzureEngSemanticVersion]::ParseVersionString($gaVerString)
    if ($gaVer.Major -ne 1 -or $gaVer.Minor -ne 2 -or $gaVer.Patch -ne 3) {
      Write-Host "Error: Didn't correctly parse ga version string $gaVerString"
    }
    if ($gaVerString -ne $gaVer.ToString()) {
      Write-Host "Error: Ga string did not correctly round trip with ToString. Expected: $($gaVerString), Actual: $($gaVer)"
    }
    $gaVer.IncrementAndSetToPrerelease()
    if ("1.3.0-beta.1" -ne $gaVer.ToString()) {
      Write-Host "Error: Ga string did not correctly increment"
    }

    $betaVerString = "1.2.3-beta.4"
    $betaVer = [AzureEngSemanticVersion]::ParseVersionString($betaVerString)
    if ($betaVer.Major -ne 1 -or $betaVer.Minor -ne 2 -or $betaVer.Patch -ne 3 -or $betaVer.PrereleaseLabel -ne "beta" -or $betaVer.PrereleaseNumber -ne 4) {
      Write-Host "Error: Didn't correctly parse beta version string $betaVerString"
    }
    if ($betaVerString -ne $betaVer.ToString()) {
      Write-Host "Error: beta string did not correctly round trip with ToString. Expected: $($betaVerString), Actual: $($betaVer)"
    }
    $betaVer.IncrementAndSetToPrerelease()
    if ("1.2.3-beta.5" -ne $betaVer.ToString()) {
      Write-Host "Error: Beta string did not correctly increment"
    }

    $pythonBetaVerString = "1.2.3b4"
    $pbetaVer = [AzureEngSemanticVersion]::ParsePythonVersionString($pythonBetaVerString)
    if ($pbetaVer.Major -ne 1 -or $pbetaVer.Minor -ne 2 -or $pbetaVer.Patch -ne 3 -or $pbetaVer.PrereleaseLabel -ne "b" -or $pbetaVer.PrereleaseNumber -ne 4) {
      Write-Host "Error: Didn't correctly parse python beta string $pythonBetaVerString"
    }
    if ($pythonBetaVerString -ne $pbetaVer.ToString()) {
      Write-Host "Error: python beta string did not correctly round trip with ToString"
    }
    $pbetaVer.IncrementAndSetToPrerelease()
    if ("1.2.3b5" -ne $pbetaVer.ToString()) {
      Write-Host "Error: Python beta string did not correctly increment"
    }

    Write-Host "QuickTests done"
  }
}
