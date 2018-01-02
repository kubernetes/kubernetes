<#
.NOTES
    Author:  @jhowardmsft

    Summary: Windows native build script. This is similar to functionality provided
             by hack\make.sh, but uses native Windows PowerShell semantics. It does
             not support the full set of options provided by the Linux counterpart.
             For example:

             - You can't cross-build Linux docker binaries on Windows
             - Hashes aren't generated on binaries
             - 'Releasing' isn't supported.
             - Integration tests. This is because they currently cannot run inside a container,
               and require significant external setup.

             It does however provided the minimum necessary to support parts of local Windows
             development and Windows to Windows CI.

             Usage Examples (run from repo root):
                "hack\make.ps1 -Client" to build docker.exe client 64-bit binary (remote repo)
                "hack\make.ps1 -TestUnit" to run unit tests
                "hack\make.ps1 -Daemon -TestUnit" to build the daemon and run unit tests
                "hack\make.ps1 -All" to run everything this script knows about that can run in a container
                "hack\make.ps1" to build the daemon binary (same as -Daemon)
                "hack\make.ps1 -Binary" shortcut to -Client and -Daemon

.PARAMETER Client
     Builds the client binaries.

.PARAMETER Daemon
     Builds the daemon binary.

.PARAMETER Binary
     Builds the client and daemon binaries. A convenient shortcut to `make.ps1 -Client -Daemon`.

.PARAMETER Race
     Use -race in go build and go test.

.PARAMETER Noisy
     Use -v in go build.

.PARAMETER ForceBuildAll
     Use -a in go build.

.PARAMETER NoOpt
     Use -gcflags -N -l in go build to disable optimisation (can aide debugging).

.PARAMETER CommitSuffix
     Adds a custom string to be appended to the commit ID (spaces are stripped).

.PARAMETER DCO
     Runs the DCO (Developer Certificate Of Origin) test (must be run outside a container).

.PARAMETER PkgImports
     Runs the pkg\ directory imports test (must be run outside a container).

.PARAMETER GoFormat
     Runs the Go formatting test (must be run outside a container).

.PARAMETER TestUnit
     Runs unit tests.

.PARAMETER All
     Runs everything this script knows about that can run in a container.


TODO
- Unify the head commit
- Add golint and other checks (swagger maybe?)

#>


param(
    [Parameter(Mandatory=$False)][switch]$Client,
    [Parameter(Mandatory=$False)][switch]$Daemon,
    [Parameter(Mandatory=$False)][switch]$Binary,
    [Parameter(Mandatory=$False)][switch]$Race,
    [Parameter(Mandatory=$False)][switch]$Noisy,
    [Parameter(Mandatory=$False)][switch]$ForceBuildAll,
    [Parameter(Mandatory=$False)][switch]$NoOpt,
    [Parameter(Mandatory=$False)][string]$CommitSuffix="",
    [Parameter(Mandatory=$False)][switch]$DCO,
    [Parameter(Mandatory=$False)][switch]$PkgImports,
    [Parameter(Mandatory=$False)][switch]$GoFormat,
    [Parameter(Mandatory=$False)][switch]$TestUnit,
    [Parameter(Mandatory=$False)][switch]$All
)

$ErrorActionPreference = "Stop"
$pushed=$False  # To restore the directory if we have temporarily pushed to one.

# Utility function to get the commit ID of the repository
Function Get-GitCommit() {
    if (-not (Test-Path ".\.git")) {
        # If we don't have a .git directory, but we do have the environment
        # variable DOCKER_GITCOMMIT set, that can override it.
        if ($env:DOCKER_GITCOMMIT.Length -eq 0) {
            Throw ".git directory missing and DOCKER_GITCOMMIT environment variable not specified."
        }
        Write-Host "INFO: Git commit ($env:DOCKER_GITCOMMIT) assumed from DOCKER_GITCOMMIT environment variable"
        return $env:DOCKER_GITCOMMIT
    }
    $gitCommit=$(git rev-parse --short HEAD)
    if ($(git status --porcelain --untracked-files=no).Length -ne 0) {
        $gitCommit="$gitCommit-unsupported"
        Write-Host ""
        Write-Warning "This version is unsupported because there are uncommitted file(s)."
        Write-Warning "Either commit these changes, or add them to .gitignore."
        git status --porcelain --untracked-files=no | Write-Warning
        Write-Host ""
    }
    return $gitCommit
}

# Utility function to get get the current build version of docker
Function Get-DockerVersion() {
    if (-not (Test-Path ".\VERSION")) { Throw "VERSION file not found. Is this running from the root of a docker repository?" }
    return $(Get-Content ".\VERSION" -raw).ToString().Replace("`n","").Trim()
}

# Utility function to determine if we are running in a container or not.
# In Windows, we get this through an environment variable set in `Dockerfile.Windows`
Function Check-InContainer() {
    if ($env:FROM_DOCKERFILE.Length -eq 0) {
        Write-Host ""
        Write-Warning "Not running in a container. The result might be an incorrect build."
        Write-Host ""
        return $False
    }
    return $True
}

# Utility function to warn if the version of go is correct. Used for local builds
# outside of a container where it may be out of date with master.
Function Verify-GoVersion() {
    Try {
        $goVersionDockerfile=(Get-Content ".\Dockerfile" | Select-String "ENV GO_VERSION").ToString().Split(" ")[2]
        $goVersionInstalled=(go version).ToString().Split(" ")[2].SubString(2)
    }
    Catch [Exception] {
        Throw "Failed to validate go version correctness: $_"
    }
    if (-not($goVersionInstalled -eq $goVersionDockerfile)) {
        Write-Host ""
        Write-Warning "Building with golang version $goVersionInstalled. You should update to $goVersionDockerfile"
        Write-Host ""
    }
}

# Utility function to get the commit for HEAD
Function Get-HeadCommit() {
    $head = Invoke-Expression "git rev-parse --verify HEAD"
    if ($LASTEXITCODE -ne 0) { Throw "Failed getting HEAD commit" }

    return $head
}

# Utility function to get the commit for upstream
Function Get-UpstreamCommit() {
    Invoke-Expression "git fetch -q https://github.com/docker/docker.git refs/heads/master"
    if ($LASTEXITCODE -ne 0) { Throw "Failed fetching" }

    $upstream = Invoke-Expression "git rev-parse --verify FETCH_HEAD"
    if ($LASTEXITCODE -ne 0) { Throw "Failed getting upstream commit" }

    return $upstream
}

# Build a binary (client or daemon)
Function Execute-Build($type, $additionalBuildTags, $directory) {
    # Generate the build flags
    $buildTags = "autogen"
    if ($Noisy)                     { $verboseParm=" -v" }
    if ($Race)                      { Write-Warning "Using race detector"; $raceParm=" -race"}
    if ($ForceBuildAll)             { $allParm=" -a" }
    if ($NoOpt)                     { $optParm=" -gcflags "+""""+"-N -l"+"""" }
    if ($additionalBuildTags -ne "") { $buildTags += $(" " + $additionalBuildTags) }

    # Do the go build in the appropriate directory
    # Note -linkmode=internal is required to be able to debug on Windows.
    # https://github.com/golang/go/issues/14319#issuecomment-189576638
    Write-Host "INFO: Building $type..."
    Push-Location $root\cmd\$directory; $global:pushed=$True
    $buildCommand = "go build" + `
                    $raceParm + `
                    $verboseParm + `
                    $allParm + `
                    $optParm + `
                    " -tags """ + $buildTags + """" + `
                    " -ldflags """ + "-linkmode=internal" + """" + `
                    " -o $root\bundles\"+$directory+".exe"
    Invoke-Expression $buildCommand
    if ($LASTEXITCODE -ne 0) { Throw "Failed to compile $type" }
    Pop-Location; $global:pushed=$False
}


# Validates the DCO marker is present on each commit
Function Validate-DCO($headCommit, $upstreamCommit) {
    Write-Host "INFO: Validating Developer Certificate of Origin..."
    # Username may only contain alphanumeric characters or dashes and cannot begin with a dash
    $usernameRegex='[a-zA-Z0-9][a-zA-Z0-9-]+'

    $dcoPrefix="Signed-off-by:"
    $dcoRegex="^(Docker-DCO-1.1-)?$dcoPrefix ([^<]+) <([^<>@]+@[^<>]+)>( \(github: ($usernameRegex)\))?$"

    $counts = Invoke-Expression "git diff --numstat $upstreamCommit...$headCommit"
    if ($LASTEXITCODE -ne 0) { Throw "Failed git diff --numstat" }

    # Counts of adds and deletes after removing multiple white spaces. AWK anyone? :(
    $adds=0; $dels=0; $($counts -replace '\s+', ' ') | %{ 
        $a=$_.Split(" "); 
        if ($a[0] -ne "-") { $adds+=[int]$a[0] }
        if ($a[1] -ne "-") { $dels+=[int]$a[1] }
    }
    if (($adds -eq 0) -and ($dels -eq 0)) { 
        Write-Warning "DCO validation - nothing to validate!"
        return
    }

    $commits = Invoke-Expression "git log  $upstreamCommit..$headCommit --format=format:%H%n"
    if ($LASTEXITCODE -ne 0) { Throw "Failed git log --format" }
    $commits = $($commits -split '\s+' -match '\S')
    $badCommits=@()
    $commits | %{
        # Skip commits with no content such as merge commits etc
        if ($(git log -1 --format=format: --name-status $_).Length -gt 0) {
            # Ignore exit code on next call - always process regardless
            $commitMessage = Invoke-Expression "git log -1 --format=format:%B --name-status $_"
            if (($commitMessage -match $dcoRegex).Length -eq 0) { $badCommits+=$_ }
        }
    }
    if ($badCommits.Length -eq 0) {
        Write-Host "Congratulations!  All commits are properly signed with the DCO!"
    } else {
        $e = "`nThese commits do not have a proper '$dcoPrefix' marker:`n"
        $badCommits | %{ $e+=" - $_`n"}
        $e += "`nPlease amend each commit to include a properly formatted DCO marker.`n`n"
        $e += "Visit the following URL for information about the Docker DCO:`n"
        $e += "https://github.com/docker/docker/blob/master/CONTRIBUTING.md#sign-your-work`n"
        Throw $e
    }
}

# Validates that .\pkg\... is safely isolated from internal code
Function Validate-PkgImports($headCommit, $upstreamCommit) {
    Write-Host "INFO: Validating pkg import isolation..."

    # Get a list of go source-code files which have changed under pkg\. Ignore exit code on next call - always process regardless
    $files=@(); $files = Invoke-Expression "git diff $upstreamCommit...$headCommit --diff-filter=ACMR --name-only -- `'pkg\*.go`'"
    $badFiles=@(); $files | %{
        $file=$_
        # For the current changed file, get its list of dependencies, sorted and uniqued.
        $imports = Invoke-Expression "go list -e -f `'{{ .Deps }}`' $file"
        if ($LASTEXITCODE -ne 0) { Throw "Failed go list for dependencies on $file" }
        $imports = $imports -Replace "\[" -Replace "\]", "" -Split(" ") | Sort-Object | Get-Unique
        # Filter out what we are looking for
        $imports = $imports -NotMatch "^github.com/docker/docker/pkg/" `
                            -NotMatch "^github.com/docker/docker/vendor" `
                            -Match "^github.com/docker/docker" `
                            -Replace "`n", ""
        $imports | % { $badFiles+="$file imports $_`n" }
    }
    if ($badFiles.Length -eq 0) {
        Write-Host 'Congratulations!  ".\pkg\*.go" is safely isolated from internal code.'
    } else {
        $e = "`nThese files import internal code: (either directly or indirectly)`n"
        $badFiles | %{ $e+=" - $_"}
        Throw $e
    }
}

# Validates that changed files are correctly go-formatted
Function Validate-GoFormat($headCommit, $upstreamCommit) {
    Write-Host "INFO: Validating go formatting on changed files..."

    # Verify gofmt is installed
    if ($(Get-Command gofmt -ErrorAction SilentlyContinue) -eq $nil) { Throw "gofmt does not appear to be installed" }

    # Get a list of all go source-code files which have changed.  Ignore exit code on next call - always process regardless
    $files=@(); $files = Invoke-Expression "git diff $upstreamCommit...$headCommit --diff-filter=ACMR --name-only -- `'*.go`'"
    $files = $files | Select-String -NotMatch "^vendor/"
    $badFiles=@(); $files | %{
        # Deliberately ignore error on next line - treat as failed
        $content=Invoke-Expression "git show $headCommit`:$_"

        # Next set of hoops are to ensure we have LF not CRLF semantics as otherwise gofmt on Windows will not succeed.
        # Also note that gofmt on Windows does not appear to support stdin piping correctly. Hence go through a temporary file.
        $content=$content -join "`n"
        $content+="`n"
        $outputFile=[System.IO.Path]::GetTempFileName()
        if (Test-Path $outputFile) { Remove-Item $outputFile }
        [System.IO.File]::WriteAllText($outputFile, $content, (New-Object System.Text.UTF8Encoding($False)))
        $currentFile = $_ -Replace("/","\")
        Write-Host Checking $currentFile
        Invoke-Expression "gofmt -s -l $outputFile"
        if ($LASTEXITCODE -ne 0) { $badFiles+=$currentFile }
        if (Test-Path $outputFile) { Remove-Item $outputFile }
    }
    if ($badFiles.Length -eq 0) {
        Write-Host 'Congratulations!  All Go source files are properly formatted.'
    } else {
        $e = "`nThese files are not properly gofmt`'d:`n"
        $badFiles | %{ $e+=" - $_`n"}
        $e+= "`nPlease reformat the above files using `"gofmt -s -w`" and commit the result."
        Throw $e
    }
}

# Run the unit tests
Function Run-UnitTests() {
    Write-Host "INFO: Running unit tests..."
    $testPath="./..."
    $goListCommand = "go list -e -f '{{if ne .Name """ + '\"github.com/docker/docker\"' + """}}{{.ImportPath}}{{end}}' $testPath"
    $pkgList = $(Invoke-Expression $goListCommand)
    if ($LASTEXITCODE -ne 0) { Throw "go list for unit tests failed" }
    $pkgList = $pkgList | Select-String -Pattern "github.com/docker/docker"
    $pkgList = $pkgList | Select-String -NotMatch "github.com/docker/docker/vendor"
    $pkgList = $pkgList | Select-String -NotMatch "github.com/docker/docker/man"
    $pkgList = $pkgList | Select-String -NotMatch "github.com/docker/docker/integration-cli"
    $pkgList = $pkgList -replace "`r`n", " "
    $goTestCommand = "go test" + $raceParm + " -cover -ldflags -w -tags """ + "autogen daemon" + """ -a """ + "-test.timeout=10m" + """ $pkgList"
    Invoke-Expression $goTestCommand
    if ($LASTEXITCODE -ne 0) { Throw "Unit tests failed" }
}

# Start of main code.
Try {
    Write-Host -ForegroundColor Cyan "INFO: make.ps1 starting at $(Get-Date)"

    # Get to the root of the repo
    $root = $(Split-Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent)
    Push-Location $root

    # Handle the "-All" shortcut to turn on all things we can handle.
    # Note we expressly only include the items which can run in a container - the validations tests cannot
    # as they require the .git directory which is excluded from the image by .dockerignore
    if ($All) { $Client=$True; $Daemon=$True; $TestUnit=$True }

    # Handle the "-Binary" shortcut to build both client and daemon.
    if ($Binary) { $Client = $True; $Daemon = $True }

    # Default to building the daemon if not asked for anything explicitly.
    if (-not($Client) -and -not($Daemon) -and -not($DCO) -and -not($PkgImports) -and -not($GoFormat) -and -not($TestUnit)) { $Daemon=$True }

    # Verify git is installed
    if ($(Get-Command git -ErrorAction SilentlyContinue) -eq $nil) { Throw "Git does not appear to be installed" }

    # Verify go is installed
    if ($(Get-Command go -ErrorAction SilentlyContinue) -eq $nil) { Throw "GoLang does not appear to be installed" }

    # Get the git commit. This will also verify if we are in a repo or not. Then add a custom string if supplied.
    $gitCommit=Get-GitCommit
    if ($CommitSuffix -ne "") { $gitCommit += "-"+$CommitSuffix -Replace ' ', '' }

    # Get the version of docker (eg 17.04.0-dev)
    $dockerVersion=Get-DockerVersion

    # Give a warning if we are not running in a container and are building binaries or running unit tests.
    # Not relevant for validation tests as these are fine to run outside of a container.
    if ($Client -or $Daemon -or $TestUnit) { $inContainer=Check-InContainer }

    # If we are not in a container, validate the version of GO that is installed.
    if (-not $inContainer) { Verify-GoVersion }

    # Verify GOPATH is set
    if ($env:GOPATH.Length -eq 0) { Throw "Missing GOPATH environment variable. See https://golang.org/doc/code.html#GOPATH" }

    # Run autogen if building binaries or running unit tests.
    if ($Client -or $Daemon -or $TestUnit) {
        Write-Host "INFO: Invoking autogen..."
        Try { .\hack\make\.go-autogen.ps1 -CommitString $gitCommit -DockerVersion $dockerVersion }
        Catch [Exception] { Throw $_ }
    }

    # DCO, Package import and Go formatting tests.
    if ($DCO -or $PkgImports -or $GoFormat) {
        # We need the head and upstream commits for these
        $headCommit=Get-HeadCommit
        $upstreamCommit=Get-UpstreamCommit

        # Run DCO validation
        if ($DCO) { Validate-DCO $headCommit $upstreamCommit }

        # Run `gofmt` validation
        if ($GoFormat) { Validate-GoFormat $headCommit $upstreamCommit }

        # Run pkg isolation validation
        if ($PkgImports) { Validate-PkgImports $headCommit $upstreamCommit }
    }

    # Build the binaries
    if ($Client -or $Daemon) {
        # Create the bundles directory if it doesn't exist
        if (-not (Test-Path ".\bundles")) { New-Item ".\bundles" -ItemType Directory | Out-Null }

        # Perform the actual build
        if ($Daemon) { Execute-Build "daemon" "daemon" "dockerd" }
        if ($Client) {
            # Get the repo and commit of the client to build.
            "hack\dockerfile\binaries-commits" | ForEach-Object {
                $dockerCliRepo = ((Get-Content $_ | Select-String "DOCKERCLI_REPO") -split "=")[1]
                $dockerCliCommit = ((Get-Content $_ | Select-String "DOCKERCLI_COMMIT") -split "=")[1]
            }

            # Build from a temporary directory.
            $tempLocation = "$env:TEMP\$(New-Guid)"
            New-Item -ItemType Directory $tempLocation | Out-Null

            # Temporarily override GOPATH, then clone, checkout, and build.
            $saveGOPATH = $env:GOPATH
            Try {
                $env:GOPATH = $tempLocation
                $dockerCliRoot = "$env:GOPATH\src\github.com\docker\cli"
                Write-Host "INFO: Cloning client repository..."
                Invoke-Expression "git clone -q $dockerCliRepo $dockerCliRoot"
                if ($LASTEXITCODE -ne 0) { Throw "Failed to clone client repository $dockerCliRepo" }
                Invoke-Expression "git -C $dockerCliRoot  checkout -q $dockerCliCommit"
                if ($LASTEXITCODE -ne 0) { Throw "Failed to checkout client commit $dockerCliCommit" }
                Write-Host "INFO: Building client..."
                Push-Location "$dockerCliRoot\cmd\docker"; $global:pushed=$True
                Invoke-Expression "go build -o $root\bundles\docker.exe"
                if ($LASTEXITCODE -ne 0) { Throw "Failed to compile client" }
                Pop-Location; $global:pushed=$False
            }
            Catch [Exception] {
                Throw $_
            }
            Finally {
                # Always restore GOPATH and remove the temporary directory.
                $env:GOPATH = $saveGOPATH
                Remove-Item -Force -Recurse $tempLocation
            }
        }
    }

    # Run unit tests
    if ($TestUnit) { Run-UnitTests }

    # Gratuitous ASCII art.
    if ($Daemon -or $Client) {
        Write-Host
        Write-Host -ForegroundColor Green " ________   ____  __."
        Write-Host -ForegroundColor Green " \_____  \ `|    `|/ _`|"
        Write-Host -ForegroundColor Green " /   `|   \`|      `<"
        Write-Host -ForegroundColor Green " /    `|    \    `|  \"
        Write-Host -ForegroundColor Green " \_______  /____`|__ \"
        Write-Host -ForegroundColor Green "         \/        \/"
        Write-Host
    }
}
Catch [Exception] {
    Write-Host -ForegroundColor Red ("`nERROR: make.ps1 failed:`n$_")

    # More gratuitous ASCII art.
    Write-Host
    Write-Host -ForegroundColor Red  "___________      .__.__             .___"
    Write-Host -ForegroundColor Red  "\_   _____/____  `|__`|  `|   ____   __`| _/"
    Write-Host -ForegroundColor Red  " `|    __) \__  \ `|  `|  `| _/ __ \ / __ `| "
    Write-Host -ForegroundColor Red  " `|     \   / __ \`|  `|  `|_\  ___// /_/ `| "
    Write-Host -ForegroundColor Red  " \___  /  (____  /__`|____/\___  `>____ `| "
    Write-Host -ForegroundColor Red  "     \/        \/             \/     \/ "
    Write-Host

    Throw $_
}
Finally {
    Pop-Location # As we pushed to the root of the repo as the very first thing
    if ($global:pushed) { Pop-Location }
    Write-Host -ForegroundColor Cyan "INFO: make.ps1 ended at $(Get-Date)"
}
