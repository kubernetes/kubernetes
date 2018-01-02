$ORG_PATH="github.com/coreos"
$REPO_PATH="$ORG_PATH/etcd"
$PWD = $((Get-Item -Path ".\" -Verbose).FullName)
$FSROOT = $((Get-Location).Drive.Name+":")
$FSYS = $((Get-WMIObject win32_logicaldisk -filter "DeviceID = '$FSROOT'").filesystem)

if ($FSYS.StartsWith("FAT","CurrentCultureIgnoreCase")) {
	echo "Error: Cannot build etcd using the $FSYS filesystem (use NTFS instead)"
	exit 1
}

# Set $Env:GO_LDFLAGS="-s" for building without symbols.
$GO_LDFLAGS="$Env:GO_LDFLAGS -X $REPO_PATH/cmd/vendor/$REPO_PATH/version.GitSHA=$GIT_SHA"

# rebuild symlinks
git ls-files -s cmd | select-string -pattern 120000 | ForEach {
	$l = $_.ToString()
	$lnkname = $l.Split('	')[1]
	$target = "$(git log -p HEAD -- $lnkname | select -last 2 | select -first 1)"
	$target = $target.SubString(1,$target.Length-1).Replace("/","\")
	$lnkname = $lnkname.Replace("/","\")

	$terms = $lnkname.Split("\")
	$dirname = $terms[0..($terms.length-2)] -join "\"
	$lnkname = "$PWD\$lnkname"
	$targetAbs = "$((Get-Item -Path "$dirname\$target").FullName)"
	$targetAbs = $targetAbs.Replace("/", "\")

	if (test-path -pathtype container "$targetAbs") {
		if (Test-Path "$lnkname") {
			if ((Get-Item "$lnkname") -is [System.IO.DirectoryInfo]) {
				# rd so deleting junction doesn't take files with it
				cmd /c rd  "$lnkname"
			}
		}
		if (Test-Path "$lnkname") {
			if (!((Get-Item "$lnkname") -is [System.IO.DirectoryInfo])) {
				cmd /c del /A /F  "$lnkname"
			}
		}
		cmd /c mklink /J  "$lnkname"   "$targetAbs"  ">NUL"
	} else {
		# Remove file with symlink data (first run)
		if (Test-Path "$lnkname") {
			cmd /c del /A /F  "$lnkname"
		}
		cmd /c mklink /H  "$lnkname"   "$targetAbs"  ">NUL"
	}
}

if (-not $env:GOPATH) {
	$orgpath="$PWD\gopath\src\" + $ORG_PATH.Replace("/", "\")
	if (Test-Path "$orgpath\etcd") {
		if ((Get-Item "$orgpath\etcd") -is [System.IO.DirectoryInfo]) {
			# rd so deleting junction doesn't take files with it
			cmd /c rd  "$orgpath\etcd"
		}
	}
	if (Test-Path "$orgpath") {
		if ((Get-Item "$orgpath") -is [System.IO.DirectoryInfo]) {
			# rd so deleting junction doesn't take files with it
			cmd /c rd  "$orgpath"
		}
	}
	if (Test-Path "$orgpath") {
		if (!((Get-Item "$orgpath") -is [System.IO.DirectoryInfo])) {
			# Remove file with symlink data (first run)
			cmd /c del /A /F  "$orgpath"
		}
	}
	cmd /c mkdir  "$orgpath"
	cmd /c mklink /J  "$orgpath\etcd"   "$PWD"  ">NUL"
	$env:GOPATH = "$PWD\gopath"
}

# Static compilation is useful when etcd is run in a container
$env:CGO_ENABLED = 0
$env:GO15VENDOREXPERIMENT = 1
$GIT_SHA="$(git rev-parse --short HEAD)"
go build -a -installsuffix cgo -ldflags $GO_LDFLAGS -o bin\etcd.exe "$REPO_PATH\cmd\etcd"
go build -a -installsuffix cgo -ldflags $GO_LDFLAGS -o bin\etcdctl.exe "$REPO_PATH\cmd\etcdctl"
