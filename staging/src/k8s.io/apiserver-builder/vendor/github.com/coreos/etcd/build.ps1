$ORG_PATH="github.com/coreos"
$REPO_PATH="$ORG_PATH/etcd"
$PWD = $((Get-Item -Path ".\" -Verbose).FullName)

# rebuild symlinks
echo "Rebuilding symlinks"
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
		# rd so deleting junction doesn't take files with it
		cmd /c rd "$lnkname"
		cmd /c del /A /F "$lnkname"
		cmd /c mklink /J "$lnkname" "$targetAbs"
	} else {
		cmd /c del /A /F "$lnkname"
		cmd /c mklink /H "$lnkname" "$targetAbs"
	}
}

if (-not $env:GOPATH) {
	$orgpath="$PWD\gopath\src\" + $ORG_PATH.Replace("/", "\")
	cmd /c rd "$orgpath\etcd"
	cmd /c del "$orgpath"
	cmd /c mkdir "$orgpath"
	cmd /c mklink /J "$orgpath\etcd" "$PWD"
	$env:GOPATH = "$PWD\gopath"
}

# Static compilation is useful when etcd is run in a container
$env:CGO_ENABLED = 0
$env:GO15VENDOREXPERIMENT = 1
$GIT_SHA="$(git rev-parse --short HEAD)"
go build -a -installsuffix cgo -ldflags "-s -X $REPO_PATH/cmd/vendor/$REPO_PATH/version.GitSHA=$GIT_SHA" -o bin\etcd.exe "$REPO_PATH\cmd"
go build -a -installsuffix cgo -ldflags "-s" -o bin\etcdctl.exe "$REPO_PATH\cmd\etcdctl"
