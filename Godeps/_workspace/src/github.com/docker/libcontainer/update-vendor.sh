#!/usr/bin/env bash
set -e

cd "$(dirname "$BASH_SOURCE")"

# Downloads dependencies into vendor/ directory
mkdir -p vendor
cd vendor

clone() {
	vcs=$1
	pkg=$2
	rev=$3
	
	pkg_url=https://$pkg
	target_dir=src/$pkg
	
	echo -n "$pkg @ $rev: "
	
	if [ -d $target_dir ]; then
		echo -n 'rm old, '
		rm -fr $target_dir
	fi
	
	echo -n 'clone, '
	case $vcs in
		git)
			git clone --quiet --no-checkout $pkg_url $target_dir
			( cd $target_dir && git reset --quiet --hard $rev )
			;;
		hg)
			hg clone --quiet --updaterev $rev $pkg_url $target_dir
			;;
	esac
	
	echo -n 'rm VCS, '
	( cd $target_dir && rm -rf .{git,hg} )
	
	echo done
}

# the following lines are in sorted order, FYI
clone git github.com/codegangsta/cli 1.1.0
clone git github.com/coreos/go-systemd v2
clone git github.com/godbus/dbus v2
clone git github.com/Sirupsen/logrus v0.7.3
clone git github.com/syndtr/gocapability 8e4cdcb
clone git github.com/golang/protobuf 655cdfa588ea

# intentionally not vendoring Docker itself...  that'd be a circle :)
