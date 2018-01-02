#!/usr/bin/env bats

# This tests pushing and pulling plugins

load helpers

user="testuser"
password="testpassword"
base="hello-world"

#TODO: Create plugin image
function create_plugin() {
	plugindir=$(mktemp -d)

	cat - > $plugindir/config.json <<CONFIGJSON
{
	"manifestVersion": "v0",
	"description": "A test plugin for integration tests",
	"entrypoint": ["/usr/bin/ncat", "-l", "-U", "//run/docker/plugins/plugin.sock"],
	"interface" : {
		"types": ["docker.volumedriver/1.0"],
		"socket": "plugin.sock"
	}
}
CONFIGJSON

	cid=$(docker create dmcgowan/ncat:latest /bin/sh)

	mkdir $plugindir/rootfs

	docker export $cid | tar -x -C $plugindir/rootfs

	docker rm $cid

	daemontmp=$(docker exec dockerdaemon mktemp -d)

	tar -c -C $plugindir . | docker exec -i dockerdaemon tar -x -C $daemontmp

	docker exec dockerdaemon docker plugin create $1 $daemontmp

	docker exec dockerdaemon rm -rf $daemontmp

	rm -rf $plugindir
}

@test "Test plugin push and pull" {
	version_check docker "$GOLEM_DIND_VERSION" "1.13.0-rc3"
	version_check docker "$GOLEM_DISTRIBUTION_VERSION" "2.6.0"

	login_oauth localregistry:5558
	image="localregistry:5558/testuser/plugin1"

	create_plugin $image

	run docker_t plugin push $image
	echo $output
	[ "$status" -eq 0 ]

	docker_t plugin rm $image

	docker_t plugin install --grant-all-permissions $image
}

@test "Test plugin push and failed image pull" {
	version_check docker "$GOLEM_DIND_VERSION" "1.13.0-rc3"
	version_check docker "$GOLEM_DISTRIBUTION_VERSION" "2.6.0"


	login_oauth localregistry:5558
	image="localregistry:5558/testuser/plugin-not-image"

	create_plugin $image

	run docker_t plugin push $image
	echo $output
	[ "$status" -eq 0 ]

	docker_t plugin rm $image

	run docker_t pull $image

	[ "$status" -ne 0 ]
}

@test "Test image push and failed plugin pull" {
	version_check docker "$GOLEM_DIND_VERSION" "1.13.0-rc3"
	version_check docker "$GOLEM_DISTRIBUTION_VERSION" "2.6.0"

	login_oauth localregistry:5558
	image="localregistry:5558/testuser/image-not-plugin"

	build $image "$base:latest"

	run docker_t push $image
	echo $output
	[ "$status" -eq 0 ]

	docker_t rmi $image

	run docker_t plugin install --grant-all-permissions $image

	[ "$status" -ne 0 ]
}
