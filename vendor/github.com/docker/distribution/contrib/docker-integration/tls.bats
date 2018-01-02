#!/usr/bin/env bats

# Registry host name, should be set to non-localhost address and match
# DNS name in nginx/ssl certificates and what is installed in /etc/docker/cert.d

load helpers

hostname="localregistry"
base="hello-world"
image="${base}:latest"

# Login information, should match values in nginx/test.passwd
user=${TEST_USER:-"testuser"}
password=${TEST_PASSWORD:-"passpassword"}

function setup() {
	tempImage $image
}

@test "Test valid certificates" {
	docker_t tag $image $hostname:5440/$image
	run docker_t push $hostname:5440/$image
	[ "$status" -eq 0 ]
	has_digest "$output"
}

@test "Test basic auth" {
	basic_auth_version_check
	login $hostname:5441
	docker_t tag $image $hostname:5441/$image
	run docker_t push $hostname:5441/$image
	[ "$status" -eq 0 ]
	has_digest "$output"
}

@test "Test basic auth with build" {
	basic_auth_version_check
	login $hostname:5441

	image1=$hostname:5441/$image-build
	image2=$hostname:5441/$image-build-2

	tempImage $image1

	run docker_t push $image1
	[ "$status" -eq 0 ]
	has_digest "$output"

	docker_t rmi $image1

	run build $image2 $image1
	echo $output
	[ "$status" -eq 0 ]

	run docker_t push $image2
	echo $output
	[ "$status" -eq 0 ]
	has_digest "$output"
}

@test "Test TLS client auth" {
	docker_t tag $image $hostname:5442/$image
	run docker_t push $hostname:5442/$image
	[ "$status" -eq 0 ]
	has_digest "$output"
}

@test "Test TLS client with invalid certificate authority fails" {
	docker_t tag $image $hostname:5443/$image
	run docker_t push $hostname:5443/$image
	[ "$status" -ne 0 ]
}

@test "Test basic auth with TLS client auth" {
	basic_auth_version_check
	login $hostname:5444
	docker_t tag $image $hostname:5444/$image
	run docker_t push $hostname:5444/$image
	[ "$status" -eq 0 ]
	has_digest "$output"
}

@test "Test unknown certificate authority fails" {
	docker_t tag $image $hostname:5445/$image
	run docker_t push $hostname:5445/$image
	[ "$status" -ne 0 ]
}

@test "Test basic auth with unknown certificate authority fails" {
	run login $hostname:5446
	[ "$status" -ne 0 ]
	docker_t tag $image $hostname:5446/$image
	run docker_t push $hostname:5446/$image
	[ "$status" -ne 0 ]
}

@test "Test TLS client auth to server with unknown certificate authority fails" {
	docker_t tag $image $hostname:5447/$image
	run docker_t push $hostname:5447/$image
	[ "$status" -ne 0 ]
}

@test "Test failure to connect to server fails to fallback to SSLv3" {
	docker_t tag $image $hostname:5448/$image
	run docker_t push $hostname:5448/$image
	[ "$status" -ne 0 ]
}

