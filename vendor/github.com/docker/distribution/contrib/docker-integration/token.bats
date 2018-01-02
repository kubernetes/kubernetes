#!/usr/bin/env bats

# This tests contacting a registry using a token server

load helpers

user="testuser"
password="testpassword"
base="hello-world"

@test "Test token server login" {
	login localregistry:5554
}

@test "Test token server bad login" {
	docker_t_login -u "testuser" -p "badpassword" localregistry:5554
	[ "$status" -ne 0 ]

	docker_t_login -u "baduser" -p "testpassword" localregistry:5554
	[ "$status" -ne 0 ]
}

@test "Test push and pull with token auth" {
	login localregistry:5555
	image="localregistry:5555/testuser/token"
	build $image "$base:latest"

	run docker_t push $image
	echo $output
	[ "$status" -eq 0 ]

	docker_t rmi $image

	docker_t pull $image
}

@test "Test push and pull with token auth wrong namespace" {
	login localregistry:5555
	image="localregistry:5555/notuser/token"
	build $image "$base:latest"

	run docker_t push $image
	[ "$status" -ne 0 ]
}

@test "Test oauth token server login" {
	version_check docker "$GOLEM_DIND_VERSION" "1.11.0"

	login_oauth localregistry:5557
}

@test "Test oauth token server bad login" {
	version_check docker "$GOLEM_DIND_VERSION" "1.11.0"

	docker_t_login -u "testuser" -p "badpassword" -e $email localregistry:5557
	[ "$status" -ne 0 ]

	docker_t_login -u "baduser" -p "testpassword" -e $email localregistry:5557
	[ "$status" -ne 0 ]
}

@test "Test oauth push and pull with token auth" {
	version_check docker "$GOLEM_DIND_VERSION" "1.11.0"

	login_oauth localregistry:5558
	image="localregistry:5558/testuser/token"
	build $image "$base:latest"

	run docker_t push $image
	echo $output
	[ "$status" -eq 0 ]

	docker_t rmi $image

	docker_t pull $image
}

@test "Test oauth push and build with token auth" {
	version_check docker "$GOLEM_DIND_VERSION" "1.11.0"

	login_oauth localregistry:5558
	image="localregistry:5558/testuser/token-build"
	tempImage $image

	run docker_t push $image
	echo $output
	[ "$status" -eq 0 ]
	has_digest "$output"

	docker_t rmi $image

	image2="localregistry:5558/testuser/token-build-2"
	run build $image2 $image
	echo $output
	[ "$status" -eq 0 ]

	run docker_t push $image2
	echo $output
	[ "$status" -eq 0 ]
	has_digest "$output"

}

@test "Test oauth push and pull with token auth wrong namespace" {
	version_check docker "$GOLEM_DIND_VERSION" "1.11.0"

	login_oauth localregistry:5558
	image="localregistry:5558/notuser/token"
	build $image "$base:latest"

	run docker_t push $image
	[ "$status" -ne 0 ]
}

@test "Test oauth with v1 search" {
	version_check docker "$GOLEM_DIND_VERSION" "1.12.0"

	run docker_t search localregistry:5600/testsearch
	[ "$status" -ne 0 ]

	login_oauth localregistry:5600

	run docker_t search localregistry:5600/testsearch
	echo $output
	[ "$status" -eq 0 ]

	echo $output | grep "testsearch-1"
	echo $output | grep "testsearch-2"
}
