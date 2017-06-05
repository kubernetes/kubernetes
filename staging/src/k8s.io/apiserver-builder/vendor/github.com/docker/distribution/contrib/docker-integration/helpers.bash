# has_digest enforces the last output line is "Digest: sha256:..."
# the input is the output from a docker push cli command
function has_digest() {
	filtered=$(echo "$1" |sed -rn '/[dD]igest\: sha(256|384|512)/ p')
	[ "$filtered" != "" ]
	# See http://wiki.alpinelinux.org/wiki/Regex#BREs before making changes to regex
	digest=$(expr "$filtered" : ".*\(sha[0-9]\{3,3\}:[a-z0-9]*\)")
}

# tempImage creates a new image using the provided name
# requires bats
function tempImage() {
	dir=$(mktemp -d)
	run dd if=/dev/urandom of="$dir/f" bs=1024 count=512
	cat <<DockerFileContent > "$dir/Dockerfile"
FROM scratch
COPY f /f

CMD []
DockerFileContent

	cp_t $dir "/tmpbuild/"
	exec_t "cd /tmpbuild/; docker build --no-cache -t $1 .; rm -rf /tmpbuild/"
}

# skip basic auth tests with Docker 1.6, where they don't pass due to
# certificate issues, requires bats
function basic_auth_version_check() {
	run sh -c 'docker version | fgrep -q "Client version: 1.6."'
	if [ "$status" -eq 0 ]; then
		skip "Basic auth tests don't support 1.6.x"
	fi
}

# login issues a login to docker to the provided server
# uses user, password, and email variables set outside of function
# requies bats
function login() {
	rm -f /root/.docker/config.json
	run docker_t login -u $user -p $password -e $email $1
	if [ "$status" -ne 0 ]; then
		echo $output
	fi
	[ "$status" -eq 0 ]
	# First line is WARNING about credential save or email deprecation (maybe both)
	[ "${lines[2]}" = "Login Succeeded" -o "${lines[1]}" = "Login Succeeded" ]
}

function login_oauth() {
	login $@

	tmpFile=$(mktemp)
	get_file_t /root/.docker/config.json $tmpFile
	run awk -v RS="" "/\"$1\": \\{[[:space:]]+\"auth\": \"[[:alnum:]]+\",[[:space:]]+\"identitytoken\"/ {exit 3}" $tmpFile
	[ "$status" -eq 3 ]
}

function parse_version() {
	version=$(echo "$1" | cut -d '-' -f1) # Strip anything after '-'
	major=$(echo "$version" | cut -d . -f1)
	minor=$(echo "$version" | cut -d . -f2)
	rev=$(echo "$version" | cut -d . -f3)

	version=$((major * 1000 * 1000 + minor * 1000 + rev))
}

function version_check() {
	name=$1
	checkv=$2
	minv=$3
	parse_version "$checkv"
	v=$version
	parse_version "$minv"
	if [ "$v" -lt "$version" ]; then
		skip "$name version \"$checkv\" does not meet required version \"$minv\""
	fi
}

function get_file_t() {
	docker cp dockerdaemon:$1 $2
}

function cp_t() {
	docker cp $1 dockerdaemon:$2
}

function exec_t() {
	docker exec dockerdaemon sh -c "$@"
}

function docker_t() {
	docker exec dockerdaemon docker $@
}

# build reates a new docker image id from another image
function build() {
	docker exec -i dockerdaemon docker build --no-cache -t $1 - <<DOCKERFILE
FROM $2
MAINTAINER distribution@docker.com
DOCKERFILE
}
