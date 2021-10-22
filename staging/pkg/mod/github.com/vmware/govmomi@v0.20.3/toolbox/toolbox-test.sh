#!/bin/bash -e

# Copyright 2017 VMware, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Create (or reuse) a VM to run toolbox and/or toolbox.test
# Requires ESX to be configured with:
# govc host.esxcli system settings advanced set -o /Net/GuestIPHack -i 1

set -o pipefail

vm="toolbox-test-$(uuidgen)"
destroy=true
verbose=true

while getopts n:qstv flag
do
  case $flag in
    n)
      vm=$OPTARG
      unset destroy
      ;;
    q)
      verbose=false # you want this if generating lots of traffic, such as large file transfers
      ;;
    s)
      start=true
      ;;
    t)
      test=true
      ;;
    *)
      echo "unknown option" 1>&2
      exit 1
      ;;
  esac
done

echo "Building toolbox binaries..."
pushd "$(git rev-parse --show-toplevel)" >/dev/null
GOOS=linux GOARCH=amd64 go build -o "$GOPATH/bin/toolbox" -v ./toolbox/toolbox
GOOS=linux GOARCH=amd64 go test -race -i -c ./toolbox -o "$GOPATH/bin/toolbox.test"
popd >/dev/null

iso=coreos_production_iso_image.iso

govc datastore.mkdir -p images

if ! govc datastore.ls images | grep -q $iso ; then
  echo "Downloading ${iso}..."
  if [ ! -e $iso ] ; then
    wget http://beta.release.core-os.net/amd64-usr/current/$iso
  fi

  echo "Uploading ${iso}..."
  govc datastore.upload $iso images/$iso
fi

if [ ! -e config.iso ] ; then
  echo "Generating config.iso..."
  keys=$(cat ~/.ssh/id_[rd]sa.pub)

  dir=$(mktemp -d toolbox.XXXXXX)
  pushd "${dir}" >/dev/null

  mkdir -p drive/openstack/latest

  cat > drive/openstack/latest/user_data <<EOF
#!/bin/bash

# Add ${USER}'s public key(s) to .ssh/authorized_keys
echo "$keys" | update-ssh-keys -u core -a coreos-cloudinit
EOF
  genisoimage=$(type -p genisoimage mkisofs | head -1)
  $genisoimage -R -V config-2 -o config.iso ./drive

  popd >/dev/null

  mv -f "$dir/config.iso" .
  rm -rf "$dir"
fi

govc datastore.mkdir -p "$vm"

vm_path="$(govc find / -type m -name "$vm")"

if [ -z "$vm_path" ] ; then
  echo "Creating VM ${vm}..."
  govc vm.create -g otherGuest64 -m 1024 -on=false "$vm"

  device=$(govc device.cdrom.add -vm "$vm")
  govc device.cdrom.insert -vm "$vm" -device "$device" images/$iso

  govc datastore.upload config.iso "$vm/config.iso" >/dev/null
  device=$(govc device.cdrom.add -vm "$vm")
  govc device.cdrom.insert -vm "$vm" -device "$device" "$vm/config.iso"

  vm_path="$(govc find / -type m -name "$vm")"
else
  govc object.collect -s "$vm_path" -guest.toolsStatus toolsNot* # wait for previous toolbox to unregister
fi

state=$(govc object.collect -s "$vm_path" runtime.powerState)

if [ "$state" != "poweredOn" ] ; then
  govc vm.power -on "$vm"
fi

echo -n "Waiting for ${vm} ip..."
ip=$(govc vm.ip -esxcli "$vm")

opts=(-o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking no" -o "LogLevel error" -o "BatchMode yes")

destroy() {
  if [ -n "$test" ] ; then
    rm -f "$GOVC_TLS_KNOWN_HOSTS"
  fi

  if [ -n "$destroy" ] ; then
    echo "Destroying VM ${vm}..."
    govc vm.destroy "$vm"
    govc datastore.rm -f "$vm"
  else
    ssh "${opts[@]}" "core@${ip}" pkill toolbox || true
  fi
}

trap destroy EXIT

scp "${opts[@]}" "$GOPATH"/bin/toolbox{,.test} "core@${ip}:"

if [ -n "$test" ] ; then
  # validate guest.FileManager adds the host thumbprint when transferring files
  unset GOVC_INSECURE
  GOVC_TLS_KNOWN_HOSTS=$(mktemp --tmpdir toolbox.XXXXXX)
  govc about.cert -k -thumbprint > "$GOVC_TLS_KNOWN_HOSTS"

  export GOVC_TLS_KNOWN_HOSTS GOVC_GUEST_LOGIN=user:pass

  echo "Running toolbox tests..."
  ssh "${opts[@]}" "core@${ip}" ./toolbox.test -test.v=$verbose -test.run TestServiceRunESX -toolbox.testesx \
      -toolbox.testpid="$$" -toolbox.powerState="$state" &

  echo "Waiting for VM ip from toolbox..."
  ip=$(govc vm.ip "$vm")
  echo "toolbox vm.ip=$ip"

  echo "Testing guest.{start,kill,ps} operations via govc..."
  export GOVC_VM="$vm"

  # should be 0 procs as toolbox only lists processes it started, for now
  test -z "$(govc guest.ps -e | grep -v STIME)"

  out=$(govc guest.start /bin/date)

  if [ "$out" != "$$" ] ; then
    echo "'$out' != '$$'" 1>&2
  fi

  # These processes would run for 1h if we didn't kill them.
  pid=$(govc guest.start sleep 1h)

  echo "Killing func $pid..."
  govc guest.kill -p "$pid"
  govc guest.ps -e -p "$pid" -X | grep "$pid"
  govc guest.ps -e -p "$pid" -json | jq -r .ProcessInfo[].ExitCode | grep -q 42

  pid=$(govc guest.start /bin/sh -c "sleep 3600")
  echo "Killing proc $pid..."
  govc guest.kill -p "$pid"
  govc guest.ps -e -p "$pid" -X | grep "$pid"

  echo "Testing file copy to and from guest via govc..."
  base="$(basename "$0")"
  dest="/tmp/$base"

  govc guest.upload -f -perm 0640 -gid 10 "$0" "$dest"
  govc guest.download "$dest" - | md5sum --quiet -c <(<"$0" md5sum)
  govc guest.chmod 0755 "$dest"
  govc guest.ls "$dest" | grep rwxr-xr-x

  echo "Testing custom hgfs.FileHandler..."
  if ! govc guest.download "/echo:$dest" - 2>/dev/null ; then
      govc guest.download "/echo:$dest?foo=bar" - >/dev/null
  fi

  home=$(govc guest.getenv HOME | cut -d= -f2)

  if date | govc guest.upload -f - /tmp 2>/dev/null ; then
    echo "guest.upload to directory should fail without .tgz source" 1>&2
    exit 1
  fi

  if [ "$verbose" = "false" ] ; then # else you don't want to see this noise
    # Download the $HOME directory, includes toolbox binaries (~30M total)
    # Note: trailing slash is required
    prefix=$(basename "$home")
    govc guest.download "$home/?prefix=$prefix/&format=tgz" - | tar -tzvf - | grep "$prefix"/toolbox

    govc guest.mkdir -p /tmp/toolbox-src
    # Upload source files from this directory
    # and validate that query string is not used as the file/dir name (see hgfs.ArchiveHandler)
    git archive --format tgz HEAD | govc guest.upload -f - /tmp/toolbox-src?skip=stuff
    # Upload .tar
    git archive --format tar HEAD | govc guest.upload -f - /tmp/toolbox-src
    # Download .tar
    govc guest.download "/tmp/toolbox-src/" - | tar -tvf - | grep "$base"
    # Download a single file as a .tar.gz (note: /archive: prefix is required)
    govc guest.download "/archive:/tmp/toolbox-src/$base?format=tgz" - | tar -tvzf - | grep -v README.md
    # Download a single file as a .tar (note: /archive: prefix is required)
    govc guest.download "/archive:/tmp/toolbox-src/$base" - | tar -tvf - | grep -v README.md
    govc guest.rmdir -r /tmp/toolbox-src
  fi

  echo "Testing we can download /proc files..."
  for name in uptime diskstats net/dev ; do
    test -n "$(govc guest.download /proc/$name -)"
  done

  addr=$(govc guest.getenv TOOLBOX_ECHO_SERVER | cut -d= -f2)

  echo "Testing http.RoundTrip via $addr..."
  govc guest.run GET "$addr/$vm" | grep "$vm"
  echo "$vm" | govc guest.run -e Content-Type:text/plain -d - POST "$addr" | grep "$vm"

  echo "Testing commands with IO..."
  # Note that we don't use a pipe here, as guest.ps transitions the vm to VM_STATE_GUEST_OPERATION,
  # which prevents guest.run from invoking guest operations.  By letting guest.ps complete before
  # guest.run, the vm will have transitioned back to VM_STATE_POWERED_ON.
  data=$(govc guest.ps -json)
  govc guest.run -d "$data" jq .

  # test ProcessIO download retries
  govc version | govc guest.run -d - bash -c "'sleep 2 && cat'"

  echo "Waiting for tests to complete..."
  wait
fi

if [ -n "$start" ] ; then
  (
    govc object.collect -s "$vm_path" -guest.toolsStatus toolsOk # wait for toolbox to register
    /usr/bin/time -f "Waiting for VM ip from toolbox...%e" govc vm.ip -v4 -n ethernet-0 "$vm"
  ) &

  echo "Starting toolbox..."
  ssh "${opts[@]}" "core@${ip}" ./toolbox -toolbox.trace=$verbose
fi
