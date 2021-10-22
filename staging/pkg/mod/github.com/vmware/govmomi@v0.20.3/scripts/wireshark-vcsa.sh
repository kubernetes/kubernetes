#!/bin/bash -e
#
# Capture SOAP traffic between web client and vpxd on 127.0.0.1:8085.
#
# Caveats: tested with VCSA 6.0, unlikely to work for other versions.
#

set -e

cache_deb() {
  wget $1
  ar x *.deb data.tar.gz
  tar zxf data.tar.gz
  rm -f data.tar.gz
  rm -f *.deb
}

dirname="$(dirname $0)"
basename="$(basename $0)"
bindir="${dirname}/.${basename}"

mkdir -p "${bindir}"

# Cache binaries required to run tcpdump on vcsa
if [ ! -f "${bindir}/.done" ]; then
  pushd ${bindir}
  cache_deb https://launchpadlibrarian.net/200649143/libssl0.9.8_0.9.8k-7ubuntu8.27_amd64.deb
  cache_deb https://launchpadlibrarian.net/37430984/libpcap0.8_1.0.0-6_amd64.deb
  cache_deb https://launchpadlibrarian.net/41774869/tcpdump_4.0.0-6ubuntu3_amd64.deb
  touch .done
  popd
fi

scp=(scp)
ssh=(ssh)

# Extract host from GOVC_URL
host="$(govc env -x GOVC_HOST)"
username=root
password="$(govc env GOVC_PASSWORD)"

if [ -x "$(which sshpass)" ] ; then
  scp=(sshpass -p "$password" scp)
  ssh=(sshpass -p "$password" ssh)
fi

ssh_opts=(-o UserKnownHostsFile=/dev/null
          -o StrictHostKeyChecking=no
          -o LogLevel=FATAL
          -o User=${username}
          -o ControlMaster=no)
dev="lo"
filter="port 8085"
tcpdump="env LD_LIBRARY_PATH=/tmp /tmp/tcpdump"

echo "Capturing $dev on $host..."

"${scp[@]}" "${ssh_opts[@]}" \
            "${bindir}/lib/libcrypto.so.0.9.8" \
            "${bindir}/usr/lib/libpcap.so.0.8" \
            "${bindir}/usr/sbin/tcpdump" \
            "${host}:/tmp"

"${ssh[@]}" "${ssh_opts[@]}" "$host" ${tcpdump} -i "$dev" -s0 -v -w - "$filter" | wireshark -k -i - 2>/dev/null
