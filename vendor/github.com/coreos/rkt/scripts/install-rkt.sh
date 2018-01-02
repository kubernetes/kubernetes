#!/bin/bash
set -e
set -x

cd $(mktemp -d)

version="1.25.0"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg2 \
        bash-completion \
        checkinstall \
        curl \
        iptables \
        wget

curl -sSL https://coreos.com/dist/pubkeys/app-signing-pubkey.gpg | gpg2 --import -
key=$(gpg2 --with-colons --keyid-format LONG -k security@coreos.com | egrep ^pub | cut -d ':' -f5)

wget --progress=bar:force https://github.com/coreos/rkt/releases/download/v"${version}"/rkt-v"${version}".tar.gz
wget --progress=bar:force https://github.com/coreos/rkt/releases/download/v"${version}"/rkt-v"${version}".tar.gz.asc
gpg2 --trusted-key "${key}" --verify-files *.asc

tar xvzf rkt-v"${version}".tar.gz

cat <<EOF >install-pak
#!/bin/bash

# abort/fail on any error
set -e

# fix mkdir issues with checkinstall and fstrans
for dir in /usr/lib/rkt/stage1-images/\\
        /usr/share/man/man1/\\
        /usr/share/bash-completion/completions/\\
        /usr/lib/tmpfiles.d/\\
        /usr/lib/systemd/system/
do
    mkdir -p \$dir 2>/dev/null || :
done

for flavor in fly coreos kvm; do
    install -Dm644 rkt-v${version}/stage1-\${flavor}.aci /usr/lib/rkt/stage1-images/stage1-\${flavor}.aci
done

install -Dm755 rkt-v${version}/rkt /usr/bin/rkt

for f in rkt-v${version}/manpages/*; do
    install -Dm644 "\${f}" "/usr/share/man/man1/\$(basename \$f)"
done

install -Dm644 rkt-v${version}/bash_completion/rkt.bash /usr/share/bash-completion/completions/rkt
install -Dm644 rkt-v${version}/init/systemd/tmpfiles.d/rkt.conf /usr/lib/tmpfiles.d/rkt.conf

for unit in rkt-gc.{timer,service} rkt-metadata.{socket,service}; do
    install -Dm644 rkt-v${version}/init/systemd/\$unit /usr/lib/systemd/system/\$unit
done
EOF
chmod +x install-pak

cat <<EOF >preinstall-pak
#!/bin/sh

groupadd --force --system rkt-admin
groupadd --force --system rkt
EOF
chmod +x preinstall-pak

cp rkt-v"${version}"/scripts/setup-data-dir.sh postinstall-pak
chmod +x postinstall-pak

cat <<EOF >>postinstall-pak
systemctl daemon-reload
systemd-tmpfiles --create /usr/lib/tmpfiles.d/rkt.conf
EOF

checkinstall -y --pkgname=rkt --pkgversion="${version}" ./install-pak
