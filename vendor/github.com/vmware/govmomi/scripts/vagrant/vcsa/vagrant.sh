#!/bin/sh

useradd vagrant -m -s /bin/bash
groupmod -A vagrant wheel

echo "vagrant ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

mkdir ~vagrant/.ssh
wget --no-check-certificate \
     https://raw.githubusercontent.com/mitchellh/vagrant/master/keys/vagrant.pub \
     -O ~vagrant/.ssh/authorized_keys
chown -R vagrant ~vagrant/.ssh
chmod -R go-rwsx ~vagrant/.ssh

sed -i -e 's/^#UseDNS yes/UseDNS no/' /etc/ssh/sshd_config
sed -i -e 's/^AllowTcpForwarding no//' /etc/ssh/sshd_config
sed -i -e 's/^PermitTunnel no//' /etc/ssh/sshd_config
sed -i -e 's/^MaxSessions 1//' /etc/ssh/sshd_config

# disable password expiration
for uid in root vagrant; do
  chage -I -1 -E -1 -m 0 -M -1 $uid
done
