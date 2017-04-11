#!/bin/sh
set -ux

for service in kube-apiserver kube-controller-manager kube-scheduler; do
  systemctl status $service > $DEBUG_SCRIPT_DIR/$service-systemctl-status
  journalctl -u $service > $DEBUG_SCRIPT_DIR/$service-journal
done

mkdir -p $DEBUG_SCRIPT_DIR/etc-default
cp -v /etc/default/kube* $DEBUG_SCRIPT_DIR/etc-default

mkdir -p $DEBUG_SCRIPT_DIR/lib-systemd-system
cp -v /lib/systemd/system/kube* $DEBUG_SCRIPT_DIR/lib-systemd-system
