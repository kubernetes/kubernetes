#!/bin/bash

set +ex

# Restart the apiserver, controller-manager, and scheduler

systemctl restart kube-apiserver

action-set 'apiserver.status' 'restarted'

systemctl restart kube-controller-manager

action-set 'controller-manager.status' 'restarted'

systemctl restart kube-scheduler

action-set 'kube-scheduler.status' 'restarted'
