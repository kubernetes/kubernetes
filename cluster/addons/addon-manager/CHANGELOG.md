### Version 9.1.8 (Mon Mar 3 2025 Jeffrey Ying <jeffrey.ying86@live.com>)
- Update kubectl to v1.32.2.
- Update base image to debian-base:bookworm-v1.0.4.

### Version 9.1.7 (Thu May 15 2023 Paco Xu <paco.xu@daocloud.io>)
- Update kubectl to v1.27.1.
- Use `--prune-allowlist` instead of deprecated `--prune-whitelist`.

### Version 9.1.6 (Thu February 24 2022 Shihang Zhang <zshihang@google.com>)
 - Clean up the wait check for service account (https://github.com/kubernetes/kubernetes/pull/108313)

### Version 9.1.5 (Mon April 19 2021 Spencer Peterson <spencerjp@google.com>)
 - Update baseimage to debian-base:v1.0.1.

### Version 9.1.4 (Wed February 10 2021 Jordan Liggitt <liggitt@google.com>)
 - Update kubectl to v1.20.2.
 - Fix a bug in leader election (https://github.com/kubernetes/kubernetes/issues/98966)

### Version 9.1.3 (Mon November 30 2020 Spencer Peterson <spencerjp@google.com>)
 - Update kubectl to v1.19.3.

### Version 9.1.2 (Thu August 6 2020 Spencer Peterson <spencerjp@google.com>)
 - Fix `start_addon` overwriting resources with `addonmanager.kubernetes.io/mode=EnsureExists`.

### Version 9.1.1 (Wed May 19 2020 Antoni Zawodny <zawodny@google.com>)
 - Fix kube-addons.sh and kubectl permissions

### Version 9.1.0 (Wed May 13 2020 Antoni Zawodny <zawodny@google.com>)
 - Enable overriding the default list of whitelisted resources

### Version 9.0.2  (Thu August 1 2019 Maciej Borsz <maciejborsz@google.com>
 - Fix a bug in leader election (https://github.com/kubernetes/kubernetes/pull/80575)

### Version 9.0.1  (Wed April 10 2019 Zihong Zheng <zihongz@google.com>)
 - Update to use debian-base:v1.0.0.

### Version 9.0  (Wed January 16 2019 Jordan Liggitt <liggitt@google.com>)
 - Prune workload resources via apps/v1 APIs
 - Update kubectl to v1.13.2.

### Version 8.9  (Fri October 19 2018 Jeff Grafton <jgrafton@google.com>)
 - Update to use debian-base:0.4.0.
 - Update kubectl to v1.11.3.

### Version 8.8  (Mon October 1 2018 Zihong Zheng <zihongz@google.com>)
 - Update to use debian-base:0.3.2.

### Version 8.7  (Tue September 4 2018 Zihong Zheng <zihongz@google.com>)
 - Support extra `--prune-whitelist` resources in kube-addon-manager.
 - Update kubectl to v1.10.7.

### Version 8.6  (Tue February 20 2018 Zihong Zheng <zihongz@google.com>)
 - Allow reconcile/ensure loop to work with resource under non-kube-system namespace.
 - Update kubectl to v1.9.3.

### Version 8.4  (Thu November 30 2017 zou nengren @zouyee)
 - Update kubectl to v1.8.4.

### Version 6.5  (Wed October 15 2017 Daniel Kłobuszewski <danielmk@google.com>)
 - Support for HA masters.

### Version 6.4-beta.2  (Mon June 12 2017 Jeff Grafton <jgrafton@google.com>)
 - Update kubectl to v1.6.4.
 - Refresh base images.

### Version 6.4-beta.1  (Wed March 8 2017 Zihong Zheng <zihongz@google.com>)
 - Create EnsureExists class addons before Reconcile class addons.

### Version 6.4-alpha.3  (Fri February 24 2017 Zihong Zheng <zihongz@google.com>)
 - Support 'ensure exist' class addon and use addon-manager specific label.

### Version 6.4-alpha.2 (Wed February 16 2017 Zihong Zheng <zihongz@google.com>)
 - Update kubectl to v1.6.0-alpha.2 to use HPA in autoscaling/v1 instead of extensions/v1beta1.

### Version 6.4-alpha.1 (Wed February 1 2017 Zihong Zheng <zihongz@google.com>)
 - Update kubectl to v1.6.0-alpha.1 for supporting optional ConfigMap.

### Version 6.3 (Fri January 27 2017 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk>)
 - Updated the arm base image to `armhf/busybox` and now using qemu v2.7 for emulation.

### Version 6.2 (Thu January 12 2017 Zihong Zheng <zihongz@google.com>)
 - Update kubectl to the stable version.

### Version 6.1 (Tue November 29 2016 Zihong Zheng <zihongz@google.com>)
 - Support pruning old Deployments.

### Version 6.0 (Fri November 18 2016 Zihong Zheng <zihongz@google.com>)
 - Upgrade Addon Manager to use `kubectl apply`.

### Version 5.2 (Wed October 26 2016 Zihong Zheng <zihongz@google.com>)
 - Added support for ConfigMap and upgraded kubectl version to v1.4.4 (pr #35255)

### Version 5.1 (Mon Jul 4 2016 Marek Grabowski <gmarek@google.com>)
 - Fixed the way addon-manager handles non-namespaced objects

### Version 5 (Fri Jun 24 2016 Jerzy Szczepkowski @jszczepkowski)
 - Added PetSet support to addon manager

### Version 4 (Tue Jun 21 2016 Mike Danese @mikedanese)
 - Increased addon check interval

### Version 3 (Sun Jun 19 2016 Lucas Käldström @luxas)
 - Bumped up addon-manager to v3

### Version 2 (Fri May 20 2016 Lucas Käldström @luxas)
 - Removed deprecated kubectl command, added support for DaemonSets

### Version 1 (Thu May 5 2016 Mike Danese @mikedanese)
 - Run kube-addon-manager in a pod
