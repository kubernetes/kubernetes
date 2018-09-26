Name: kubeadm
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Cluster Manager - Kubernetes Cluster Bootstrapping Tool
Requires: kubelet >= 1.8.0
Requires: kubectl >= 1.8.0
Requires: kubernetes-cni >= 0.5.1
Requires: cri-tools >= 1.11.0

URL: https://kubernetes.io

%description
Command-line utility for deploying a Kubernetes cluster.

%install
install -m 755 -d %{buildroot}%{_bindir}
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/kubelet.service.d/
install -m 755 -d %{buildroot}%{_sysconfdir}/sysconfig/
install -p -m 755 -t %{buildroot}%{_bindir} {kubeadm}
install -p -m 755 -t %{buildroot}%{_sysconfdir}/systemd/system/kubelet.service.d/ {10-kubeadm.conf}
install -p -m 755 -T {kubelet.env} %{buildroot}%{_sysconfdir}/sysconfig/kubelet
mkdir -p %{buildroot}%{_libexecdir}/modules-load.d
mkdir -p %{buildroot}%{_sysctldir}
install -p -m 0644 -t %{buildroot}%{_libexecdir}/modules-load.d/ {kubeadm.conf}
install -p -m 0644 -t %{buildroot}%{_sysctldir} %{50-kubeadm.conf}

%files
%{_bindir}/kubeadm
%{_sysconfdir}/systemd/system/kubelet.service.d/10-kubeadm.conf
%{_sysconfdir}/sysconfig/kubelet
%dir %{_libexecdir}/modules-load.d
%{_libexecdir}/modules-load.d/kubeadm.conf
%{_sysctldir}/50-kubeadm.conf
