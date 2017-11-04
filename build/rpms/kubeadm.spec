Name: kubeadm
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Cluster Manager - Kubernetes Cluster Bootstrapping Tool
Requires: kubelet >= 1.8.0
Requires: kubectl >= 1.8.0
Requires: kubernetes-cni >= 0.5.1

URL: https://kubernetes.io

%description
Command-line utility for deploying a Kubernetes cluster.

%install
install -m 755 -d %{buildroot}%{_bindir}
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/kubelet.service.d/
install -p -m 755 -t %{buildroot}%{_bindir} kubeadm
install -p -m 755 -t %{buildroot}%{_sysconfdir}/systemd/system/kubelet.service.d/ 10-kubeadm.conf

%files
%{_bindir}/kubeadm
%{_sysconfdir}/systemd/system/kubelet.service.d/10-kubeadm.conf
