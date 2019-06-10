%global KUBE_MAJOR 1
%global KUBE_MINOR 11
%global KUBE_PATCH 0
%global KUBE_VERSION %{KUBE_MAJOR}.%{KUBE_MINOR}.%{KUBE_PATCH}
%global RPM_RELEASE 0
%global ARCH amd64

# This expands a (major, minor, patch) tuple into a single number so that it
# can be compared against other versions. It has the current implementation
# assumption that none of these numbers will exceed 255.
%define semver() (%1 * 256 * 256 + %2 * 256 + %3)
%global KUBE_SEMVER %{semver %{KUBE_MAJOR} %{KUBE_MINOR} %{KUBE_PATCH}}

# no support for %elseif (??)
# https://github.com/rpm-software-management/rpm/issues/311
%if %{KUBE_SEMVER} >= %{semver 1 11 0}
%global CNI_VERSION 0.7.5
%else
%if %{KUBE_SEMVER} >= %{semver 1 9 0}
%global CNI_VERSION 0.6.0
%else
%global CNI_VERSION 0.5.1
%endif
%endif

%if %{KUBE_SEMVER} >= %{semver 1 12 1}
%global CRI_TOOLS_VERSION 1.12.0
%else
%global CRI_TOOLS_VERSION 1.11.1
%endif

Name: kubelet
Version: %{KUBE_VERSION}
Release: %{RPM_RELEASE}
Summary: Container cluster management
License: ASL 2.0

URL: https://kubernetes.io
Source0: https://dl.k8s.io/v%{KUBE_VERSION}/bin/linux/%{ARCH}/kubelet
Source1: kubelet.service
Source2: https://dl.k8s.io/v%{KUBE_VERSION}/bin/linux/%{ARCH}/kubectl
Source3: https://dl.k8s.io/v%{KUBE_VERSION}/bin/linux/%{ARCH}/kubeadm
Source4: 10-kubeadm.conf
%if %{KUBE_SEMVER} < %{semver 1 9 0}
Source5: https://dl.k8s.io/network-plugins/cni-%{ARCH}-0799f5732f2a11b329d9e3d51b9c8f2e3759f2ff.tar.gz
%else
Source5: https://dl.k8s.io/network-plugins/cni-plugins-%{ARCH}-v%{CNI_VERSION}.tgz
%endif
%if %{KUBE_SEMVER} >= %{semver 1 11 0}
Source6: kubelet.env
%endif
Source7: https://github.com/kubernetes-incubator/cri-tools/releases/download/v%{CRI_TOOLS_VERSION}/crictl-v%{CRI_TOOLS_VERSION}-linux-%{ARCH}.tar.gz

BuildRequires: systemd
BuildRequires: curl
Requires: iptables >= 1.4.21
Requires: kubernetes-cni >= %{CNI_VERSION}
Requires: socat
Requires: util-linux
Requires: ethtool
Requires: iproute
Requires: ebtables
Requires: conntrack


%description
The node agent of Kubernetes, the container cluster manager.

%package -n kubernetes-cni

Version: %{CNI_VERSION}
Release: %{RPM_RELEASE}
Summary: Binaries required to provision kubernetes container networking
Requires: kubelet

%description -n kubernetes-cni
Binaries required to provision container networking.

%package -n kubectl

Version: %{KUBE_VERSION}
Release: %{RPM_RELEASE}
Summary: Command-line utility for interacting with a Kubernetes cluster.

%description -n kubectl
Command-line utility for interacting with a Kubernetes cluster.

%package -n kubeadm

Version: %{KUBE_VERSION}
Release: %{RPM_RELEASE}
Summary: Command-line utility for administering a Kubernetes cluster.
Requires: kubelet >= 1.6.0
Requires: kubectl >= 1.6.0
Requires: kubernetes-cni >= 0.7.5
Requires: cri-tools >= 1.11.0

%description -n kubeadm
Command-line utility for administering a Kubernetes cluster.

%package -n cri-tools

Version: %{CRI_TOOLS_VERSION}
Release: %{RPM_RELEASE}
Summary: Command-line utility for interacting with a container runtime.

%description -n cri-tools
Command-line utility for interacting with a container runtime.

%prep
# Assumes the builder has overridden sourcedir to point to directory
# with this spec file. (where these files are stored) Copy them into
# the builddir so they can be installed.
# This is a useful hack for faster Docker builds when working on the spec or
# with locally obtained sources.
#
# Example:
#   spectool -gf kubelet.spec
#   rpmbuild --define "_sourcedir $PWD" -bb kubelet.spec
#

%if %{KUBE_SEMVER} >= %{semver 1 11 0}
ln -s 10-kubeadm-post-1.11.conf %SOURCE4
%else
%if %{KUBE_SEMVER} >= %{semver 1 8 0} && %{KUBE_SEMVER} < %{semver 1 11 0}
ln -s 10-kubeadm-post-1.8.conf %SOURCE4
%else
ln -s 10-kubeadm-pre-1.8.conf %SOURCE4
%endif
%endif

cp -p %SOURCE0 %{_builddir}/
cp -p %SOURCE1 %{_builddir}/
cp -p %SOURCE2 %{_builddir}/
cp -p %SOURCE3 %{_builddir}/
cp -p %SOURCE4 %{_builddir}/
%if %{KUBE_SEMVER} >= %{semver 1 11 0}
cp -p %SOURCE6 %{_builddir}/
%endif
%setup -c -D -T -a 5 -n cni-plugins
%setup -c -a 7 -T -n cri-tools

%install

# The setup macro from prep will make install start in the cni-plugins directory, so cd back to the root.
cd %{_builddir}
install -m 755 -d %{buildroot}%{_unitdir}
install -m 755 -d %{buildroot}%{_unitdir}/kubelet.service.d/
install -m 755 -d %{buildroot}%{_bindir}
install -m 755 -d %{buildroot}%{_sysconfdir}/cni/net.d/
install -m 755 -d %{buildroot}%{_sysconfdir}/kubernetes/manifests/
install -m 755 -d %{buildroot}/var/lib/kubelet/
install -p -m 755 -t %{buildroot}%{_bindir}/ kubelet
install -p -m 755 -t %{buildroot}%{_bindir}/ kubectl
install -p -m 755 -t %{buildroot}%{_bindir}/ kubeadm
install -p -m 644 -t %{buildroot}%{_unitdir}/ kubelet.service
install -p -m 644 -t %{buildroot}%{_unitdir}/kubelet.service.d/ 10-kubeadm.conf
install -p -m 755 -t %{buildroot}%{_bindir}/ cri-tools/crictl

%if %{KUBE_SEMVER} >= %{semver 1 11 0}
install -m 755 -d %{buildroot}%{_sysconfdir}/sysconfig/
install -p -m 644 -T kubelet.env %{buildroot}%{_sysconfdir}/sysconfig/kubelet
%endif


install -m 755 -d %{buildroot}/opt/cni/bin
# bin directory from cni-plugins-%{ARCH}-%{CNI_VERSION}.tgz with a list of cni plugins (among other things)
%if %{KUBE_SEMVER} >= %{semver 1 9 0}
mv cni-plugins/* %{buildroot}/opt/cni/bin/
%else
mv cni-plugins/bin/ %{buildroot}/opt/cni/
%endif

%files
%{_bindir}/kubelet
%{_unitdir}/kubelet.service
%{_sysconfdir}/kubernetes/manifests/

%if %{KUBE_SEMVER} >= %{semver 1 11 0}
%config(noreplace) %{_sysconfdir}/sysconfig/kubelet
%endif

%files -n kubernetes-cni
/opt/cni

%files -n kubectl
%{_bindir}/kubectl

%files -n kubeadm
%{_bindir}/kubeadm
%{_unitdir}/kubelet.service.d/10-kubeadm.conf

%files -n cri-tools
%{_bindir}/crictl

%doc


%changelog
* Thu May 30 2019 Tim Pepper <tpepper@vmware.com>
- Change CNI version check to ">="

* Tue Mar 20 2019 Lubomir I. Ivanov <lubomirivanov@vmware.com>
- Bump CNI version to v0.7.5.

* Tue Sep 25 2018 Chuck Ha <chuck@heptio.com> - 1.12.1
- Bump cri-tools to 1.12.0.

* Fri Jul 13 2018 Lantao Liu <lantaol@google.com> - 1.11.0
- Bump cri-tools to 1.11.1.

* Tue Jun 19 2018 Chuck Ha <chuck@heptio.com> - 1.11.0
- Bump cri-tools to GA version.

* Thu Jun 14 2018 Chuck Ha <chuck@heptio.com> - 1.11.0
- Add a crictl sub-package.

* Fri Jun 8 2018 Chuck Ha <chuck@heptio.com> - 1.11.0
- Bump version and update rpm manifest for kubeadm.

* Fri Dec 15 2017 Anthony Yeh <enisoc@google.com> - 1.9.0
- Release of Kubernetes 1.9.0.

* Thu Oct 19 2017 Di Xu <stephenhsu90@gmail.com>
- Bump CNI version to v0.6.0.

* Fri Sep 29 2017 Jacob Beacham <beacham@google.com> - 1.8.0
- Bump version of kubelet and kubectl to v1.8.0.

* Thu Aug 3 2017 Jacob Beacham <beacham@google.com> - 1.7.3
- Bump version of kubelet and kubectl to v1.7.3.

* Wed Jul 26 2017 Jacob Beacham <beacham@google.com> - 1.7.2
- Bump version of kubelet and kubectl to v1.7.2.

* Fri Jul 14 2017 Jacob Beacham <beacham@google.com> - 1.7.1
- Bump version of kubelet and kubectl to v1.7.1.

* Mon Jun 30 2017 Mike Danese <mikedanese@google.com> - 1.7.0
- Bump version of kubelet and kubectl to v1.7.0.

* Fri May 19 2017 Jacob Beacham <beacham@google.com> - 1.6.4
- Bump version of kubelet and kubectl to v1.6.4.

* Wed May 10 2017 Jacob Beacham <beacham@google.com> - 1.6.3
- Bump version of kubelet and kubectl to v1.6.3.

* Wed Apr 26 2017 Jacob Beacham <beacham@google.com> - 1.6.2
- Bump version of kubelet and kubectl to v1.6.2.

* Mon Apr 3 2017 Mike Danese <mikedanese@google.com> - 1.6.1
- Bump version of kubelet and kubectl to v1.6.1.

* Tue Mar 28 2017 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk>
- Bump CNI version to v0.5.1.

* Wed Mar 15 2017 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk> - 1.6.0
- Bump version of kubelet, kubectl and kubeadm to v1.6.0.

* Tue Dec 13 2016 Mike Danese <mikedanese@google.com> - 1.5.4
- Bump version of kubelet and kubectl to v1.5.4.

* Tue Dec 13 2016 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk> - 1.5.1
- Bump version of kubelet and kubectl to v1.5.1, plus kubeadm to the third stable version

* Tue Dec 6 2016 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk> - 1.5.0-beta.2
- Bump version of kubelet and kubectl

* Wed Nov 16 2016 Alexander Kanevskiy <alexander.kanevskiy@intel.com>
- fix iproute and mount dependencies (#204)

* Sun Nov 6 2016 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk>
- Sync the debs and rpm files; add some kubelet dependencies to the rpm manifest

* Wed Nov 2 2016 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk>
- Bump version of kubeadm to v1.5.0-alpha.2.380+85fe0f1aadf91e

* Fri Oct 21 2016 Ilya Dmitrichenko <errordeveloper@gmail.com> - 1.4.4-0
- Bump version of kubelet and kubectl

* Mon Oct 17 2016 Lucas Käldström <lucas.kaldstrom@hotmail.co.uk> - 1.4.3-0
- Bump version of kubeadm

* Fri Oct 14 2016 Matthew Mosesohn  <mmosesohn@mirantis.com> - 1.4.0-1
- Allow locally built/previously downloaded binaries

* Tue Sep 20 2016 dgoodwin <dgoodwin@redhat.com> - 1.4.0-0
- Add kubectl and kubeadm sub-packages.
- Rename to kubernetes-cni.
- Update versions of CNI.

* Wed Jul 20 2016 dgoodwin <dgoodwin@redhat.com> - 1.3.4-1
- Initial packaging.
