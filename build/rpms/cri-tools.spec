Name: cri-tools
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Runtime Interface tools

URL: https://kubernetes.io

%description
Binaries to interface with the container runtime.

%prep
# TODO(chuckha): update this to use %{version} when the dash is removed from the release
tar -xzf {crictl-v1.0.0-beta.1-linux-amd64.tar.gz}

%install
install -m 755 -d %{buildroot}%{_bindir}
install -p -m 755 -t %{buildroot}%{_bindir} crictl

%files
%{_bindir}/crictl
