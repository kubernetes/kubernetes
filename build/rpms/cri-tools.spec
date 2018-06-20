Name: cri-tools
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Runtime Interface tools

URL: https://kubernetes.io

%description
Binary to interface with the container runtime.

%prep

%install
install -m 755 -d %{buildroot}%{_bindir}
install -p -m 755 -t %{buildroot}%{_bindir} {crictl}

%files
%{_bindir}/crictl
