FROM fedora:29 AS build

COPY --from=registry.svc.ci.openshift.org/openshift/origin-v4.0:machine-os-content /srv/ /srv/
RUN set -x && yum install -y ostree rpm-ostree yum-utils selinux-policy-targeted && \
    curl http://base-4-3-rhel8.ocp.svc > /etc/yum.repos.d/rhel8.repo && \
    commit=$( find /srv -name *.commit | sed -Ee 's|.*objects/(.+)/(.+)\.commit|\1\2|' | head -1 ) && \
    mkdir /tmp/working && cd /tmp/working && \
    rpm-ostree db list --repo /srv/repo $commit > /tmp/packages && \
    PACKAGES=(openshift-hyperkube) && \
    yumdownloader -y --disablerepo=* --enablerepo=built --destdir=/tmp/rpms "${PACKAGES[@]}" && \
    if ! grep -q cri-o /tmp/packages; then yumdownloader -y --disablerepo=* --enablerepo=rhel-8* --destdir=/tmp/rpms cri-o cri-tools; fi && \
    if ! grep -q machine-config-daemon /tmp/packages; then yumdownloader -y --disablerepo=* --enablerepo=rhel-8* --destdir=/tmp/rpms machine-config-daemon; fi && \
    ls /tmp/rpms/ && (cd /tmp/rpms/ && ls ${PACKAGES[@]/%/*}) && \
    for i in $(find /tmp/rpms/ -name *.rpm); do echo "Extracting $i ..."; rpm2cpio $i | cpio -div; done && \
    if [[ -d etc ]]; then mv etc usr/; fi && \
    mkdir -p /tmp/tmprootfs/etc && \
    ostree --repo=/srv/repo checkout -U $commit --subpath /usr/etc/selinux /tmp/tmprootfs/etc/selinux && \
    ostree --repo=/srv/repo commit --parent=$commit --tree=ref=$commit --tree=dir=. \
        --selinux-policy /tmp/tmprootfs \
        -s "origin-ci-dev overlay RPMs" --branch=origin-ci-dev

FROM scratch
COPY --from=build /srv/ /srv/
