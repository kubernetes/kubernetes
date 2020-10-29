# this should match whatever we have in https://github.com/openshift/release/blob/master/ci-operator/config/openshift/kubernetes/openshift-kubernetes-master.yaml
# - context_dir: openshift-hack/images/os/
#   from: base
#   inputs:
#     base-machine-with-rpms:
#       as:
#       - centos:stream9 <-- here
#     machine-os-content-base:
#       as:
#       - registry.svc.ci.openshift.org/openshift/origin-v4.0:machine-os-content
#   to: machine-os-content
FROM centos:stream9 As build

# the registry is defined here:
# https://github.com/openshift/release/blob/b45a09d248b8cdb8fe3bf5f3cfa0b4fee57d04c8/ci-operator/config/openshift/kubernetes/openshift-kubernetes-release-4.10.yaml#L68
COPY --from=registry.svc.ci.openshift.org/openshift/origin-v4.0:machine-os-content /srv/ /srv/
COPY --from=registry.svc.ci.openshift.org/openshift/origin-v4.0:machine-os-content /extensions/ /extensions/
WORKDIR /
COPY install.sh .
RUN ./install.sh

FROM scratch
COPY --from=build /srv/ /srv/
COPY --from=build /extensions/ /extensions/

LABEL io.openshift.build.version-display-names="machine-os=rhcos image for testing openshift kubernetes kubelet only- if you see this outside of PR runs for openshift kubernetes- you found an urgent blocker bug"
LABEL io.openshift.build.versions="machine-os=1.2.3-testing-if-you-see-this-outside-of-PR-runs-for-openshift-kubernetes-you-found-an-urgent-blocker-bug"
