FROM registry.svc.ci.openshift.org/ocp/builder:golang-1.12 AS builder
WORKDIR /go/src/github.com/openshift/origin
COPY . .
RUN make build WHAT=cmd/openshift-tests; \
    mkdir -p /tmp/build; \
    cp /go/src/github.com/openshift/origin/_output/local/bin/linux/$(go env GOARCH)/openshift-tests /tmp/build/openshift-tests

FROM registry.svc.ci.openshift.org/ocp/4.2:cli
COPY --from=builder /tmp/build/openshift-tests /usr/bin/
RUN yum install --setopt=tsflags=nodocs -y git gzip util-linux && yum clean all && rm -rf /var/cache/yum/* && \
    git config --system user.name test && \
    git config --system user.email test@test.com && \
    chmod g+w /etc/passwd
LABEL io.k8s.display-name="OpenShift End-to-End Tests" \
      io.k8s.description="OpenShift is a platform for developing, building, and deploying containerized applications." \
      io.openshift.tags="openshift,tests,e2e"
