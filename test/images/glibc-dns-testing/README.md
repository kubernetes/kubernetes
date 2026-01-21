# glibc-dns-testing

A glibc-based test image for verifying DNS resolution behavior in Kubernetes.

## Purpose

This image exists to test that **glibc-based programs** can resolve Service DNS
names correctly. It complements `agnhost` (which uses Alpine/musl libc) to ensure
Kubernetes DNS works correctly regardless of which C library the application uses.

## Why glibc Testing Matters

DNS resolution behavior differs between glibc (used by Debian, Ubuntu, RHEL) and
musl libc (used by Alpine Linux):

- **glibc** queries nameservers sequentially; **musl** queries in parallel
- glibc and musl handle `ndots` and search domains differently
- The `hostname` command behaves differently between glibc and musl (see issue #134737)

Many production workloads use glibc-based distributions, so testing DNS resolution
with glibc ensures compatibility with:
- Java applications
- Python/Ruby/PHP applications
- Most enterprise Linux distributions

## History

This image was originally named `jessie-dnsutils` (based on Debian Jessie). It was
renamed to `glibc-dns-testing` to better reflect its actual purpose: testing glibc
DNS resolution behavior, not any specific Debian version.

See issue [#10161](https://github.com/kubernetes/kubernetes/issues/10161) for the
original discussion about libc DNS differences that motivated creating this image.

## Contents

- **Base**: Debian Bookworm (glibc-based)
- **dnsutils**: Provides `dig`, `nslookup`, and `host` commands
- **CoreDNS**: Embedded DNS server for testing

## Usage

This image is used in Kubernetes E2E tests alongside `agnhost` to verify DNS
resolution works correctly with both musl and glibc resolvers.

**Note**: This image is for test purposes only, not for production use.
