# contrib

The `contrib` directory contains packages that do not belong in the core containerd packages but still contribute to overall containerd usability.

Package such as Apparmor or Selinux are placed in `contrib` because they are platform dependent and often require higher level tools and profiles to work.

Packaging and other built tools can be added to `contrib` to aid in packaging containerd for various distributions.

## Testing

Code in the `contrib` directory may or may not have been tested in the normal test pipeline for core components.
