# nautilus

A basic file server that serves an image. The underlying base image does most
of the work but we load a specific image when building this one so that we
can test that a pod's image really changed. Often used to contrast against
the kitten image which loads/serves a different image.