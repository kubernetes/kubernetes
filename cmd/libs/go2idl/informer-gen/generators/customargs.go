package generators

import "github.com/spf13/pflag"

type CustomArgs struct {
	VersionedClientSetPackage string
	InternalClientSetPackage  string
	ListersPackage            string
}

func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&ca.InternalClientSetPackage, "internal-clientset-package", ca.InternalClientSetPackage, "the full package name for the internal clientset to use")
	fs.StringVar(&ca.VersionedClientSetPackage, "versioned-clientset-package", ca.VersionedClientSetPackage, "the full package name for the versioned clientset to use")
	fs.StringVar(&ca.ListersPackage, "listers-package", ca.ListersPackage, "the full package name for the listers to use")
}
