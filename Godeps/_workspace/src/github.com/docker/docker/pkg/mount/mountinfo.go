package mount

type MountInfo struct {
	Id, Parent, Major, Minor         int
	Root, Mountpoint, Opts, Optional string
	Fstype, Source, VfsOpts          string
}
