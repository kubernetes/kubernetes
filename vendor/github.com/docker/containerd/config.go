package containerd

type Process struct {
	Args []string
	Env  []string
	Cwd  string
	Uid  int
	Gid  int
	TTY  bool
}

type Config struct {
	Process    Process
	Hostname   string
	Domain     string
	Labels     map[string]string
	StopSignal int
}
