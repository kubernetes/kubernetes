package apparmor

import (
	"io"
	"os"
	"text/template"
)

type data struct {
	Name         string
	Imports      []string
	InnerImports []string
}

const baseTemplate = `
{{range $value := .Imports}}
{{$value}}
{{end}}

profile {{.Name}} flags=(attach_disconnected,mediate_deleted) {
{{range $value := .InnerImports}}
  {{$value}}
{{end}}

  network,
  capability,
  file,
  umount,

  mount fstype=tmpfs,
  mount fstype=mqueue,
  mount fstype=fuse.*,
  mount fstype=binfmt_misc -> /proc/sys/fs/binfmt_misc/,
  mount fstype=efivarfs -> /sys/firmware/efi/efivars/,
  mount fstype=fusectl -> /sys/fs/fuse/connections/,
  mount fstype=securityfs -> /sys/kernel/security/,
  mount fstype=debugfs -> /sys/kernel/debug/,
  mount fstype=proc -> /proc/,
  mount fstype=sysfs -> /sys/,

  deny @{PROC}/sys/fs/** wklx,
  deny @{PROC}/sysrq-trigger rwklx,
  deny @{PROC}/mem rwklx,
  deny @{PROC}/kmem rwklx,
  deny @{PROC}/sys/kernel/[^s][^h][^m]* wklx,
  deny @{PROC}/sys/kernel/*/** wklx,

  deny mount options=(ro, remount) -> /,
  deny mount fstype=debugfs -> /var/lib/ureadahead/debugfs/,
  deny mount fstype=devpts,

  deny /sys/[^f]*/** wklx,
  deny /sys/f[^s]*/** wklx,
  deny /sys/fs/[^c]*/** wklx,
  deny /sys/fs/c[^g]*/** wklx,
  deny /sys/fs/cg[^r]*/** wklx,
  deny /sys/firmware/efi/efivars/** rwklx,
  deny /sys/kernel/security/** rwklx,
}
`

func generateProfile(out io.Writer) error {
	compiled, err := template.New("apparmor_profile").Parse(baseTemplate)
	if err != nil {
		return err
	}
	data := &data{
		Name: "docker-default",
	}
	if tuntablesExists() {
		data.Imports = append(data.Imports, "#include <tunables/global>")
	} else {
		data.Imports = append(data.Imports, "@{PROC}=/proc/")
	}
	if abstrctionsEsists() {
		data.InnerImports = append(data.InnerImports, "#include <abstractions/base>")
	}
	if err := compiled.Execute(out, data); err != nil {
		return err
	}
	return nil
}

// check if the tunables/global exist
func tuntablesExists() bool {
	_, err := os.Stat("/etc/apparmor.d/tunables/global")
	return err == nil
}

// check if abstractions/base exist
func abstrctionsEsists() bool {
	_, err := os.Stat("/etc/apparmor.d/abstractions/base")
	return err == nil
}
